#!/usr/bin/env python3
"""Compare AttentionBlock before and after the explicit-SDPA change."""

from __future__ import annotations

import argparse
import gc
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import AttentionBlock
except ImportError:  # Allow direct execution from the repository root.
    from layers import AttentionBlock


class ExperimentalSDPAAttention(nn.Module):
    """Explicit SDPA implementation kept local to this benchmark."""

    def __init__(self, dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = dim
        self.num_heads = nhead
        self.head_dim = dim // nhead
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_num, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(
            batch_size, token_num, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        dropout_p = self.dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p
        )
        x = x.transpose(1, 2).reshape(batch_size, token_num, self.embed_dim)
        return self.out_proj(x)


class ExperimentalSDPAAttentionBlock(AttentionBlock):
    """Experimental explicit-SDPA block; production code remains unchanged."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        mlp_ratio: float = 4.0,
        nhead: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        pre_norm: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            dim_out=dim_out,
            mlp_ratio=mlp_ratio,
            nhead=nhead,
            dropout=dropout,
            drop_path=drop_path,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre_norm=pre_norm,
        )
        self.token_mixer = ExperimentalSDPAAttention(dim_out, nhead, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != self.dim_out:
            x = self.proj(x)
        x_norm = self.norm1(x)
        attention = self.token_mixer(x_norm)
        x = x + self.drop_path(attention)
        return x + self.drop_path(self.mlp(self.norm2(x)))


@dataclass
class Result:
    name: str
    milliseconds: float
    iterations_per_second: float
    memory_mib: float
    reserved_mib: float | None


class RSSSampler:
    """Best-effort Linux RSS sampler for CPU-only runs."""

    def __init__(self) -> None:
        self.peak_bytes = self.read()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    @staticmethod
    def read() -> int:
        try:
            with open("/proc/self/status", encoding="utf-8") as status:
                for line in status:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self.read())
            self._stop.wait(0.0005)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join()
        return max(self.peak_bytes, self.read())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_blocks(
    dim: int,
    heads: int,
    dropout: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, AttentionBlock]:
    before = AttentionBlock(
        dim,
        dim,
        nhead=heads,
        dropout=dropout,
        drop_path=0.0,
        pre_norm=True,
    ).to(device=device, dtype=dtype)
    after = ExperimentalSDPAAttentionBlock(
        dim,
        dim,
        nhead=heads,
        dropout=dropout,
        drop_path=0.0,
        pre_norm=True,
    ).to(device=device, dtype=dtype)

    # The experimental module retains the production module's state keys.
    after.load_state_dict(before.state_dict(), strict=True)
    return {
        "before: MultiheadAttention": before,
        "after: explicit SDPA": after,
    }


def compare_outputs_and_gradients(
    blocks: dict[str, AttentionBlock], x: torch.Tensor
) -> None:
    """Compare deterministic eval outputs and backward gradients."""
    before, after = blocks.values()
    before.eval()
    after.eval()

    before_input = x.detach().clone().requires_grad_(True)
    after_input = x.detach().clone().requires_grad_(True)
    before_output = before(before_input)
    after_output = after(after_input)

    output_error = (before_output - after_output).abs().max().item()
    before_output.float().square().mean().backward()
    after_output.float().square().mean().backward()
    input_grad_error = (before_input.grad - after_input.grad).abs().max().item()
    before_parameters = dict(before.named_parameters())
    after_parameters = dict(after.named_parameters())
    if before_parameters.keys() != after_parameters.keys():
        raise RuntimeError("before/after parameter names do not match")
    parameter_grad_error = max(
        (before_parameters[name].grad - after_parameters[name].grad)
        .abs()
        .max()
        .item()
        for name in before_parameters
    )
    print("Correctness check (eval mode, dropout disabled by eval):")
    print(f"  max output absolute error:         {output_error:.3e}")
    print(f"  max input-gradient absolute error: {input_grad_error:.3e}")
    print(f"  max parameter-gradient error:      {parameter_grad_error:.3e}")

    for block in blocks.values():
        block.zero_grad(set_to_none=True)


def make_step(
    block: nn.Module, x: torch.Tensor, training: bool
) -> Callable[[], None]:
    if training:
        def step() -> None:
            block.zero_grad(set_to_none=True)
            x.grad = None
            block(x).float().square().mean().backward()
        return step

    def step() -> None:
        with torch.inference_mode():
            block(x)
    return step


def benchmark(
    name: str,
    block: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iterations: int,
    training: bool,
) -> Result:
    device = x.device
    block.train(training)
    step = make_step(block, x, training)

    for _ in range(warmup):
        step()
    synchronize(device)

    if device.type == "cuda":
        block.zero_grad(set_to_none=True)
        x.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        base_allocated = torch.cuda.memory_allocated(device)
        base_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        synchronize(device)
        elapsed = time.perf_counter() - start
        memory = torch.cuda.max_memory_allocated(device) - base_allocated
        reserved = torch.cuda.max_memory_reserved(device) - base_reserved
    else:
        gc.collect()
        base_rss = RSSSampler.read()
        sampler = RSSSampler()
        sampler.start()
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        elapsed = time.perf_counter() - start
        memory = max(0, sampler.stop() - base_rss)
        reserved = None

    return Result(
        name=name,
        milliseconds=elapsed * 1000.0 / iterations,
        iterations_per_second=iterations / elapsed,
        memory_mib=memory / 1024**2,
        reserved_mib=None if reserved is None else reserved / 1024**2,
    )


def print_results(results: list[Result], device: torch.device) -> None:
    fastest = min(result.milliseconds for result in results)
    if device.type == "cuda":
        print(
            f"{'implementation':<29} {'ms/iter':>10} {'iter/s':>10} "
            f"{'peak alloc':>12} {'peak reserv':>12} {'relative':>10}"
        )
    else:
        print(
            f"{'implementation':<29} {'ms/iter':>10} {'iter/s':>10} "
            f"{'peak RSS*':>12} {'relative':>10}"
        )
    for result in results:
        relative = result.milliseconds / fastest
        common = (
            f"{result.name:<29} {result.milliseconds:>10.3f} "
            f"{result.iterations_per_second:>10.1f} "
            f"{result.memory_mib:>9.1f} MiB"
        )
        if result.reserved_mib is None:
            print(f"{common} {relative:>9.2f}x")
        else:
            print(
                f"{common} {result.reserved_mib:>9.1f} MiB "
                f"{relative:>9.2f}x"
            )


def parse_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--training", action="store_true", help="benchmark forward + backward"
    )
    parser.add_argument(
        "--stage", choices=("all", "1", "2", "3"), default="all"
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Fix the CUDA/driver setup or pass --device cpu."
        )
    if min(args.batch_size, args.heads, args.warmup, args.iterations) <= 0:
        parser.error("batch size, heads, warmup and iterations must be positive")

    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype, device)
    stages = {"1": (21, 256), "2": (84, 128), "3": (336, 64)}
    selected = (
        stages.items()
        if args.stage == "all"
        else [(args.stage, stages[args.stage])]
    )

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    print(f"PyTorch: {torch.__version__}")
    print(
        f"Device: {device}; dtype: {dtype}; "
        f"benchmark mode: {'train' if args.training else 'eval'}"
    )
    print(
        f"Batch: {args.batch_size}; warmup: {args.warmup}; "
        f"iterations: {args.iterations}"
    )

    for stage, (tokens, dim) in selected:
        print(f"\nStage {stage}: input [{args.batch_size}, {tokens}, {dim}], heads={args.heads}")
        x = torch.randn(
            args.batch_size,
            tokens,
            dim,
            device=device,
            dtype=dtype,
            requires_grad=args.training,
        )
        blocks = make_blocks(
            dim, args.heads, args.dropout, device=device, dtype=dtype
        )
        compare_outputs_and_gradients(blocks, x)
        results = [
            benchmark(
                name,
                block,
                x,
                warmup=args.warmup,
                iterations=args.iterations,
                training=args.training,
            )
            for name, block in blocks.items()
        ]
        print_results(results, device)
        del blocks, x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cpu":
        print("\n* CPU peak RSS is sampled and approximate; CUDA peak counters are exact.")


if __name__ == "__main__":
    if "OMP_NUM_THREADS" in os.environ:
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    main()
