#!/usr/bin/env python3
"""Load and call only the ViTPose++-B MoE backbone.

This is a dependency-light extraction of the official ViTPose ``ViTMoE``
backbone.  It intentionally does not construct a neck, heatmap head, decoder,
or a 2D keypoint estimator.

Official reference:
https://github.com/ViTAE-Transformer/ViTPose/blob/main/mmpose/models/backbones/vit_moe.py

Minimal integration::

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ViTPosePlusBackbone(config_path, checkpoint_path,
                                  freeze=False, device=device)
    images = torch.randn(2, 3, 256, 192, device=encoder.device)
    feature_map = encoder(images, dataset_source=5)       # [2, 768, 16, 12]
    tokens = encoder.feature_map_to_tokens(feature_map)   # [2, 192, 768]
    pred_3d = graphormer(tokens)

If Graphormer's hidden size is not 768, insert a trainable Linear token adapter
or a 1x1 Conv2d map adapter.  Keep the 16x12 grid long enough to construct a 2D
positional encoding; flattening alone does not encode patch coordinates.
"""

from __future__ import annotations

import argparse
import json
import pickle
import runpy
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PathLike = Union[str, Path]


def _to_2tuple(value: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(f"Expected a pair, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _load_config(path: PathLike) -> Dict[str, Any]:
    """Load a local Python/JSON/YAML config without importing MMPose."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".py":
        namespace = runpy.run_path(str(path))
        return {key: value for key, value in namespace.items() if not key.startswith("__")}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional format
            raise RuntimeError("Reading YAML configs requires the already-optional PyYAML package") from exc
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise TypeError(f"YAML config must contain a mapping, got {type(loaded).__name__}")
        return loaded
    raise ValueError(f"Unsupported config suffix {suffix!r}; use .py, .json, .yaml or .yml")


def _extract_backbone_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    model = config.get("model", config)
    if not isinstance(model, Mapping):
        raise TypeError("Config field 'model' must be a mapping")
    backbone = model.get("backbone", model)
    if not isinstance(backbone, Mapping):
        raise TypeError("Config field 'model.backbone' must be a mapping")
    return dict(backbone)


def _validate_vitpose_plus_base_config(config: Mapping[str, Any]) -> None:
    expected = {
        "type": "ViTMoE",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "patch_size": 16,
        "num_expert": 6,
        "part_features": 192,
    }
    mismatches = []
    for key, expected_value in expected.items():
        actual = config.get(key)
        if actual != expected_value:
            mismatches.append(f"{key}={actual!r} (expected {expected_value!r})")
    if mismatches:
        raise ValueError(
            "The supplied config is not the released multi-task ViTPose++-B configuration: "
            + ", ".join(mismatches)
        )


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to the official timm helper."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PatchEmbed(nn.Module):
    """Image-to-patch projection matching official ViTPose padding semantics."""

    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: Union[int, Sequence[int]],
        in_chans: int,
        embed_dim: int,
        ratio: int = 1,
    ) -> None:
        super().__init__()
        self.img_size = _to_2tuple(img_size)
        self.patch_size = _to_2tuple(patch_size)
        self.ratio = int(ratio)
        if self.ratio < 1:
            raise ValueError("ratio must be >= 1")
        self.patch_shape = (
            int(self.img_size[0] // self.patch_size[0] * self.ratio),
            int(self.img_size[1] // self.patch_size[1] * self.ratio),
        )
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        stride = (self.patch_size[0] // self.ratio, self.patch_size[1] // self.ratio)
        padding = 4 + 2 * (self.ratio // 2 - 1)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        features = self.proj(images)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return tokens, (height, width)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(output))


class MoEMlp(nn.Module):
    """ViTPose++ shared/expert MLP with explicit per-sample routing."""

    def __init__(
        self,
        num_expert: int,
        in_features: int,
        hidden_features: int,
        part_features: int,
        drop: float,
    ) -> None:
        super().__init__()
        if not 0 < part_features < in_features:
            raise ValueError("part_features must be between zero and embed_dim")
        self.num_expert = int(num_expert)
        self.part_features = int(part_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features - part_features)
        self.experts = nn.ModuleList(
            nn.Linear(hidden_features, part_features) for _ in range(num_expert)
        )
        self.drop = nn.Dropout(drop)

    def forward(self, tokens: Tensor, dataset_source: Tensor) -> Tensor:
        hidden = self.act(self.fc1(tokens))
        shared = self.fc2(hidden)

        # The official implementation evaluates every expert and masks the
        # results.  Sparse index_copy is numerically equivalent while avoiding
        # five unused expert projections for a single-domain batch.
        expert_output = hidden.new_zeros((*hidden.shape[:-1], self.part_features))
        for expert_index, expert in enumerate(self.experts):
            batch_indices = torch.nonzero(
                dataset_source == expert_index, as_tuple=False
            ).flatten()
            if batch_indices.numel() == 0:
                continue
            selected = expert(hidden.index_select(0, batch_indices))
            expert_output = expert_output.index_copy(0, batch_indices, selected)
        return self.drop(torch.cat((shared, expert_output), dim=-1))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        num_expert: int,
        part_features: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MoEMlp(
            num_expert=num_expert,
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            part_features=part_features,
            drop=drop,
        )

    def forward(self, tokens: Tensor, dataset_source: Tensor) -> Tensor:
        tokens = tokens + self.drop_path(self.attn(self.norm1(tokens)))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens), dataset_source))
        return tokens


class ViTMoE(nn.Module):
    """Backbone-only ViTPose++ Vision Transformer with six MLP experts."""

    def __init__(self, config: Mapping[str, Any], interpolate_pos_encoding: bool = False) -> None:
        super().__init__()
        _validate_vitpose_plus_base_config(config)

        self.img_size = _to_2tuple(config["img_size"])
        self.embed_dim = int(config["embed_dim"])
        self.depth = int(config["depth"])
        self.num_expert = int(config["num_expert"])
        self.interpolate_pos_encoding = bool(interpolate_pos_encoding)
        norm_eps = float(config.get("norm_eps", 1e-6))

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=config["patch_size"],
            in_chans=int(config.get("in_chans", 3)),
            embed_dim=self.embed_dim,
            ratio=int(config.get("ratio", 1)),
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim)
        )
        drop_path_rates = torch.linspace(
            0, float(config.get("drop_path_rate", 0.0)), self.depth
        ).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim=self.embed_dim,
                num_heads=int(config["num_heads"]),
                mlp_ratio=float(config.get("mlp_ratio", 4.0)),
                qkv_bias=bool(config.get("qkv_bias", False)),
                drop=float(config.get("drop_rate", 0.0)),
                attn_drop=float(config.get("attn_drop_rate", 0.0)),
                drop_path=drop_path_rates[index],
                num_expert=self.num_expert,
                part_features=int(config["part_features"]),
                norm_eps=norm_eps,
            )
            for index in range(self.depth)
        )
        self.last_norm = (
            nn.LayerNorm(self.embed_dim, eps=norm_eps)
            if bool(config.get("last_norm", True))
            else nn.Identity()
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _position_embedding(self, height: int, width: int) -> Tensor:
        canonical_height, canonical_width = self.patch_embed.patch_shape
        if (height, width) == (canonical_height, canonical_width):
            return self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        if not self.interpolate_pos_encoding:
            expected = self.img_size
            raise ValueError(
                f"Input produced a {height}x{width} patch grid, but the checkpoint is "
                f"trained for {canonical_height}x{canonical_width} from HxW={expected}. "
                "Resize/crop to the configured size or set interpolate_pos_encoding=True."
            )
        patch_pos = self.pos_embed[:, 1:].reshape(
            1, canonical_height, canonical_width, self.embed_dim
        )
        patch_pos = F.interpolate(
            patch_pos.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, height * width, self.embed_dim)
        return patch_pos + self.pos_embed[:, :1]

    def _prepare_dataset_source(
        self,
        dataset_source: Union[int, Tensor, Sequence[int]],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        source = torch.as_tensor(dataset_source, device=device)
        if source.ndim == 0:
            source = source.expand(batch_size)
        else:
            source = source.reshape(-1)
        if source.numel() != batch_size:
            raise ValueError(
                f"dataset_source needs one index per sample; got {source.numel()} for batch {batch_size}"
            )
        if source.is_floating_point() and not torch.equal(source, source.round()):
            raise ValueError("dataset_source values must be integer expert indices")
        source = source.to(dtype=torch.long)
        if bool(((source < 0) | (source >= self.num_expert)).any()):
            raise ValueError(f"dataset_source must be in [0, {self.num_expert - 1}], got {source.tolist()}")
        return source

    def forward(
        self,
        images: Tensor,
        dataset_source: Union[int, Tensor, Sequence[int]],
    ) -> Tensor:
        batch_size = images.shape[0]
        source = self._prepare_dataset_source(dataset_source, batch_size, images.device)
        tokens, (height, width) = self.patch_embed(images)
        tokens = tokens + self._position_embedding(height, width)
        for block in self.blocks:
            tokens = block(tokens, source)
        tokens = self.last_norm(tokens)
        return tokens.transpose(1, 2).reshape(
            batch_size, self.embed_dim, height, width
        ).contiguous()


def _extract_state_dict(checkpoint: Any) -> Tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Checkpoint must be a parameter mapping or contain one, got {type(checkpoint).__name__}"
        )
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value, key
    return checkpoint, "<root>"


def _load_checkpoint_file(path: PathLike) -> Any:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(str(path), map_location="cpu")
    except Exception as torch_error:
        if path.suffix.lower() != ".pkl":
            raise RuntimeError(f"torch.load failed for {path}: {torch_error}") from torch_error
        try:
            with path.open("rb") as stream:
                return pickle.load(stream)
        except Exception as pickle_error:
            raise RuntimeError(
                f"Neither torch.load nor pickle.load could read {path}. "
                f"torch.load: {torch_error}; pickle.load: {pickle_error}"
            ) from pickle_error


def _strip_wrapper_prefixes(key: str) -> Tuple[str, bool]:
    parts = key.split(".")
    while parts and parts[0] in {"module", "model", "state_dict"}:
        parts.pop(0)
    marked_backbone = bool(parts and parts[0] == "backbone")
    if marked_backbone:
        parts.pop(0)
    return ".".join(parts), marked_backbone


class ViTPosePlusBackbone(nn.Module):
    """Build, load and call only the released ViTPose++-B backbone.

    ``forward`` returns one NCHW feature map.  For the canonical 256x192 input,
    its shape is [B, 768, 16, 12].  The 192 spatial cells are final-layer ViT
    patch embeddings; no CLS token, heatmap or keypoint prediction is returned.
    """

    def __init__(
        self,
        config_path: PathLike,
        checkpoint_path: PathLike,
        freeze: bool = False,
        device: Union[str, torch.device] = "cuda",
        min_load_ratio: float = 0.90,
        interpolate_pos_encoding: bool = False,
        output_index: int = -1,
    ) -> None:
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.config = _load_config(self.config_path)
        self.backbone_config = _extract_backbone_config(self.config)
        _validate_vitpose_plus_base_config(self.backbone_config)
        self.backbone = ViTMoE(
            self.backbone_config,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        self.out_channels = int(self.backbone_config["embed_dim"])
        self.output_index = int(output_index)
        self.default_dataset_source = int(self.config.get("default_dataset_source", 5))
        self.dataset_experts = dict(self.config.get("dataset_experts", {}))
        self.preprocess_config = dict(self.config.get("preprocess_cfg", {}))
        self.last_load_report = self._load_backbone_weights(min_load_ratio=min_load_ratio)

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            print(f"[device] CUDA was requested but is unavailable; falling back to CPU")
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.to(self.device)
        self.set_frozen(freeze)

    def _load_backbone_weights(self, min_load_ratio: float) -> Dict[str, Any]:
        if not 0.0 < min_load_ratio <= 1.0:
            raise ValueError("min_load_ratio must be in (0, 1]")
        checkpoint = _load_checkpoint_file(self.checkpoint_path)
        state_dict, container = _extract_state_dict(checkpoint)
        expected = self.backbone.state_dict()
        filtered: Dict[str, Tensor] = {}
        mismatched_shapes: List[str] = []
        unexpected_backbone: List[str] = []
        ignored_non_backbone = 0

        for original_key, value in state_dict.items():
            if not isinstance(original_key, str):
                ignored_non_backbone += 1
                continue
            candidate, marked_backbone = _strip_wrapper_prefixes(original_key)
            if not torch.is_tensor(value):
                try:
                    value = torch.as_tensor(value)
                except (TypeError, ValueError):
                    ignored_non_backbone += 1
                    continue
            if candidate not in expected:
                if marked_backbone:
                    unexpected_backbone.append(original_key)
                else:
                    ignored_non_backbone += 1
                continue
            if tuple(value.shape) != tuple(expected[candidate].shape):
                mismatched_shapes.append(
                    f"{original_key}: checkpoint{tuple(value.shape)} != model{tuple(expected[candidate].shape)}"
                )
                continue
            if candidate in filtered:
                raise RuntimeError(f"Multiple checkpoint keys map to backbone key {candidate!r}")
            filtered[candidate] = value

        incompatible = self.backbone.load_state_dict(filtered, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys) + unexpected_backbone
        loaded_keys = len(filtered)
        total_keys = len(expected)
        loaded_numel = sum(expected[key].numel() for key in filtered)
        total_numel = sum(value.numel() for value in expected.values())
        key_ratio = loaded_keys / total_keys
        numel_ratio = loaded_numel / total_numel

        print(f"[checkpoint] path={self.checkpoint_path}")
        print(f"[checkpoint] top-level container={container!r}, entries={len(state_dict)}")
        print(
            f"[checkpoint] backbone loaded: {loaded_keys}/{total_keys} tensors "
            f"({key_ratio:.2%}), {loaded_numel}/{total_numel} values ({numel_ratio:.2%})"
        )
        print(
            f"[checkpoint] ignored non-backbone tensors/entries={ignored_non_backbone}; "
            "these are intentionally not passed to load_state_dict"
        )
        print(f"[checkpoint] missing keys ({len(missing)}): {missing or 'none'}")
        print(f"[checkpoint] unexpected backbone keys ({len(unexpected)}): {unexpected or 'none'}")
        print(f"[checkpoint] shape mismatches ({len(mismatched_shapes)}): {mismatched_shapes or 'none'}")

        if key_ratio < min_load_ratio or numel_ratio < min_load_ratio:
            raise RuntimeError(
                "Too little of the ViTPose++-B backbone was loaded: "
                f"key_ratio={key_ratio:.2%}, numel_ratio={numel_ratio:.2%}, "
                f"required>={min_load_ratio:.2%}. Check config/checkpoint pairing and prefixes."
            )
        if mismatched_shapes:
            raise RuntimeError("Backbone checkpoint has shape mismatches: " + "; ".join(mismatched_shapes))

        representative_keys = [
            "pos_embed",
            "patch_embed.proj.weight",
            "blocks.0.mlp.experts.5.weight",
            "blocks.11.attn.qkv.weight",
            "last_norm.weight",
        ]
        verification = {}
        loaded_state = self.backbone.state_dict()
        for key in representative_keys:
            if key in filtered:
                max_diff = float((loaded_state[key].cpu() - filtered[key].cpu()).abs().max())
                verification[key] = max_diff
        print(f"[checkpoint] representative max_abs_diff after copy: {verification}")
        if any(value != 0.0 for value in verification.values()):
            raise RuntimeError("At least one representative backbone tensor was not copied exactly")

        return {
            "container": container,
            "loaded_keys": loaded_keys,
            "total_keys": total_keys,
            "key_ratio": key_ratio,
            "loaded_numel": loaded_numel,
            "total_numel": total_numel,
            "numel_ratio": numel_ratio,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": mismatched_shapes,
            "ignored_non_backbone": ignored_non_backbone,
            "representative_max_abs_diff": verification,
        }

    def set_frozen(self, freeze: bool = True) -> "ViTPosePlusBackbone":
        self._frozen = bool(freeze)
        self.backbone.requires_grad_(not self._frozen)
        if self._frozen:
            self.backbone.eval()
        else:
            self.backbone.train(self.training)
        return self

    def train(self, mode: bool = True) -> "ViTPosePlusBackbone":
        """Set mode; a frozen backbone always remains in eval mode.

        For a trainable backbone, ``train()`` enables stochastic depth and
        ``eval()`` disables it.  LayerNorm has no running statistics.
        """
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.backbone.eval()
        return self

    @staticmethod
    def _normalize_output(output: Any, output_index: int = -1) -> Tensor:
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)):
            tensors = [item for item in output if torch.is_tensor(item)]
            if not tensors:
                raise TypeError("Backbone returned a tuple/list without any Tensor")
            try:
                return tensors[output_index]
            except IndexError as exc:
                raise IndexError(
                    f"output_index={output_index} is invalid for {len(tensors)} tensor outputs"
                ) from exc
        raise TypeError(
            f"Backbone must return Tensor, tuple or list, got {type(output).__name__}"
        )

    def forward(
        self,
        images: Tensor,
        dataset_source: Optional[Union[int, Tensor, Sequence[int]]] = None,
    ) -> Tensor:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("images must be a Tensor shaped [B, 3, H, W]")
        if images.shape[1] != int(self.backbone_config.get("in_chans", 3)):
            raise ValueError(f"Expected 3 image channels, got shape {tuple(images.shape)}")
        if not images.is_floating_point():
            raise TypeError("forward expects an already normalized floating-point tensor")
        images = images.to(self.device)
        if dataset_source is None:
            dataset_source = self.default_dataset_source
        context = torch.no_grad() if self._frozen else nullcontext()
        with context:
            output = self.backbone(images, dataset_source=dataset_source)
        return self._normalize_output(output, self.output_index)

    @staticmethod
    def feature_map_to_tokens(feature_map: Tensor) -> Tensor:
        if feature_map.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(feature_map.shape)}")
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def preprocess(
        self,
        images: Tensor,
        input_color: str = "RGB",
        input_range: str = "0_255",
    ) -> Tensor:
        """Resize and normalize an already cropped top-down image batch.

        Args:
            images: [C,H,W] or [B,C,H,W], uint8 or floating Tensor.
            input_color: RGB or BGR.  Official ViTPose loading uses RGB.
            input_range: ``0_255`` or ``0_1``; no range guessing is performed.

        Full-frame inputs must first be cropped/affine-warped from a hand bbox.
        This helper intentionally does not invent a bbox or run a detector.
        """
        if not torch.is_tensor(images):
            raise TypeError("preprocess expects a torch.Tensor")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("preprocess expects [3,H,W] or [B,3,H,W]")
        input_color = input_color.upper()
        if input_color not in {"RGB", "BGR"}:
            raise ValueError("input_color must be 'RGB' or 'BGR'")
        if input_color == "BGR":
            images = images[:, [2, 1, 0], :, :]
        images = images.to(dtype=torch.float32)
        if input_range == "0_255":
            images = images / 255.0
        elif input_range != "0_1":
            raise ValueError("input_range must be '0_255' or '0_1'")

        image_size_wh = self.config.get("data_cfg", {}).get("image_size", [192, 256])
        target_size_hw = (int(image_size_wh[1]), int(image_size_wh[0]))
        images = F.interpolate(images, size=target_size_hw, mode="bilinear", align_corners=False)
        mean = images.new_tensor(self.preprocess_config.get("mean", [0.485, 0.456, 0.406]))
        std = images.new_tensor(self.preprocess_config.get("std", [0.229, 0.224, 0.225]))
        images = (images - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
        return images.to(self.device)


def _print_gradient_status(model: ViTPosePlusBackbone) -> None:
    parameters = list(model.backbone.parameters())
    requires_grad = sum(parameter.requires_grad for parameter in parameters)
    has_grad = sum(parameter.grad is not None for parameter in parameters)
    print(
        f"[grad] trainable_parameter_tensors={requires_grad}/{len(parameters)}, "
        f"tensors_with_grad={has_grad}"
    )


def _run_example(args: argparse.Namespace) -> None:
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTPosePlusBackbone(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        freeze=args.freeze,
        device=requested_device,
        interpolate_pos_encoding=args.interpolate_pos_encoding,
    )
    height, width = model.backbone.img_size
    images = torch.randn(args.batch_size, 3, height, width, device=model.device)
    dataset_source = torch.full(
        (args.batch_size,), args.dataset_source, dtype=torch.long, device=model.device
    )

    model.train(not args.eval)
    print(f"[model] wrapper.training={model.training}, backbone.training={model.backbone.training}")
    print(f"[input] shape={tuple(images.shape)}, dtype={images.dtype}, device={images.device}")
    print(
        f"[routing] dataset_source={dataset_source.tolist()}, "
        f"expert_names={model.dataset_experts}"
    )
    features = model(images, dataset_source=dataset_source)
    tokens = model.feature_map_to_tokens(features)
    print(f"[output] feature_map={tuple(features.shape)} (no heatmap/keypoints)")
    print(f"[output] graphormer_tokens={tuple(tokens.shape)}")
    print(f"[output] all_finite={bool(torch.isfinite(features).all())}")
    forbidden = [
        name
        for name, _ in model.named_modules()
        if any(word in name.lower() for word in ("keypoint_head", "heatmap", "neck"))
    ]
    print(f"[model] registered neck/heatmap/keypoint modules={forbidden or 'none'}")
    print(f"[grad] output_requires_grad={features.requires_grad}")

    if args.verify_backward:
        if args.freeze:
            print("[grad] frozen mode: backward is intentionally skipped because output is detached")
            _print_gradient_status(model)
        else:
            # Do not sum all LayerNorm channels (their sum is approximately
            # constant); a channel-weighted loss gives a meaningful test.
            weights = torch.linspace(0.5, 1.5, features.shape[1], device=features.device)
            loss = (features * weights.view(1, -1, 1, 1)).mean()
            loss.backward()
            _print_gradient_status(model)
            patch_grad = model.backbone.patch_embed.proj.weight.grad
            selected_grad = model.backbone.blocks[0].mlp.experts[args.dataset_source].weight.grad
            unselected_index = (args.dataset_source + 1) % model.backbone.num_expert
            unselected_grad = model.backbone.blocks[0].mlp.experts[unselected_index].weight.grad
            print(
                "[grad] patch_embed_grad_abs_sum="
                f"{float(patch_grad.abs().sum()) if patch_grad is not None else None}"
            )
            print(
                f"[grad] selected_expert_{args.dataset_source}_has_grad={selected_grad is not None}; "
                f"unselected_expert_{unselected_index}_has_grad={unselected_grad is not None} "
                "(expected False for sparse routing)"
            )


def _build_argument_parser() -> argparse.ArgumentParser:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=directory / "vitpose_plus_base_backbone_config.py",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=directory / "vitpose _base.pth",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset-source", type=int, default=5, choices=range(6))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true", help="freeze and detach backbone features")
    mode.add_argument("--trainable", dest="freeze", action="store_false", help="allow fine-tuning")
    parser.set_defaults(freeze=True)
    parser.add_argument("--eval", action="store_true", help="use eval mode (frozen always stays eval)")
    parser.add_argument("--verify-backward", action="store_true")
    parser.add_argument("--interpolate-pos-encoding", action="store_true")
    return parser


if __name__ == "__main__":
    _run_example(_build_argument_parser().parse_args())
