#!/usr/bin/env python3
"""Compare official ViTPose ViTMoE with the existing local extraction."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torchvision
import mmcv
import mmpose
import timm
import einops
from torch import Tensor

from load_official_vitpose_plus_backbone import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    OfficialViTPosePlusBackbone,
    deterministic_input,
)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EXISTING_SCRIPT = PROJECT_ROOT / "pretrain_vitpose_pkl_and_call/load_vitpose_plus_backbone.py"
EXISTING_CONFIG = PROJECT_ROOT / "pretrain_vitpose_pkl_and_call/vitpose_plus_base_backbone_config.py"
DEFAULT_REFERENCE = HERE / "comparison_results/existing_vit_reference.npz"
DEFAULT_REPORT = HERE / "comparison_results/comparison_report.json"


def _existing_class():
    spec = importlib.util.spec_from_file_location("existing_vitpose_loader_for_compare", str(EXISTING_SCRIPT))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import {}".format(EXISTING_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ViTPosePlusBackbone


def _diff(left: Tensor, right: Tensor) -> Dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    delta = (left.float() - right.float()).abs()
    denominator = float(torch.linalg.norm(right.float()))
    relative_l2 = float(torch.linalg.norm(delta)) / max(denominator, 1e-30)
    return {
        "shape_equal": True,
        "exact_equal": bool(torch.equal(left, right)),
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "allclose_atol_1e-6_rtol_1e-5": bool(torch.allclose(left, right, atol=1e-6, rtol=1e-5)),
        "allclose_atol_2e-5_rtol_1e-5": bool(torch.allclose(left, right, atol=2e-5, rtol=1e-5)),
        "relative_l2": relative_l2,
    }


def _compare_parameters(official, existing) -> Dict[str, Any]:
    official_state = official.backbone.state_dict()
    existing_state = existing.backbone.state_dict()
    official_keys = set(official_state)
    existing_keys = set(existing_state)
    common = sorted(official_keys & existing_keys)
    mismatched_shapes = []
    unequal = []
    max_abs = 0.0
    for key in common:
        left, right = official_state[key].cpu(), existing_state[key].cpu()
        if tuple(left.shape) != tuple(right.shape):
            mismatched_shapes.append(key)
            continue
        if not torch.equal(left, right):
            unequal.append(key)
            max_abs = max(max_abs, float((left.float() - right.float()).abs().max()))
    report = {
        "official_keys": len(official_state),
        "existing_keys": len(existing_state),
        "common_keys": len(common),
        "only_official": sorted(official_keys - existing_keys),
        "only_existing": sorted(existing_keys - official_keys),
        "shape_mismatches": mismatched_shapes,
        "unequal_value_keys": unequal,
        "max_parameter_abs_diff": max_abs,
    }
    print("[compare/parameters] {}".format(report))
    return report


def _activation_hooks(backbone) -> Tuple[Dict[str, Tensor], Iterable[Any]]:
    activations: Dict[str, Tensor] = {}
    handles = []

    def hook(name):
        def save(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                activations[name] = value.detach().cpu()
        return save

    handles.append(backbone.patch_embed.register_forward_hook(hook("patch_embed")))
    for index, block in enumerate(backbone.blocks):
        handles.append(block.register_forward_hook(hook("blocks.{}".format(index))))
        if index == 0:
            handles.append(block.norm1.register_forward_hook(hook("blocks.0.norm1")))
            handles.append(block.attn.register_forward_hook(hook("blocks.0.attn")))
            handles.append(block.norm2.register_forward_hook(hook("blocks.0.norm2")))
            handles.append(block.mlp.register_forward_hook(hook("blocks.0.mlp")))
    handles.append(backbone.last_norm.register_forward_hook(hook("last_norm")))
    return activations, handles


def _run_with_activations(model, images: Tensor, expert: int):
    activations, handles = _activation_hooks(model.backbone)
    try:
        with torch.no_grad():
            output = model(images, dataset_source=expert).cpu()
    finally:
        for handle in handles:
            handle.remove()
    return output, activations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--layer-expert", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if any(index not in range(6) for index in args.experts):
        raise ValueError("Expert indices must be in [0, 5]")

    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "torchvision": torchvision.__version__,
        "mmcv": mmcv.__version__,
        "mmpose": mmpose.__version__,
        "timm": timm.__version__,
        "einops": einops.__version__,
        "numpy": np.__version__,
    }
    print("[compare/environment] {}".format(environment))
    official = OfficialViTPosePlusBackbone(
        args.config, args.checkpoint, freeze=True, device="cpu"
    )
    existing = _existing_class()(
        EXISTING_CONFIG, args.checkpoint, freeze=True, device="cpu"
    )
    official.eval()
    existing.eval()
    images = deterministic_input(args.batch_size)

    report: Dict[str, Any] = {
        "environment": environment,
        "official_class": "{}.{}".format(
            official.backbone.__class__.__module__, official.backbone.__class__.__name__
        ),
        "existing_class": "{}.{}".format(
            existing.backbone.__class__.__module__, existing.backbone.__class__.__name__
        ),
        "parameters": _compare_parameters(official, existing),
        "same_runtime_outputs": {},
        "layer_outputs": {},
        "cross_environment_reference": {},
    }

    official_outputs: Dict[int, Tensor] = {}
    existing_outputs: Dict[int, Tensor] = {}
    for expert in args.experts:
        with torch.no_grad():
            official_output = official(images, dataset_source=expert).cpu()
            existing_output = existing(images, dataset_source=expert).cpu()
        official_outputs[expert] = official_output
        existing_outputs[expert] = existing_output
        comparison = _diff(official_output, existing_output)
        report["same_runtime_outputs"][str(expert)] = comparison
        print("[compare/same-runtime] expert={} {}".format(expert, comparison))

    official_layer_output, official_activations = _run_with_activations(
        official, images, args.layer_expert
    )
    existing_layer_output, existing_activations = _run_with_activations(
        existing, images, args.layer_expert
    )
    if not torch.equal(official_layer_output, official_outputs[args.layer_expert]):
        raise RuntimeError("Repeated official eval forward was not deterministic")
    if not torch.equal(existing_layer_output, existing_outputs[args.layer_expert]):
        raise RuntimeError("Repeated existing eval forward was not deterministic")
    for name in sorted(set(official_activations) | set(existing_activations)):
        if name not in official_activations or name not in existing_activations:
            comparison = {"present_in_both": False}
        else:
            comparison = _diff(official_activations[name], existing_activations[name])
            comparison["present_in_both"] = True
        report["layer_outputs"][name] = comparison
        print("[compare/layer] {} {}".format(name, comparison))

    if args.reference.is_file():
        with np.load(str(args.reference), allow_pickle=False) as reference:
            reference_input = torch.from_numpy(reference["input"])
            input_diff = _diff(images, reference_input)
            report["cross_environment_reference"]["input"] = input_diff
            print("[compare/cross-env] input {}".format(input_diff))
            if not torch.equal(images, reference_input):
                raise RuntimeError("Reference input differs from comparison input")
            metadata = json.loads(str(reference["metadata_json"]))
            report["cross_environment_reference"]["metadata"] = metadata
            for expert in args.experts:
                key = "features_expert_{}".format(expert)
                if key not in reference:
                    continue
                comparison = _diff(official_outputs[expert], torch.from_numpy(reference[key]))
                report["cross_environment_reference"][str(expert)] = comparison
                print("[compare/cross-env] expert={} {}".format(expert, comparison))
    else:
        print("[compare/cross-env] reference not found; skipped: {}".format(args.reference))

    parameter_report = report["parameters"]
    if (
        parameter_report["only_official"]
        or parameter_report["only_existing"]
        or parameter_report["shape_mismatches"]
        or parameter_report["unequal_value_keys"]
    ):
        raise AssertionError("Backbone parameters are not identical")
    failures = [
        expert for expert, value in report["same_runtime_outputs"].items()
        if not value.get("allclose_atol_2e-5_rtol_1e-5", False)
        or value.get("relative_l2", float("inf")) > 2e-6
    ]
    if failures:
        raise AssertionError("Same-runtime output mismatch for experts {}".format(failures))

    cross_failures = [
        expert for expert, value in report["cross_environment_reference"].items()
        if expert.isdigit()
        and (
            not value.get("allclose_atol_2e-5_rtol_1e-5", False)
            or value.get("relative_l2", float("inf")) > 2e-6
        )
    ]
    if cross_failures:
        raise AssertionError("Cross-environment output mismatch for experts {}".format(cross_failures))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("[compare/result] PASS")
    print("[compare/result] report={}".format(args.report.resolve()))


if __name__ == "__main__":
    main()
