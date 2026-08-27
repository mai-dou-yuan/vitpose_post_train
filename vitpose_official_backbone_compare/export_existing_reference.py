#!/usr/bin/env python3
"""Export deterministic features from the existing dependency-light loader."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EXISTING_SCRIPT = PROJECT_ROOT / "pretrain_vitpose_pkl_and_call/load_vitpose_plus_backbone.py"
EXISTING_CONFIG = PROJECT_ROOT / "pretrain_vitpose_pkl_and_call/vitpose_plus_base_backbone_config.py"
CHECKPOINT = PROJECT_ROOT / "pretrain_vitpose_pkl_and_call/vitpose _base.pth"
DEFAULT_OUTPUT = HERE / "comparison_results/existing_vit_reference.npz"


def _existing_class():
    spec = importlib.util.spec_from_file_location("existing_vitpose_loader", str(EXISTING_SCRIPT))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import {}".format(EXISTING_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ViTPosePlusBackbone


def deterministic_input(batch_size: int = 1) -> torch.Tensor:
    count = batch_size * 3 * 256 * 192
    values = torch.arange(count, dtype=torch.float32)
    return ((values.remainder(2048) / 1023.5) - 1.0).reshape(batch_size, 3, 256, 192)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    if any(index not in range(6) for index in args.experts):
        raise ValueError("Expert indices must be in [0, 5]")

    model = _existing_class()(
        EXISTING_CONFIG, CHECKPOINT, freeze=True, device="cpu"
    )
    model.eval()
    images = deterministic_input(args.batch_size)
    arrays = {"input": images.numpy()}
    with torch.no_grad():
        for expert in args.experts:
            features = model(images, dataset_source=expert)
            arrays["features_expert_{}".format(expert)] = features.cpu().numpy()
            print(
                "[existing/reference] expert={}, shape={}, mean={:.9g}, std={:.9g}".format(
                    expert, tuple(features.shape), float(features.mean()), float(features.std())
                )
            )
    metadata = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "implementation": str(EXISTING_SCRIPT),
        "config": str(EXISTING_CONFIG),
        "checkpoint": str(CHECKPOINT),
        "experts": args.experts,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.output), **arrays)
    print("[existing/reference] saved={}".format(args.output.resolve()))
    print("[existing/reference] metadata={}".format(metadata))


if __name__ == "__main__":
    main()
