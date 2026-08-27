#!/usr/bin/env python3
"""Build and call only the official ViTPose++-B ViTMoE backbone.

This module deliberately uses the ViTPose repository's vendored MMPose and
MMCV-1.x registry.  It does not construct TopDownMoE, a keypoint head, a neck,
or a heatmap decoder.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

# MMPose imports matplotlib through its package initializers.  Keep caches in a
# writable process-local location on read-only/home-mounted compute nodes.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vitpose-official-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/vitpose-official-cache")

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PathLike = Union[str, Path]
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
OFFICIAL_REPO = HERE / "official_vitpose"
DEFAULT_CONFIG = (
    OFFICIAL_REPO
    / "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco"
    / "vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "pretrain_vitpose_pkl_and_call/vitpose _base.pth"

# Prefer the checked-out official source even if another mmpose is installed.
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from mmcv import Config  # noqa: E402
from mmpose.models.builder import build_backbone  # noqa: E402


def _extract_state_dict(checkpoint: Any) -> Tuple[Mapping[str, Any], str]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a mapping")
    for container in ("state_dict", "model", "module"):
        value = checkpoint.get(container)
        if isinstance(value, Mapping):
            return value, container
    return checkpoint, "<root>"


def _backbone_key(key: str) -> Tuple[str, bool]:
    """Remove wrappers and one backbone prefix; report explicit ownership."""
    parts = key.split(".")
    while parts and parts[0] in {"module", "model", "state_dict"}:
        parts.pop(0)
    explicitly_backbone = bool(parts and parts[0] == "backbone")
    if explicitly_backbone:
        parts.pop(0)
    return ".".join(parts), explicitly_backbone


def _validate_plus_base_config(backbone_cfg: Mapping[str, Any]) -> None:
    expected = {
        "type": "ViTMoE",
        "img_size": (256, 192),
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "num_expert": 6,
        "part_features": 192,
    }
    errors = []
    for key, wanted in expected.items():
        actual = backbone_cfg.get(key)
        if key == "img_size" and actual is not None:
            actual = tuple(actual)
        if actual != wanted:
            errors.append("{}={!r}, expected {!r}".format(key, actual, wanted))
    if errors:
        raise ValueError(
            "Official config is not the released multi-task ViTPose++-B: "
            + "; ".join(errors)
        )


class OfficialViTPosePlusBackbone(nn.Module):
    """Backbone-only wrapper around the official MMPose ``ViTMoE`` class."""

    def __init__(
        self,
        config_path: PathLike = DEFAULT_CONFIG,
        checkpoint_path: PathLike = DEFAULT_CHECKPOINT,
        freeze: bool = False,
        device: Union[str, torch.device] = "cuda",
        min_load_ratio: float = 0.90,
    ) -> None:
        super().__init__()
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError("Official config not found: {}".format(self.config_path))
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Checkpoint not found: {}".format(self.checkpoint_path))

        self.config = Config.fromfile(str(self.config_path))
        backbone_cfg = self.config.model.backbone.copy()
        _validate_plus_base_config(backbone_cfg)

        # This is the only project model registered by the wrapper.  Building
        # model/TopDownMoE would also allocate and call 2-D heatmap heads.
        self.backbone = build_backbone(backbone_cfg)
        self.backbone_config = dict(backbone_cfg)
        self.out_channels = int(backbone_cfg["embed_dim"])
        self.img_size = tuple(int(x) for x in backbone_cfg["img_size"])
        self.num_expert = int(backbone_cfg["num_expert"])
        self.default_dataset_source = 5
        self.last_load_report = self._load_backbone_weights(min_load_ratio)

        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable; falling back to CPU")
            requested = torch.device("cpu")
        self.device = requested
        self.to(self.device)
        self.set_frozen(freeze)

    def _load_backbone_weights(self, min_load_ratio: float) -> Dict[str, Any]:
        if not 0.0 < min_load_ratio <= 1.0:
            raise ValueError("min_load_ratio must be in (0, 1]")
        checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
        state_dict, container = _extract_state_dict(checkpoint)
        expected = self.backbone.state_dict()
        selected: Dict[str, Tensor] = {}
        ignored = 0
        unexpected_backbone = []
        shape_mismatches = []

        for original_key, value in state_dict.items():
            if not isinstance(original_key, str) or not torch.is_tensor(value):
                ignored += 1
                continue
            candidate, explicit = _backbone_key(original_key)
            if candidate not in expected:
                if explicit:
                    unexpected_backbone.append(original_key)
                else:
                    ignored += 1
                continue
            if tuple(value.shape) != tuple(expected[candidate].shape):
                shape_mismatches.append(
                    "{}: checkpoint{} != model{}".format(
                        original_key, tuple(value.shape), tuple(expected[candidate].shape)
                    )
                )
                continue
            if candidate in selected:
                raise RuntimeError("Duplicate checkpoint mapping for {}".format(candidate))
            selected[candidate] = value

        incompatible = self.backbone.load_state_dict(selected, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys) + unexpected_backbone
        loaded_numel = sum(expected[key].numel() for key in selected)
        total_numel = sum(value.numel() for value in expected.values())
        key_ratio = len(selected) / len(expected)
        numel_ratio = loaded_numel / total_numel

        print("[official/checkpoint] path={}".format(self.checkpoint_path))
        print("[official/checkpoint] container={!r}, source entries={}".format(container, len(state_dict)))
        print(
            "[official/checkpoint] loaded {}/{} tensors ({:.2%}), {}/{} values ({:.2%})".format(
                len(selected), len(expected), key_ratio,
                loaded_numel, total_numel, numel_ratio,
            )
        )
        print("[official/checkpoint] ignored non-backbone entries={}".format(ignored))
        print("[official/checkpoint] missing keys={}".format(missing or "none"))
        print("[official/checkpoint] unexpected backbone keys={}".format(unexpected or "none"))
        print("[official/checkpoint] shape mismatches={}".format(shape_mismatches or "none"))

        if key_ratio < min_load_ratio or numel_ratio < min_load_ratio:
            raise RuntimeError(
                "Insufficient official backbone load: key_ratio={:.2%}, numel_ratio={:.2%}".format(
                    key_ratio, numel_ratio
                )
            )
        if missing or unexpected or shape_mismatches:
            raise RuntimeError("Official backbone checkpoint did not match exactly")

        representative = [
            "pos_embed",
            "patch_embed.proj.weight",
            "blocks.0.mlp.experts.5.weight",
            "blocks.11.attn.qkv.weight",
            "last_norm.weight",
        ]
        copied_diff = {
            key: float((self.backbone.state_dict()[key].cpu() - selected[key].cpu()).abs().max())
            for key in representative
        }
        print("[official/checkpoint] representative max_abs_diff={}".format(copied_diff))
        if any(value != 0.0 for value in copied_diff.values()):
            raise RuntimeError("At least one checkpoint tensor was not copied exactly")

        return {
            "container": container,
            "loaded_keys": len(selected),
            "total_keys": len(expected),
            "key_ratio": key_ratio,
            "loaded_numel": loaded_numel,
            "total_numel": total_numel,
            "numel_ratio": numel_ratio,
            "ignored_non_backbone": ignored,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
            "representative_max_abs_diff": copied_diff,
        }

    def set_frozen(self, freeze: bool = True) -> "OfficialViTPosePlusBackbone":
        self._frozen = bool(freeze)
        self.backbone.requires_grad_(not self._frozen)
        if self._frozen:
            self.backbone.eval()
        else:
            self.backbone.train(self.training)
        return self

    def train(self, mode: bool = True) -> "OfficialViTPosePlusBackbone":
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.backbone.eval()
        return self

    def _dataset_source(
        self, value: Union[int, Sequence[int], Tensor], batch_size: int, device: torch.device
    ) -> Tensor:
        source = torch.as_tensor(value, device=device)
        if source.ndim == 0:
            source = source.expand(batch_size)
        source = source.reshape(-1).long()
        if source.numel() != batch_size:
            raise ValueError("dataset_source must have one value per image")
        if bool(((source < 0) | (source >= self.num_expert)).any()):
            raise ValueError("dataset_source values must be in [0, {}]".format(self.num_expert - 1))
        return source

    def forward(
        self,
        images: Tensor,
        dataset_source: Optional[Union[int, Sequence[int], Tensor]] = None,
    ) -> Tensor:
        if not torch.is_tensor(images) or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if not images.is_floating_point():
            raise TypeError("forward expects a normalized floating-point tensor")
        if tuple(images.shape[-2:]) != self.img_size:
            raise ValueError(
                "Official forward expects HxW={}, got {}".format(
                    self.img_size, tuple(images.shape[-2:])
                )
            )
        images = images.to(self.device)
        route = self.default_dataset_source if dataset_source is None else dataset_source
        source = self._dataset_source(route, images.shape[0], images.device)
        context = torch.no_grad() if self._frozen else nullcontext()
        with context:
            output = self.backbone(images, dataset_source=source)
        if not torch.is_tensor(output):
            if not isinstance(output, (tuple, list)):
                raise TypeError("Official backbone returned {}".format(type(output).__name__))
            tensors = [item for item in output if torch.is_tensor(item)]
            if not tensors:
                raise TypeError("Official backbone returned no tensor")
            output = tensors[-1]
        return output

    @staticmethod
    def feature_map_to_tokens(feature_map: Tensor) -> Tensor:
        if feature_map.ndim != 4:
            raise ValueError("feature_map must be [B, C, H, W]")
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    def preprocess(
        self, images: Tensor, input_color: str = "RGB", input_range: str = "0_255"
    ) -> Tensor:
        """Resize/normalize an already cropped ROI using the official config."""
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [3,H,W] or [B,3,H,W]")
        input_color = input_color.upper()
        if input_color not in {"RGB", "BGR"}:
            raise ValueError("input_color must be RGB or BGR")
        if input_color == "BGR":
            images = images[:, [2, 1, 0]]
        images = images.float()
        if input_range == "0_255":
            images = images / 255.0
        elif input_range != "0_1":
            raise ValueError("input_range must be 0_255 or 0_1")
        image_size_wh = self.config.data_cfg.image_size
        images = F.interpolate(
            images,
            size=(int(image_size_wh[1]), int(image_size_wh[0])),
            mode="bilinear",
            align_corners=False,
        )
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((images - mean) / std).to(self.device)


def deterministic_input(batch_size: int = 1) -> Tensor:
    count = batch_size * 3 * 256 * 192
    values = torch.arange(count, dtype=torch.float32)
    return ((values.remainder(2048) / 1023.5) - 1.0).reshape(batch_size, 3, 256, 192)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-source", type=int, default=5, choices=range(6))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OfficialViTPosePlusBackbone(
        args.config, args.checkpoint, freeze=args.freeze, device=device
    )
    model.eval()
    images = deterministic_input(args.batch_size).to(model.device)
    features = model(images, dataset_source=args.dataset_source)
    tokens = model.feature_map_to_tokens(features)
    print("[official/model] class={}.{}".format(
        model.backbone.__class__.__module__, model.backbone.__class__.__name__))
    print("[official/model] registered child modules={}".format(list(dict(model.named_children()))))
    print("[official/input] shape={}, device={}".format(tuple(images.shape), images.device))
    print("[official/output] feature_map={}, tokens={}".format(tuple(features.shape), tuple(tokens.shape)))
    print("[official/output] heatmap/keypoints=none, all_finite={}".format(bool(torch.isfinite(features).all())))
    print("[official/output] requires_grad={}".format(features.requires_grad))


if __name__ == "__main__":
    main()
