"""Checkpoint loading helpers for experiment architecture updates."""

from pathlib import Path

import torch


INITIAL_2D_HEAD_PREFIX = "initial_2d_head."
MESH_CROSS_STAGE_ATTENTION_PREFIX = "mesh_regressor.cross_stage_attn."
MESH_HEAD_PREFIXES = (
    "mesh_token_projection.",
    "mesh_regressor.",
    "mesh_to_joints.",
)
NEW_ARCHITECTURE_PREFIXES = (INITIAL_2D_HEAD_PREFIX, *MESH_HEAD_PREFIXES)


def load_pose_checkpoint_weights(model, checkpoint_path):
    """Load model weights non-strictly and print the incompatibility report.

    This weights-only path is suitable for checkpoints created before the
    Initial 2D Reference Head existed.  Optimizer/scheduler state is
    intentionally not restored because its parameter groups predate the head.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be a mapping, got {type(checkpoint).__name__}"
        )
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict must be a mapping")

    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_keys = sorted(incompatible.missing_keys)
    unexpected_keys = sorted(incompatible.unexpected_keys)
    initial_head_missing_keys = [
        key for key in missing_keys if key.startswith(INITIAL_2D_HEAD_PREFIX)
    ]
    mesh_attention_missing_keys = [
        key
        for key in missing_keys
        if key.startswith(MESH_CROSS_STAGE_ATTENTION_PREFIX)
    ]
    architecture_missing_keys = [
        key
        for key in missing_keys
        if key.startswith(NEW_ARCHITECTURE_PREFIXES)
    ]
    print(f"[checkpoint] weights-only non-strict load: {checkpoint_path}")
    print(f"[checkpoint] missing keys ({len(missing_keys)}): {missing_keys or 'none'}")
    print(
        "[checkpoint] Initial 2D Head missing keys "
        f"({len(initial_head_missing_keys)}): {initial_head_missing_keys or 'none'}"
    )
    print(
        "[checkpoint] Mesh Cross-Stage Attention missing keys "
        f"({len(mesh_attention_missing_keys)}): "
        f"{mesh_attention_missing_keys or 'none'}"
    )
    print(
        "[checkpoint] new architecture missing keys "
        f"({len(architecture_missing_keys)}): {architecture_missing_keys or 'none'}"
    )
    print(
        f"[checkpoint] unexpected keys ({len(unexpected_keys)}): "
        f"{unexpected_keys or 'none'}"
    )
    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "initial_2d_head_missing_keys": initial_head_missing_keys,
        "mesh_attention_missing_keys": mesh_attention_missing_keys,
        "architecture_missing_keys": architecture_missing_keys,
    }


def is_legacy_initial_2d_checkpoint(load_report):
    """Return whether only parameters introduced by architecture updates are absent."""
    missing_keys = load_report["missing_keys"]
    architecture_missing_keys = load_report.get(
        "architecture_missing_keys", load_report["initial_2d_head_missing_keys"]
    )
    return (
        bool(missing_keys)
        and not load_report["unexpected_keys"]
        and len(missing_keys) == len(architecture_missing_keys)
    )
