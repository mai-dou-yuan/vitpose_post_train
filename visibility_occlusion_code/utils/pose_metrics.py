"""Pose evaluation metrics used by the occlusion analysis.

The functions in this module accept NumPy arrays or PyTorch tensors. Inputs are
expected to have shape ``[N, J, 3]`` for 3D poses and optional masks with shape
``[N, J]`` where ``True`` means the joint should be included.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, "object"]


HAND_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

FINGERTIP_INDICES: Tuple[int, ...] = (4, 8, 12, 16, 20)

VISIBLE_LABEL = 0
OCCLUDED_LABEL = 1
OUT_OF_VIEW_LABEL = 2
UNCERTAIN_LABEL = 3


def _to_numpy(value: ArrayLike) -> np.ndarray:
    """Convert NumPy/Torch-like arrays to NumPy without requiring torch import."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _as_pose_array(value: ArrayLike) -> np.ndarray:
    array = _to_numpy(value).astype(np.float64, copy=False)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected pose shape [N, J, 3], got {array.shape}")
    return array


def _as_mask(mask: Optional[ArrayLike], shape: Tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    mask_array = _to_numpy(mask).astype(bool, copy=False)
    if mask_array.ndim == 1:
        mask_array = mask_array[None, :]
    if mask_array.shape != shape:
        raise ValueError(f"Expected mask shape {shape}, got {mask_array.shape}")
    return mask_array


def _masked_mean(values: np.ndarray, mask: Optional[np.ndarray] = None, axis=None):
    if mask is None:
        valid = np.isfinite(values)
    else:
        valid = mask & np.isfinite(values)
    if axis is None:
        if not np.any(valid):
            return float("nan")
        return float(np.mean(values[valid]))
    counts = valid.sum(axis=axis)
    sums = np.where(valid, values, 0.0).sum(axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def joint_errors(pred: ArrayLike, gt: ArrayLike) -> np.ndarray:
    """Return per-joint Euclidean errors with shape ``[N, J]``."""
    pred_array = _as_pose_array(pred)
    gt_array = _as_pose_array(gt)
    if pred_array.shape != gt_array.shape:
        raise ValueError(f"Pose shape mismatch: {pred_array.shape} vs {gt_array.shape}")
    return np.linalg.norm(pred_array - gt_array, axis=-1)


def mpjpe(pred: ArrayLike, gt: ArrayLike, mask: Optional[ArrayLike] = None) -> float:
    errors = joint_errors(pred, gt)
    return _masked_mean(errors, _as_mask(mask, errors.shape) if mask is not None else None)


def root_aligned_mpjpe(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: Optional[ArrayLike] = None,
    root_index: int = 0,
) -> float:
    pred_array = _as_pose_array(pred)
    gt_array = _as_pose_array(gt)
    pred_rel = pred_array - pred_array[:, root_index:root_index + 1, :]
    gt_rel = gt_array - gt_array[:, root_index:root_index + 1, :]
    return mpjpe(pred_rel, gt_rel, mask=mask)


def per_joint_mpjpe(pred: ArrayLike, gt: ArrayLike, mask: Optional[ArrayLike] = None) -> np.ndarray:
    errors = joint_errors(pred, gt)
    mask_array = _as_mask(mask, errors.shape) if mask is not None else None
    return _masked_mean(errors, mask_array, axis=0)


def pck(
    pred: ArrayLike,
    gt: ArrayLike,
    threshold: Union[float, Sequence[float]],
    mask: Optional[ArrayLike] = None,
) -> Union[float, Dict[float, float]]:
    errors = joint_errors(pred, gt)
    mask_array = _as_mask(mask, errors.shape) if mask is not None else np.ones(errors.shape, dtype=bool)

    def _single_pck(th: float) -> float:
        valid = mask_array & np.isfinite(errors)
        if not np.any(valid):
            return float("nan")
        return float(np.mean(errors[valid] <= th))

    if isinstance(threshold, (list, tuple, np.ndarray)):
        return {float(th): _single_pck(float(th)) for th in threshold}
    return _single_pck(float(threshold))


def auc_pck(
    pred: ArrayLike,
    gt: ArrayLike,
    max_threshold: float = 50.0,
    step: float = 5.0,
    mask: Optional[ArrayLike] = None,
) -> float:
    thresholds = np.arange(0.0, max_threshold + 1e-9, step, dtype=np.float64)
    values = np.array([pck(pred, gt, th, mask=mask) for th in thresholds], dtype=np.float64)
    if np.all(~np.isfinite(values)):
        return float("nan")
    return float(np.trapz(values, thresholds) / max_threshold)


def bone_length_error(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: Optional[ArrayLike] = None,
    bones: Iterable[Tuple[int, int]] = HAND_BONES,
) -> float:
    pred_array = _as_pose_array(pred)
    gt_array = _as_pose_array(gt)
    bones_tuple = tuple(bones)
    pred_lengths = []
    gt_lengths = []
    bone_masks = []
    joint_mask = _as_mask(mask, pred_array.shape[:2]) if mask is not None else None

    for start, end in bones_tuple:
        pred_lengths.append(np.linalg.norm(pred_array[:, start] - pred_array[:, end], axis=-1))
        gt_lengths.append(np.linalg.norm(gt_array[:, start] - gt_array[:, end], axis=-1))
        if joint_mask is not None:
            bone_masks.append(joint_mask[:, start] & joint_mask[:, end])

    length_error = np.abs(np.stack(pred_lengths, axis=1) - np.stack(gt_lengths, axis=1))
    mask_array = np.stack(bone_masks, axis=1) if bone_masks else None
    return _masked_mean(length_error, mask_array)


def labels_to_mask(labels: ArrayLike, include: Union[int, Sequence[int]]) -> np.ndarray:
    labels_array = _to_numpy(labels)
    if labels_array.ndim == 1:
        labels_array = labels_array[None, :]
    include_values = np.array([include] if isinstance(include, int) else list(include))
    return np.isin(labels_array, include_values)


def summarize_pose_metrics(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: Optional[ArrayLike] = None,
    root_index: int = 0,
    pck_thresholds: Sequence[float] = (20.0, 30.0, 50.0),
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "MPJPE": mpjpe(pred, gt, mask=mask),
        "RA-MPJPE": root_aligned_mpjpe(pred, gt, mask=mask, root_index=root_index),
        "AUC@50": auc_pck(pred, gt, max_threshold=50.0, step=5.0, mask=mask),
        "BoneLengthError": bone_length_error(pred, gt, mask=mask),
        "PerJointMPJPE": per_joint_mpjpe(pred, gt, mask=mask),
    }
    for threshold, value in pck(pred, gt, pck_thresholds, mask=mask).items():
        summary[f"PCK@{int(threshold)}"] = value
    return summary


def summarize_by_visibility(
    pred: ArrayLike,
    gt: ArrayLike,
    visibility_labels: ArrayLike,
) -> Mapping[str, Dict[str, object]]:
    visible_mask = labels_to_mask(visibility_labels, VISIBLE_LABEL)
    occluded_mask = labels_to_mask(visibility_labels, OCCLUDED_LABEL)
    in_view_mask = labels_to_mask(visibility_labels, (VISIBLE_LABEL, OCCLUDED_LABEL))

    fingertip_mask = np.zeros_like(in_view_mask, dtype=bool)
    fingertip_mask[:, FINGERTIP_INDICES] = True

    return {
        "all_in_view": summarize_pose_metrics(pred, gt, mask=in_view_mask),
        "visible": summarize_pose_metrics(pred, gt, mask=visible_mask),
        "occluded": summarize_pose_metrics(pred, gt, mask=occluded_mask),
        "occluded_fingertips": summarize_pose_metrics(pred, gt, mask=occluded_mask & fingertip_mask),
    }
