from typing import Dict

import torch

from losses import build_self_occ_mask


VISIBLE_LABEL = 0
FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask_f = mask.float()
    count = mask_f.sum()
    if count < 1:
        return values.sum() * 0.0
    return (values * mask_f).sum() / (count + eps)


def _build_fingertip_mask(reference_mask: torch.Tensor) -> torch.Tensor:
    fingertip_mask = torch.zeros_like(reference_mask, dtype=torch.bool)
    fingertip_mask[:, list(FINGERTIP_INDICES)] = True
    return fingertip_mask


def compute_pose_metrics(pred_pose: torch.Tensor, gt_pose: torch.Tensor, batch) -> Dict[str, torch.Tensor]:
    joint_error = torch.norm(pred_pose - gt_pose, dim=-1)

    visibility_label = batch["visibility_label"]
    in_view_mask = batch["in_view_mask"].bool()
    visible_mask = (visibility_label == VISIBLE_LABEL) & in_view_mask
    self_occ_mask = build_self_occ_mask(batch)
    out_of_view_mask = ~in_view_mask
    fingertip_mask = _build_fingertip_mask(in_view_mask)
    self_occ_fingertip_mask = self_occ_mask & fingertip_mask

    metrics = {
        "overall_mpjpe": joint_error.mean(),
        "visible_mpjpe": _masked_mean(joint_error, visible_mask),
        "self_occ_mpjpe": _masked_mean(joint_error, self_occ_mask),
        "out_of_view_mpjpe": _masked_mean(joint_error, out_of_view_mask),
        "fingertip_mpjpe": _masked_mean(joint_error, fingertip_mask),
        "self_occ_fingertip_mpjpe": _masked_mean(joint_error, self_occ_fingertip_mask),
        "pck_20": _masked_mean((joint_error <= 20.0).float(), in_view_mask),
        "pck_30": _masked_mean((joint_error <= 30.0).float(), in_view_mask),
    }
    return metrics
