from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import torch

from losses import VISIBLE_LABEL


FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.float()
    return (values * mask_f).sum(), mask_f.sum()


def init_metric_state() -> Dict[str, float]:
    metric_keys = (
        "overall_mpjpe",
        "visible_mpjpe",
        "self_occ_mpjpe",
        "out_of_view_mpjpe",
        "fingertip_mpjpe",
        "self_occ_fingertip_mpjpe",
        "pck_20",
        "pck_30",
    )
    state = OrderedDict()
    for key in metric_keys:
        state[f"{key}_sum"] = 0.0
        state[f"{key}_count"] = 0.0
    return state


def update_metric_state(state: Dict[str, float], pred_pose: torch.Tensor, gt_pose: torch.Tensor, batch) -> None:
    joint_error = torch.norm(pred_pose - gt_pose, dim=-1)

    visibility_label = batch["visibility_label"]
    in_view_mask = batch["in_view_mask"].bool()
    visible_mask = (visibility_label == VISIBLE_LABEL) & in_view_mask
    self_occ_mask = (visibility_label != VISIBLE_LABEL) & in_view_mask
    out_of_view_mask = ~in_view_mask
    fingertip_mask = torch.zeros_like(in_view_mask, dtype=torch.bool)
    fingertip_mask[:, list(FINGERTIP_INDICES)] = True
    self_occ_fingertip_mask = self_occ_mask & fingertip_mask

    masked_specs = (
        ("overall_mpjpe", torch.ones_like(in_view_mask, dtype=torch.bool), joint_error),
        ("visible_mpjpe", visible_mask, joint_error),
        ("self_occ_mpjpe", self_occ_mask, joint_error),
        ("out_of_view_mpjpe", out_of_view_mask, joint_error),
        ("fingertip_mpjpe", fingertip_mask, joint_error),
        ("self_occ_fingertip_mpjpe", self_occ_fingertip_mask, joint_error),
        ("pck_20", in_view_mask, (joint_error <= 20.0).float()),
        ("pck_30", in_view_mask, (joint_error <= 30.0).float()),
    )
    for name, mask, values in masked_specs:
        value_sum, count = _masked_sum(values, mask)
        state[f"{name}_sum"] += float(value_sum.detach().cpu())
        state[f"{name}_count"] += float(count.detach().cpu())


def finalize_metric_state(state: Dict[str, float]) -> OrderedDict:
    metrics = OrderedDict()
    for key in (
        "overall_mpjpe",
        "visible_mpjpe",
        "self_occ_mpjpe",
        "out_of_view_mpjpe",
        "fingertip_mpjpe",
        "self_occ_fingertip_mpjpe",
        "pck_20",
        "pck_30",
    ):
        value_sum = state[f"{key}_sum"]
        count = state[f"{key}_count"]
        metrics[key] = value_sum / count if count > 0 else float("nan")
    return metrics


def evaluate_model_on_loader(model, loader, device: torch.device) -> OrderedDict:
    model = model.to(device)
    model.eval()
    state = init_metric_state()
    with torch.no_grad():
        for batch in loader:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)
            outputs = model(batch["img"], batch["hand_back"])
            update_metric_state(state, outputs["pose3d"], batch["gt_pose"], batch)
    return finalize_metric_state(state)


def format_metric_table(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    *,
    before_name: str,
    after_name: str,
) -> str:
    lines = [f"{'metric':28s} {before_name:>12s} {after_name:>12s} {'delta':>12s}"]
    for key in before_metrics.keys():
        before_value = before_metrics[key]
        after_value = after_metrics[key]
        delta = after_value - before_value
        lines.append(f"{key:28s} {before_value:12.4f} {after_value:12.4f} {delta:12.4f}")
    return "\n".join(lines)


def evaluate_checkpoint_pair(
    before_model,
    after_model,
    loader,
    *,
    device: torch.device,
    before_name: str,
    after_name: str,
) -> Tuple[OrderedDict, OrderedDict, str]:
    before_metrics = evaluate_model_on_loader(before_model, loader, device)
    after_metrics = evaluate_model_on_loader(after_model, loader, device)
    table = format_metric_table(
        before_metrics,
        after_metrics,
        before_name=before_name,
        after_name=after_name,
    )
    return before_metrics, after_metrics, table
