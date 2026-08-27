"""Evaluate pose metrics by frame-level visible joint ratio groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_metrics import OCCLUDED_LABEL, VISIBLE_LABEL, summarize_pose_metrics


def parse_bins(text: str) -> List[float]:
    bins = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(bins) < 2:
        raise ValueError("At least two bin edges are required.")
    if any(bins[idx] >= bins[idx + 1] for idx in range(len(bins) - 1)):
        raise ValueError(f"Bins must be strictly increasing, got {bins}")
    return bins


def jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="NPZ with pred_pose, gt_pose, visibility_label.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--bins",
        default="0,0.25,0.5,0.75,1.000001",
        help="Comma-separated visible-ratio bin edges. Last edge is inclusive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    labels = data["visibility_label"]
    pred = data["pred_pose"]
    gt = data["gt_pose"]

    if "visible_joint_ratio" in data.files:
        ratios = data["visible_joint_ratio"].astype(np.float64)
    else:
        in_view = (labels == VISIBLE_LABEL) | (labels == OCCLUDED_LABEL)
        visible = labels == VISIBLE_LABEL
        counts = in_view.sum(axis=1)
        ratios = np.divide(
            visible.sum(axis=1),
            counts,
            out=np.full(counts.shape, np.nan, dtype=np.float64),
            where=counts > 0,
        )

    bins = parse_bins(args.bins)
    in_view_mask = (labels == VISIBLE_LABEL) | (labels == OCCLUDED_LABEL)
    result: Dict[str, Any] = {
        "bins": bins,
        "groups": [],
    }

    for idx in range(len(bins) - 1):
        low = bins[idx]
        high = bins[idx + 1]
        if idx == len(bins) - 2:
            frame_mask = (ratios >= low) & (ratios <= high)
        else:
            frame_mask = (ratios >= low) & (ratios < high)

        group_mask = np.zeros_like(in_view_mask, dtype=bool)
        group_mask[frame_mask] = in_view_mask[frame_mask]

        visible_count = int(((labels == VISIBLE_LABEL) & group_mask).sum())
        occluded_count = int(((labels == OCCLUDED_LABEL) & group_mask).sum())
        in_view_count = int(group_mask.sum())

        group = {
            "name": f"{low:.2f}-{min(high, 1.0):.2f}",
            "visible_ratio_min": low,
            "visible_ratio_max": min(high, 1.0),
            "frame_count": int(frame_mask.sum()),
            "mean_visible_joint_ratio": float(np.nanmean(ratios[frame_mask])) if np.any(frame_mask) else float("nan"),
            "in_view_joint_count": in_view_count,
            "visible_joint_count": visible_count,
            "occluded_joint_count": occluded_count,
            "occluded_joint_ratio_in_view": (
                float(occluded_count / in_view_count) if in_view_count > 0 else float("nan")
            ),
            "metrics": summarize_pose_metrics(pred, gt, mask=group_mask),
        }
        result["groups"].append(group)

    json_result = jsonable(result)
    print(json.dumps(json_result, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
