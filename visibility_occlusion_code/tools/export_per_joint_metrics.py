"""Export per-joint pose errors for all/visible/occluded hand joints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_metrics import (
    FINGERTIP_INDICES,
    OCCLUDED_LABEL,
    VISIBLE_LABEL,
    OUT_OF_VIEW_LABEL,
    joint_errors,
)


JOINT_NAMES = [
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(values[valid].mean())


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
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    errors = joint_errors(data["pred_pose"], data["gt_pose"])
    labels = data["visibility_label"]

    rows: List[Dict[str, Any]] = []
    for joint_idx in range(errors.shape[1]):
        joint_visible = labels[:, joint_idx] == VISIBLE_LABEL
        joint_occluded = labels[:, joint_idx] == OCCLUDED_LABEL
        joint_out = labels[:, joint_idx] == OUT_OF_VIEW_LABEL
        joint_in_view = joint_visible | joint_occluded

        visible_mpjpe = masked_mean(errors[:, joint_idx], joint_visible)
        occluded_mpjpe = masked_mean(errors[:, joint_idx], joint_occluded)
        row = {
            "joint_index": joint_idx,
            "joint_name": JOINT_NAMES[joint_idx] if joint_idx < len(JOINT_NAMES) else f"joint_{joint_idx}",
            "is_fingertip": joint_idx in FINGERTIP_INDICES,
            "all_in_view_mpjpe": masked_mean(errors[:, joint_idx], joint_in_view),
            "visible_mpjpe": visible_mpjpe,
            "occluded_mpjpe": occluded_mpjpe,
            "occluded_minus_visible": (
                float(occluded_mpjpe - visible_mpjpe)
                if np.isfinite(visible_mpjpe) and np.isfinite(occluded_mpjpe)
                else float("nan")
            ),
            "visible_count": int(joint_visible.sum()),
            "occluded_count": int(joint_occluded.sum()),
            "out_of_view_count": int(joint_out.sum()),
        }
        rows.append(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(jsonable(rows), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved per-joint metrics to {output_csv}")


if __name__ == "__main__":
    main()
