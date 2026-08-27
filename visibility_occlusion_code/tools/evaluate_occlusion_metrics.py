"""Evaluate pose metrics grouped by automatic visibility labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_metrics import summarize_by_visibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input NPZ with pred_pose, gt_pose, visibility_label.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _jsonable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    if "visibility_label" not in data:
        raise KeyError("Input file does not contain visibility_label. Run generate_visibility_labels.py first.")

    summary = summarize_by_visibility(
        data["pred_pose"],
        data["gt_pose"],
        data["visibility_label"],
    )
    jsonable = _jsonable(summary)
    print(json.dumps(jsonable, indent=2, ensure_ascii=False))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(jsonable, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
