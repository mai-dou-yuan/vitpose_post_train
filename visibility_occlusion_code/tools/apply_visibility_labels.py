"""Copy reference visibility labels into another exported prediction NPZ.

Use this when multiple methods are evaluated on the same test split. The
visibility labels should be generated once from GT/camera annotations and then
reused for every method, so visible/occluded groups stay identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


VISIBILITY_FIELDS = (
    "visibility_label",
    "projected_2d",
    "in_view_mask",
    "visible_joint_ratio",
    "occluder_index",
    "occluder_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Prediction NPZ of the method to evaluate.")
    parser.add_argument("--reference", required=True, help="Reference NPZ with visibility labels.")
    parser.add_argument("--output", required=True, help="Output NPZ with copied visibility labels.")
    return parser.parse_args()


def reorder_reference_field(field: np.ndarray, reference_indices: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    if np.array_equal(reference_indices, target_indices):
        return field

    index_to_row = {int(dataset_idx): row_idx for row_idx, dataset_idx in enumerate(reference_indices)}
    missing = [int(dataset_idx) for dataset_idx in target_indices if int(dataset_idx) not in index_to_row]
    if missing:
        raise KeyError(f"Reference file is missing dataset_idx values, first missing: {missing[:10]}")
    order = np.array([index_to_row[int(dataset_idx)] for dataset_idx in target_indices], dtype=np.int64)
    return field[order]


def main() -> None:
    args = parse_args()
    source = dict(np.load(args.input, allow_pickle=True))
    reference = np.load(args.reference, allow_pickle=True)

    if "dataset_idx" not in source:
        raise KeyError("Input prediction file does not contain dataset_idx.")
    if "dataset_idx" not in reference:
        raise KeyError("Reference visibility file does not contain dataset_idx.")

    reference_indices = np.asarray(reference["dataset_idx"])
    target_indices = np.asarray(source["dataset_idx"])

    for field in VISIBILITY_FIELDS:
        if field not in reference.files:
            continue
        source[field] = reorder_reference_field(reference[field], reference_indices, target_indices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **source)
    print(f"Saved prediction file with reference visibility labels to {output_path}")


if __name__ == "__main__":
    main()
