"""Generate automatic joint visibility labels for exported prediction files.

Expected input NPZ fields:
    gt_pose: [N, 21, 3]
    cam_k: [N, 3, 3]
    dist_coeffs: [N, 5] or [N, 1, 5]

The output NPZ keeps all original fields and adds:
    visibility_label: [N, 21]
    projected_2d: [N, 21, 2]
    in_view_mask: [N, 21]
    visible_joint_ratio: [N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.visibility_proxy import VisibilityConfig, classify_batch_visibility, visible_joint_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input NPZ prediction/GT file.")
    parser.add_argument("--output", required=True, help="Output NPZ file with visibility labels.")
    parser.add_argument("--image-size", type=int, default=336, help="Square image size after preprocessing.")
    parser.add_argument("--height", type=int, default=None, help="Image height. Overrides --image-size.")
    parser.add_argument("--width", type=int, default=None, help="Image width. Overrides --image-size.")
    parser.add_argument("--finger-radius", type=float, default=5.0)
    parser.add_argument("--joint-radius", type=float, default=5.0)
    parser.add_argument("--palm-radius", type=float, default=10.0)
    parser.add_argument("--depth-margin", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    data = dict(np.load(input_path, allow_pickle=True))

    height = args.height if args.height is not None else args.image_size
    width = args.width if args.width is not None else args.image_size
    config = VisibilityConfig(
        image_size=(height, width),
        finger_radius=args.finger_radius,
        joint_radius=args.joint_radius,
        palm_radius=args.palm_radius,
        depth_margin=args.depth_margin,
    )

    labels = classify_batch_visibility(
        poses_3d=data["gt_pose"],
        cam_ks=data["cam_k"],
        dist_coeffs=data.get("dist_coeffs"),
        config=config,
    )
    data.update(labels)
    data["visible_joint_ratio"] = visible_joint_ratio(labels["visibility_label"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
    print(f"Saved visibility labels to {output_path}")


if __name__ == "__main__":
    main()
