"""Visualize automatic joint visibility labels on wrist-camera images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.dataset import Unrealego3DPoseDataset
from utils.visibility_proxy import HAND_BONES, LABEL_NAMES, OUT_OF_VIEW


LABEL_COLORS_RGB: Dict[int, tuple] = {
    0: (40, 170, 80),    # visible
    1: (220, 65, 55),    # self-occluded
    2: (130, 130, 130),  # out-of-view
    3: (230, 180, 35),   # uncertain
}
BONE_COLOR_RGB = (55, 125, 190)


def parse_user_ids(text: str) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_indices(text: str) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_data_root(config_path: Path) -> str:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["data"]["root"]


def tensor_image_to_uint8(sample: dict) -> np.ndarray:
    image = sample["img"]
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def draw_sample(
    image_rgb: np.ndarray,
    points_2d: np.ndarray,
    labels: np.ndarray,
    draw_index: bool = False,
) -> np.ndarray:
    canvas = image_rgb.copy()
    points = np.asarray(points_2d, dtype=np.float32)

    for start, end in HAND_BONES:
        if labels[start] == OUT_OF_VIEW or labels[end] == OUT_OF_VIEW:
            continue
        if not np.all(np.isfinite(points[[start, end]])):
            continue
        p0 = tuple(np.round(points[start]).astype(int))
        p1 = tuple(np.round(points[end]).astype(int))
        cv2.line(canvas, p0, p1, BONE_COLOR_RGB, 2, lineType=cv2.LINE_AA)

    for joint_idx, point in enumerate(points):
        if not np.all(np.isfinite(point)):
            continue
        x, y = np.round(point).astype(int)
        if x < -20 or y < -20 or x >= canvas.shape[1] + 20 or y >= canvas.shape[0] + 20:
            continue
        color = LABEL_COLORS_RGB.get(int(labels[joint_idx]), LABEL_COLORS_RGB[3])
        cv2.circle(canvas, (x, y), 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        if draw_index:
            cv2.putText(
                canvas,
                str(joint_idx),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    counts = {LABEL_NAMES[k]: int((labels == k).sum()) for k in LABEL_NAMES}
    caption = (
        f"visible={counts['visible']}  "
        f"occluded={counts['self_occluded']}  "
        f"out={counts['out_of_view']}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        caption,
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="NPZ produced by generate_visibility_labels.py.")
    parser.add_argument("--out-dir", required=True, help="Directory for visualization images.")
    parser.add_argument("--data-root", default="", help="Dataset root. Defaults to configs/config.yaml:data.root.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--target-user-ids", default="", help="Must match the export split, e.g. 4,8,13.")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--indices", default="", help="Comma-separated rows in the NPZ to visualize.")
    parser.add_argument("--draw-index", action="store_true", help="Draw joint ids next to points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.npz, allow_pickle=True)
    required = {"visibility_label", "projected_2d", "dataset_idx"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Missing fields in {args.npz}: {sorted(missing)}")

    data_root = args.data_root or load_data_root(PROJECT_ROOT / args.config)
    dataset_kwargs = {
        "data_root": data_root,
        "img_size": args.image_size,
        "is_train": False,
    }
    target_user_ids = parse_user_ids(args.target_user_ids)
    if target_user_ids:
        dataset_kwargs["target_user_ids"] = target_user_ids
    dataset = Unrealego3DPoseDataset(**dataset_kwargs)

    row_indices = parse_indices(args.indices)
    if not row_indices:
        row_indices = list(range(min(args.num, len(data["dataset_idx"]))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row_idx in row_indices:
        dataset_idx = int(data["dataset_idx"][row_idx])
        sample = dataset[dataset_idx]
        image_rgb = tensor_image_to_uint8(sample)
        canvas_rgb = draw_sample(
            image_rgb,
            data["projected_2d"][row_idx],
            data["visibility_label"][row_idx],
            draw_index=args.draw_index,
        )
        output_path = out_dir / f"visibility_row_{row_idx:05d}_dataset_{dataset_idx:05d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(row_indices)} visibility visualizations to {out_dir}")


if __name__ == "__main__":
    main()
