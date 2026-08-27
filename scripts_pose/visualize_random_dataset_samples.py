import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset import Unrealego3DPoseDataset
from visibility_occlusion_code.utils.visibility_proxy import HAND_BONES


LABEL_COLORS_RGB = {
    0: (40, 170, 80),
    1: (220, 65, 55),
    2: (130, 130, 130),
    3: (230, 180, 35),
}
BONE_COLOR_RGB = (55, 125, 190)

VISIBILITY_PRESETS = {
    "conservative": {
        "finger_scale": 0.08,
        "finger_min": 0.5,
        "finger_max": 2.5,
        "palm_scale": 2.0,
        "palm_min": 1.0,
        "palm_max": 5.0,
        "depth_scale": 1.0,
        "depth_min": 0.5,
        "depth_max": 2.5,
    },
    "slightly_conservative": {
        "finger_scale": 0.10,
        "finger_min": 0.5,
        "finger_max": 3.0,
        "palm_scale": 2.0,
        "palm_min": 1.0,
        "palm_max": 6.0,
        "depth_scale": 1.0,
        "depth_min": 0.5,
        "depth_max": 3.0,
    },
    "balanced": {
        "finger_scale": 0.12,
        "finger_min": 0.5,
        "finger_max": 3.5,
        "palm_scale": 2.0,
        "palm_min": 1.0,
        "palm_max": 6.5,
        "depth_scale": 1.0,
        "depth_min": 0.5,
        "depth_max": 3.5,
    },
    "slightly_aggressive": {
        "finger_scale": 0.14,
        "finger_min": 0.5,
        "finger_max": 3.5,
        "palm_scale": 2.0,
        "palm_min": 1.0,
        "palm_max": 6.5,
        "depth_scale": 1.0,
        "depth_min": 0.5,
        "depth_max": 3.5,
    },
    "aggressive": {
        "finger_scale": 0.16,
        "finger_min": 0.5,
        "finger_max": 3.5,
        "palm_scale": 2.0,
        "palm_min": 1.0,
        "palm_max": 6.5,
        "depth_scale": 1.0,
        "depth_min": 0.5,
        "depth_max": 3.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomly visualize dataset samples.")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Dataset root. Defaults to data.root in config.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT_DIR / "outputs" / "random_dataset_vis_25"),
        help="Directory to save visualizations.",
    )
    parser.add_argument("--num", type=int, default=25, help="Number of random samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--is-train",
        action="store_true",
        help="Use training mode dataset. Default is eval mode without augmentation.",
    )
    parser.add_argument(
        "--target-user-ids",
        default="",
        help="Comma separated user ids. Defaults to dataset.py constructor behavior.",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Generate one folder per threshold preset.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_user_ids(text: str):
    if not text.strip():
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def tensor_image_to_uint8(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def draw_sample(sample: dict, dataset_idx: int) -> np.ndarray:
    image_rgb = tensor_image_to_uint8(sample["img"])
    points_2d = to_numpy(sample["gt_pose_2d"]).astype(np.float32)
    labels = to_numpy(sample["visibility_label"]).astype(np.int64)
    visible_ratio = float(to_numpy(sample["visible_joint_ratio"]).reshape(-1)[0])

    canvas = image_rgb.copy()
    for start, end in HAND_BONES:
        if labels[start] == 2 or labels[end] == 2:
            continue
        p0 = tuple(np.round(points_2d[start]).astype(int))
        p1 = tuple(np.round(points_2d[end]).astype(int))
        cv2.line(canvas, p0, p1, BONE_COLOR_RGB, 2, lineType=cv2.LINE_AA)

    for joint_idx, point in enumerate(points_2d):
        if not np.all(np.isfinite(point)):
            continue
        x, y = np.round(point).astype(int)
        if x < -20 or y < -20 or x >= canvas.shape[1] + 20 or y >= canvas.shape[0] + 20:
            continue
        color = LABEL_COLORS_RGB.get(int(labels[joint_idx]), LABEL_COLORS_RGB[3])
        cv2.circle(canvas, (x, y), 5, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(joint_idx),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

    counts = {
        "visible": int((labels == 0).sum()),
        "occluded": int((labels == 1).sum()),
        "out": int((labels == 2).sum()),
    }
    caption = (
        f"idx={dataset_idx}  visible={counts['visible']}  "
        f"occluded={counts['occluded']}  out={counts['out']}  ratio={visible_ratio:.3f}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        caption,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return canvas


def build_montage(images):
    if not images:
        raise ValueError("No images to montage.")
    cell_h, cell_w = images[0].shape[:2]
    cols = int(math.ceil(math.sqrt(len(images))))
    rows = int(math.ceil(len(images) / cols))
    montage = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        montage[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = image
    return montage


def build_dataset(data_root, image_size, is_train, target_user_ids, visibility_profile):
    dataset_kwargs = {
        "data_root": data_root,
        "img_size": image_size,
        "is_train": is_train,
        "visibility_profile": visibility_profile,
    }
    if target_user_ids is not None:
        dataset_kwargs["target_user_ids"] = target_user_ids
    return Unrealego3DPoseDataset(**dataset_kwargs)


def save_visualizations(dataset, sampled_indices, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_images = []
    for order_idx, dataset_idx in enumerate(sampled_indices):
        sample = dataset[dataset_idx]
        canvas_rgb = draw_sample(sample, dataset_idx)
        saved_images.append(canvas_rgb)
        output_path = out_dir / f"sample_{order_idx:02d}_dataset_{dataset_idx:05d}.jpg"
        cv2.imwrite(str(output_path), cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))

    montage = build_montage(saved_images)
    cv2.imwrite(str(out_dir / "montage.jpg"), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))

    with (out_dir / "sample_indices.txt").open("w", encoding="utf-8") as handle:
        for dataset_idx in sampled_indices:
            handle.write(f"{dataset_idx}\n")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = args.data_root or config["data"]["root"]
    image_size = int(config["data"]["image_size"])

    user_ids = parse_user_ids(args.target_user_ids)
    dataset = build_dataset(
        data_root,
        image_size,
        args.is_train,
        user_ids,
        visibility_profile=None,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    num_samples = min(args.num, len(dataset))
    rng = random.Random(args.seed)
    sampled_indices = rng.sample(range(len(dataset)), k=num_samples)

    out_dir = Path(args.out_dir)
    if args.all_presets:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "sample_indices.txt").open("w", encoding="utf-8") as handle:
            for dataset_idx in sampled_indices:
                handle.write(f"{dataset_idx}\n")

        for preset_name, visibility_profile in VISIBILITY_PRESETS.items():
            preset_dataset = build_dataset(
                data_root,
                image_size,
                args.is_train,
                user_ids,
                visibility_profile=visibility_profile,
            )
            save_visualizations(preset_dataset, sampled_indices, out_dir / preset_name)
            print(f"Saved preset '{preset_name}' to: {out_dir / preset_name}")
    else:
        save_visualizations(dataset, sampled_indices, out_dir)

    print(f"Dataset root: {data_root}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Randomly sampled indices ({len(sampled_indices)}): {sampled_indices}")
    print(f"Saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
