"""Export deterministic visual checks from the filtered full DexYCB dataset."""

import argparse
import csv
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = THIS_DIR.parent
ROOT_DIR = EXPERIMENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_graphormer_dexycb_light_fastvit.config_utils import (
    build_dataset,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_dexycb_light_fastvit.dataset import (
    FORBIDDEN_DATA_PARTS,
    HAND_EDGES,
)


FINGER_COLORS_RGB = (
    (255, 96, 96),
    (255, 184, 72),
    (80, 220, 130),
    (64, 180, 255),
    (190, 120, 255),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(EXPERIMENT_DIR / "configs" / "dexycb_graphormer.yaml"),
    )
    parser.add_argument("--setup", default=None, choices=("s0", "s1", "s2", "s3"))
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--count", "--num-samples", dest="count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to outputs/full_<setup>_<split>_100_seed_42.",
    )
    return parser.parse_args()


def save_rgb(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write image: {path}")


def tensor_image_to_rgb(image_tensor):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def draw_joints(image, joints_2d):
    canvas = image.copy()
    joints_2d = np.asarray(joints_2d, dtype=np.float32)
    valid = np.isfinite(joints_2d).all(axis=1)
    height, width = canvas.shape[:2]
    in_frame = (
        valid
        & (joints_2d[:, 0] >= 0)
        & (joints_2d[:, 0] < width)
        & (joints_2d[:, 1] >= 0)
        & (joints_2d[:, 1] < height)
    )
    for edge_index, (start, end) in enumerate(HAND_EDGES):
        if in_frame[start] and in_frame[end]:
            color = FINGER_COLORS_RGB[min(edge_index // 4, 4)]
            p1 = tuple(np.rint(joints_2d[start]).astype(int))
            p2 = tuple(np.rint(joints_2d[end]).astype(int))
            cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)
    for index, point in enumerate(joints_2d):
        if in_frame[index]:
            center = tuple(np.rint(point).astype(int))
            cv2.circle(canvas, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 3, (15, 15, 15), -1, cv2.LINE_AA)
    return canvas, int(in_frame.sum())


def add_title(image, title):
    canvas = np.pad(image, ((30, 0), (0, 0), (0, 0)), constant_values=18)
    cv2.putText(
        canvas,
        title,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def choose_diverse_indices(dataset, count, seed):
    """Prefer distinct subject/sequence/camera groups, then fill randomly."""
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    chosen = []
    seen_groups = set()
    for index in indices:
        sample = dataset.samples[index]
        group = (sample["subject"], sample["sequence"], sample["camera_serial"])
        if group not in seen_groups:
            seen_groups.add(group)
            chosen.append(index)
            if len(chosen) == count:
                return chosen
    chosen_set = set(chosen)
    chosen.extend(index for index in indices if index not in chosen_set)
    return chosen[:count]


def make_views(dataset, dataset_index):
    record = dataset.samples[dataset_index]
    if FORBIDDEN_DATA_PARTS.intersection(Path(record["img_relative_path"]).parts):
        raise RuntimeError(f"Forbidden sampled path selected: {record['img_relative_path']}")
    original_bgr = cv2.imread(str(record["img_path"]), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(record["img_path"])
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    with np.load(record["ann_path"]) as annotation:
        annotation_joints = np.asarray(annotation["joint_2d"], dtype=np.float32).reshape(21, 2)
        segmentation = np.asarray(annotation["seg"])

    joints_3d, _, cam_k, _ = dataset._load_annotation(record)
    projected = dataset._project_3d_to_pixel(joints_3d, cam_k)
    bbox = dataset._bbox_from_projected_joints(joints_3d, projected, original.shape)
    center, crop_side, crop_x1, crop_y1 = dataset._crop_params_from_bbox(bbox, original.shape)
    baseline, affine, _ = dataset._crop_resize_image(original, crop_x1, crop_y1, crop_side)
    baseline_joints = dataset._transform_keypoints(annotation_joints, affine)

    original_view, original_visible = draw_joints(original, annotation_joints)
    padded_box = np.rint(
        [crop_x1, crop_y1, crop_x1 + crop_side, crop_y1 + crop_side]
    ).astype(int)
    cv2.rectangle(
        original_view,
        tuple(padded_box[:2]),
        tuple(padded_box[2:]),
        (64, 255, 255),
        3,
    )
    baseline_view, baseline_visible = draw_joints(baseline, baseline_joints)

    mask = (segmentation == 255).astype(np.uint8) * 255
    mask_crop = cv2.warpAffine(
        mask,
        affine,
        (dataset.img_size, dataset.img_size),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_overlay = baseline.copy()
    hand_pixels = mask_crop > 0
    mask_overlay[hand_pixels] = (
        0.45 * mask_overlay[hand_pixels] + 0.55 * np.array([255, 70, 70])
    ).astype(np.uint8)

    processed = dataset[dataset_index]
    augmented = tensor_image_to_rgb(processed["img"])
    augmented_view, augmented_visible = draw_joints(
        augmented, processed["gt_pose_2d"].numpy()
    )

    target_size = dataset.img_size
    original_square = cv2.resize(
        original_view, (target_size, target_size), interpolation=cv2.INTER_AREA
    )
    views = (
        add_title(original_square, "original + joints + crop"),
        add_title(baseline_view, "unaugmented crop + joints"),
        add_title(augmented_view, "training augmentation + joints"),
        add_title(mask_overlay, "hand segmentation in crop"),
    )
    metadata = {
        "dataset_index": dataset_index,
        "source": record["img_relative_path"],
        "annotation": str(record["ann_path"].relative_to(dataset.data_root)),
        "subject": record["subject"],
        "sequence": record["sequence"],
        "camera_serial": record["camera_serial"],
        "frame_id": record["frame_id"],
        "mano_side": record["mano_side"],
        "crop_center": center.tolist(),
        "crop_side": float(crop_side),
        "crop_xyxy": padded_box.tolist(),
        "original_visible_joints": original_visible,
        "crop_visible_joints": baseline_visible,
        "augmented_visible_joints": augmented_visible,
        "hand_pixels_in_crop": int(hand_pixels.sum()),
    }
    return views, metadata


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)

    config = normalize_config_paths(load_config(args.config))
    if args.setup is not None:
        config["data"]["setup"] = args.setup
    dataset = build_dataset(config, args.split)
    if args.count > len(dataset):
        raise ValueError(f"Requested {args.count} samples from {len(dataset)}")

    setup = config["data"].get("setup", "s0")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        THIS_DIR / "outputs" / f"full_{setup}_{args.split}_{args.count}_seed_{args.seed}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    indices = choose_diverse_indices(dataset, args.count, args.seed)
    manifest = {
        "seed": args.seed,
        "setup": setup,
        "split": args.split,
        "count": args.count,
        "data_root": str(dataset.data_root.resolve()),
        "dataset_audit": dataset.audit,
        "samples": [],
    }
    overview_rows = []
    for number, dataset_index in enumerate(indices, 1):
        views, metadata = make_views(dataset, dataset_index)
        sample_dir = output_dir / "samples" / f"sample_{number:03d}"
        names = (
            "01_original_crop_box.jpg",
            "02_unaugmented_crop.jpg",
            "03_augmented_crop.jpg",
            "04_segmentation_crop.jpg",
        )
        for name, view in zip(names, views):
            save_rgb(sample_dir / name, view)
        panel = np.concatenate(views, axis=1)
        save_rgb(sample_dir / "00_panel.jpg", panel)
        overview_rows.append(panel)
        metadata.update(
            {"sample_number": number, "output_folder": str(sample_dir.relative_to(output_dir))}
        )
        manifest["samples"].append(metadata)
        print(
            f"[{number:03d}/{args.count}] {metadata['source']} "
            f"hand_pixels={metadata['hand_pixels_in_crop']}"
        )

    for start in range(0, len(overview_rows), 10):
        page = np.concatenate(overview_rows[start : start + 10], axis=0)
        save_rgb(output_dir / f"overview_{start // 10 + 1:02d}.jpg", page)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, ensure_ascii=False, indent=2)
    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as output_file:
        fields = tuple(manifest["samples"][0])
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest["samples"])
    print(f"output_dir={output_dir}")
    print(f"samples={len(manifest['samples'])}")
    print("visualization=PASSED")


if __name__ == "__main__":
    main()
