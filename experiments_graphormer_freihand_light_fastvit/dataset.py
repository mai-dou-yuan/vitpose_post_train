import argparse
import importlib.util
import json
import math
import os
import random
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_repo_datasets_package():
    existing = sys.modules.get("datasets")
    if existing is not None and getattr(existing, "__file__", None):
        existing_path = Path(existing.__file__).resolve()
        if DATASETS_DIR in existing_path.parents:
            return

    package = types.ModuleType("datasets")
    package.__path__ = [str(DATASETS_DIR)]
    sys.modules["datasets"] = package
    _load_module("datasets.dataset", DATASETS_DIR / "dataset.py")


_ensure_repo_datasets_package()

from datasets.dataset import Unrealego3DPoseDataset


class FreiHANDExperimentDataset(Dataset):
    """
    Standalone FreiHAND dataset for the experiment directory.

    Split policy:
    - train: 100% samples from training/rgb
    - val: 100% samples from evaluation/rgb
    - test: 100% samples from evaluation/rgb
    """

    SPLIT_ALIASES = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
        "eval": "evaluation",
        "evaluation": "evaluation",
    }

    def __init__(
        self,
        data_root="FreiHAND",
        split="train",
        img_size=518,
        use_annotation_uv=True,
        crop_padding=1.5,
        min_crop_size=32.0,
        augmentation=None,
        grouped_background_sampling=False,
        background_variant_weights=None,
    ):
        super().__init__()

        split_key = self.SPLIT_ALIASES.get(split)
        if split_key is None:
            raise ValueError(f"Unsupported split: {split}")

        self.data_root = data_root
        self.split = split_key
        self.img_size = int(img_size)
        self.use_annotation_uv = use_annotation_uv
        self.crop_padding = float(crop_padding)
        self.min_crop_size = float(min_crop_size)
        self.samples = self._build_samples()
        self.augmentation = dict(augmentation or {})
        self.augmentation_enabled = bool(
            self.split == "train" and self.augmentation.get("enabled", False)
        )
        self.grouped_background_sampling = bool(
            self.split == "train" and grouped_background_sampling
        )
        self.background_variant_weights = background_variant_weights
        self.num_background_variants = 1
        self.num_pose_groups = len(self.samples)
        if self.grouped_background_sampling:
            self._configure_grouped_background_sampling()

    def _configure_grouped_background_sampling(self):
        weights = self.background_variant_weights or [0.25, 0.25, 0.25, 0.25]
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or len(weights) < 2:
            raise ValueError("background_variant_weights must contain at least two values")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("background_variant_weights must be finite, non-negative, and non-zero")
        if len(self.samples) % len(weights) != 0:
            raise ValueError(
                f"Training sample count {len(self.samples)} is not divisible by "
                f"{len(weights)} background variants"
            )
        self.num_background_variants = len(weights)
        self.num_pose_groups = len(self.samples) // self.num_background_variants
        self.background_variant_weights = (weights / weights.sum()).tolist()

    def _list_sample_stems(self, split_dir):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"FreiHAND split directory not found: {split_dir}")

        stems = []
        for file_name in sorted(os.listdir(split_dir)):
            if not file_name.endswith(".json"):
                continue

            stem = os.path.splitext(file_name)[0]
            img_path = os.path.join(split_dir, stem + ".jpg")
            json_path = os.path.join(split_dir, file_name)
            if not os.path.isfile(img_path) or not os.path.isfile(json_path):
                continue
            stems.append(stem)

        if not stems:
            raise RuntimeError(f"No FreiHAND samples found under: {split_dir}")
        return stems

    def _build_samples_for_dir(self, split_name):
        rgb_dir = os.path.join(self.data_root, split_name, "rgb")
        stems = self._list_sample_stems(rgb_dir)
        return [
            {
                "img_relative_path": os.path.join(split_name, "rgb", stem + ".jpg"),
                "ann_relative_path": os.path.join(split_name, "rgb", stem + ".json"),
            }
            for stem in stems
        ]

    def _build_samples(self):
        if self.split == "train":
            return self._build_samples_for_dir("training")
        if self.split in ("val", "test", "evaluation"):
            return self._build_samples_for_dir("evaluation")
        raise ValueError(f"Unhandled split: {self.split}")

    def _load_annotation(self, ann_path):
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        pose_gt_3d = np.asarray(ann["xyz"], dtype=np.float32).reshape(21, 3)
        vertices_gt_3d = np.asarray(
            ann["vertices"], dtype=np.float32
        ).reshape(778, 3)
        cam_k = np.asarray(ann["K"], dtype=np.float32).reshape(3, 3)

        gt_pose_2d = None
        if "uv" in ann and ann["uv"] is not None:
            gt_pose_2d = np.asarray(ann["uv"], dtype=np.float32).reshape(21, 2)

        return pose_gt_3d, vertices_gt_3d, cam_k, gt_pose_2d

    def _project_3d_to_pixel(self, joints_3d, cam_k):
        return Unrealego3DPoseDataset._project_3d_to_pixel(self, joints_3d, cam_k, dist_coeffs=None)

    def _valid_projected_keypoints(self, joints_3d, keypoints_2d):
        joints_3d = np.asarray(joints_3d, dtype=np.float32)
        keypoints_2d = np.asarray(keypoints_2d, dtype=np.float32)
        return (
            np.isfinite(keypoints_2d).all(axis=1)
            & np.isfinite(joints_3d).all(axis=1)
            & (joints_3d[:, 2] > 1e-2)
            & (keypoints_2d[:, 0] > -1e5)
            & (keypoints_2d[:, 1] > -1e5)
        )

    def _bbox_from_projected_joints(self, joints_3d, keypoints_2d, image_shape):
        h, w = image_shape[:2]
        valid = self._valid_projected_keypoints(joints_3d, keypoints_2d)
        if not np.any(valid):
            return np.array([0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)

        pts = keypoints_2d[valid]
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))

        if not np.all(np.isfinite([x1, y1, x2, y2])):
            return np.array([0.0, 0.0, float(w - 1), float(h - 1)], dtype=np.float32)

        x1 = float(np.clip(x1, 0.0, max(float(w - 1), 0.0)))
        y1 = float(np.clip(y1, 0.0, max(float(h - 1), 0.0)))
        x2 = float(np.clip(x2, 0.0, max(float(w - 1), 0.0)))
        y2 = float(np.clip(y2, 0.0, max(float(h - 1), 0.0)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _crop_params_from_bbox(self, bbox, image_shape):
        h, w = image_shape[:2]
        x1, y1, x2, y2 = bbox.astype(np.float32)
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
        bbox_w = max(float(x2 - x1), 1.0)
        bbox_h = max(float(y2 - y1), 1.0)
        side = max(bbox_w, bbox_h) * self.crop_padding
        side = max(side, self.min_crop_size)
        side = min(side, float(max(w, h)) * 2.0)

        crop_x1 = float(center[0] - side * 0.5)
        crop_y1 = float(center[1] - side * 0.5)
        return center, side, crop_x1, crop_y1

    def _crop_resize_image(self, img, crop_x1, crop_y1, crop_side):
        resize_scale = self.img_size / crop_side
        affine = np.array(
            [
                [resize_scale, 0.0, -crop_x1 * resize_scale],
                [0.0, resize_scale, -crop_y1 * resize_scale],
            ],
            dtype=np.float32,
        )
        cropped = cv2.warpAffine(
            img,
            affine,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return cropped, affine, resize_scale

    def _sample_augmented_crop(self, center, crop_side):
        center_jitter = float(self.augmentation.get("center_jitter", 0.07))
        scale_min, scale_max = self.augmentation.get("scale_range", [0.85, 1.20])
        scale_min = float(scale_min)
        scale_max = float(scale_max)
        if scale_min <= 0 or scale_max < scale_min:
            raise ValueError(f"Invalid augmentation scale_range: {[scale_min, scale_max]}")

        jitter = np.array(
            [
                random.uniform(-center_jitter, center_jitter),
                random.uniform(-center_jitter, center_jitter),
            ],
            dtype=np.float32,
        )
        center = center + jitter * crop_side
        scale_factor = math.exp(random.uniform(math.log(scale_min), math.log(scale_max)))
        return center.astype(np.float32), float(crop_side * scale_factor)

    def _crop_affine(self, center, crop_side):
        crop_x1 = float(center[0] - crop_side * 0.5)
        crop_y1 = float(center[1] - crop_side * 0.5)
        resize_scale = self.img_size / crop_side
        affine = np.array(
            [
                [resize_scale, 0.0, -crop_x1 * resize_scale],
                [0.0, resize_scale, -crop_y1 * resize_scale],
            ],
            dtype=np.float32,
        )
        return affine, crop_x1, crop_y1, resize_scale

    def _apply_training_geometry(
        self,
        img,
        pose_gt_3d,
        vertices_gt_3d,
        cam_k_original,
        gt_pose_2d,
        center,
        crop_side,
    ):
        center, crop_side = self._sample_augmented_crop(center, crop_side)
        crop_affine, crop_x1, crop_y1, resize_scale = self._crop_affine(center, crop_side)
        crop_h = np.eye(3, dtype=np.float32)
        crop_h[:2] = crop_affine
        cam_k_new = crop_h @ cam_k_original

        rotation_probability = float(self.augmentation.get("rotation_probability", 0.6))
        max_rotation_degrees = float(self.augmentation.get("max_rotation_degrees", 30.0))
        angle = (
            random.uniform(-max_rotation_degrees, max_rotation_degrees)
            if random.random() < rotation_probability
            else 0.0
        )

        # OpenCV's positive image angle is counter-clockwise in image space.
        # FreiHAND camera coordinates use x-right/y-down, hence the negative
        # angle for the equivalent camera-Z rotation.
        theta = math.radians(-angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        rotation_3d = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pose_gt_3d = (pose_gt_3d @ rotation_3d.T).astype(np.float32)
        vertices_gt_3d = (vertices_gt_3d @ rotation_3d.T).astype(np.float32)

        rotation_h = cam_k_new @ rotation_3d @ np.linalg.inv(cam_k_new)
        image_h = rotation_h @ crop_h
        img = cv2.warpPerspective(
            img,
            image_h,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        if self.use_annotation_uv and gt_pose_2d is not None:
            gt_pose_2d = cv2.perspectiveTransform(
                np.asarray(gt_pose_2d, dtype=np.float32)[None, :, :], image_h
            )[0]
        else:
            gt_pose_2d = self._project_3d_to_pixel(pose_gt_3d, cam_k_new)

        return (
            img,
            pose_gt_3d,
            vertices_gt_3d,
            cam_k_new.astype(np.float32),
            gt_pose_2d.astype(np.float32),
            center,
            crop_side,
            crop_x1,
            crop_y1,
            resize_scale,
            angle,
        )

    @staticmethod
    def _adjust_brightness(img, factor):
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_contrast(img, factor):
        mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
        return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_saturation(img, factor):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None].astype(np.float32)
        return np.clip(
            (img.astype(np.float32) - gray) * factor + gray, 0, 255
        ).astype(np.uint8)

    @staticmethod
    def _adjust_hue(img, hue_fraction):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue_shift = int(round(hue_fraction * 180.0))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _apply_color_jitter(self, img):
        if random.random() >= float(self.augmentation.get("color_jitter_probability", 0.8)):
            return img
        brightness = float(self.augmentation.get("brightness", 0.20))
        contrast = float(self.augmentation.get("contrast", 0.20))
        saturation = float(self.augmentation.get("saturation", 0.15))
        hue = float(self.augmentation.get("hue", 0.03))
        operations = [
            lambda x: self._adjust_brightness(x, random.uniform(1.0 - brightness, 1.0 + brightness)),
            lambda x: self._adjust_contrast(x, random.uniform(1.0 - contrast, 1.0 + contrast)),
            lambda x: self._adjust_saturation(x, random.uniform(1.0 - saturation, 1.0 + saturation)),
            lambda x: self._adjust_hue(x, random.uniform(-hue, hue)),
        ]
        random.shuffle(operations)
        for operation in operations:
            img = operation(img)
        return img

    def _apply_degradation(self, img):
        if random.random() >= float(self.augmentation.get("degradation_probability", 0.25)):
            return img
        degradation = random.choices(
            ["gaussian_blur", "motion_blur", "noise", "jpeg"],
            weights=[0.35, 0.20, 0.30, 0.15],
            k=1,
        )[0]
        if degradation == "gaussian_blur":
            sigma_min, sigma_max = self.augmentation.get("gaussian_blur_sigma", [0.1, 1.2])
            sigma = random.uniform(float(sigma_min), float(sigma_max))
            return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if degradation == "motion_blur":
            kernel_size = random.choice([3, 5])
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            if random.random() < 0.5:
                kernel[kernel_size // 2, :] = 1.0 / kernel_size
            else:
                np.fill_diagonal(kernel, 1.0 / kernel_size)
            return cv2.filter2D(img, -1, kernel)
        if degradation == "noise":
            noise_min, noise_max = self.augmentation.get("noise_std", [0.005, 0.02])
            noise_std = random.uniform(float(noise_min), float(noise_max)) * 255.0
            noise = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        quality_min, quality_max = self.augmentation.get("jpeg_quality", [70, 95])
        quality = random.randint(int(quality_min), int(quality_max))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not ok:
            return img
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def _apply_occlusion(self, img, keypoints_2d):
        if random.random() >= float(self.augmentation.get("occlusion_probability", 0.20)):
            return img
        area_min, area_max = self.augmentation.get("occlusion_area_range", [0.02, 0.10])
        h, w = img.shape[:2]
        valid_points = np.asarray(keypoints_2d, dtype=np.float32)
        valid_points = valid_points[np.isfinite(valid_points).all(axis=1)]
        if len(valid_points) == 0:
            return img

        selected = None
        for _ in range(12):
            area = random.uniform(float(area_min), float(area_max)) * h * w
            aspect = math.exp(random.uniform(math.log(0.5), math.log(2.0)))
            rect_w = int(round(math.sqrt(area * aspect)))
            rect_h = int(round(math.sqrt(area / aspect)))
            anchor = valid_points[random.randrange(len(valid_points))]
            cx = float(anchor[0]) + random.uniform(-0.15, 0.15) * rect_w
            cy = float(anchor[1]) + random.uniform(-0.15, 0.15) * rect_h
            x1 = max(0, int(round(cx - rect_w / 2)))
            y1 = max(0, int(round(cy - rect_h / 2)))
            x2 = min(w, x1 + max(rect_w, 1))
            y2 = min(h, y1 + max(rect_h, 1))
            covered = (
                (valid_points[:, 0] >= x1)
                & (valid_points[:, 0] < x2)
                & (valid_points[:, 1] >= y1)
                & (valid_points[:, 1] < y2)
            ).sum()
            selected = (x1, y1, x2, y2)
            if 1 <= covered <= 3:
                break

        x1, y1, x2, y2 = selected
        if x2 <= x1 or y2 <= y1:
            return img
        image_mean = img.astype(np.float32).mean(axis=(0, 1))
        fill = np.random.normal(image_mean, 12.0, size=(y2 - y1, x2 - x1, 3))
        img = img.copy()
        img[y1:y2, x1:x2] = np.clip(fill, 0, 255).astype(np.uint8)
        return img

    def _apply_training_appearance(self, img, keypoints_2d):
        img = self._apply_color_jitter(img)
        img = self._apply_degradation(img)
        img = self._apply_occlusion(img, keypoints_2d)
        return img

    def _transform_keypoints(self, keypoints_2d, affine):
        pts = np.asarray(keypoints_2d, dtype=np.float32)
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
        return (pts_h @ affine.T).astype(np.float32)

    def _transform_camera_intrinsics(self, cam_k, crop_x1, crop_y1, resize_scale):
        cam_k_new = cam_k.copy()
        cam_k_new[0, 0] *= resize_scale
        cam_k_new[1, 1] *= resize_scale
        cam_k_new[0, 2] = (cam_k_new[0, 2] - crop_x1) * resize_scale
        cam_k_new[1, 2] = (cam_k_new[1, 2] - crop_y1) * resize_scale
        return cam_k_new

    def __len__(self):
        return self.num_pose_groups

    def _sample_for_index(self, idx):
        if not self.grouped_background_sampling:
            return self.samples[idx]
        variant = random.choices(
            range(self.num_background_variants),
            weights=self.background_variant_weights,
            k=1,
        )[0]
        return self.samples[idx + variant * self.num_pose_groups]

    def __getitem__(self, idx):
        sample = self._sample_for_index(idx)
        img_path = os.path.join(self.data_root, sample["img_relative_path"])
        ann_path = os.path.join(self.data_root, sample["ann_relative_path"])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (
            pose_gt_3d,
            vertices_gt_3d,
            cam_k_original,
            gt_pose_2d,
        ) = self._load_annotation(ann_path)

        projected_2d = self._project_3d_to_pixel(pose_gt_3d, cam_k_original)
        bbox = self._bbox_from_projected_joints(pose_gt_3d, projected_2d, img.shape)
        center, crop_side, crop_x1, crop_y1 = self._crop_params_from_bbox(bbox, img.shape)
        if self.augmentation_enabled:
            (
                img,
                pose_gt_3d,
                vertices_gt_3d,
                cam_k_new,
                gt_pose_2d,
                center,
                crop_side,
                crop_x1,
                crop_y1,
                resize_scale,
                _,
            ) = self._apply_training_geometry(
                img,
                pose_gt_3d,
                vertices_gt_3d,
                cam_k_original,
                gt_pose_2d,
                center,
                crop_side,
            )
            img = self._apply_training_appearance(img, gt_pose_2d)
        else:
            img, crop_affine, resize_scale = self._crop_resize_image(
                img, crop_x1, crop_y1, crop_side
            )
            cam_k_new = self._transform_camera_intrinsics(
                cam_k_original, crop_x1, crop_y1, resize_scale
            )

            if self.use_annotation_uv and gt_pose_2d is not None:
                gt_pose_2d = self._transform_keypoints(gt_pose_2d, crop_affine)
            else:
                gt_pose_2d = self._project_3d_to_pixel(pose_gt_3d, cam_k_new)

        hand_central_3d = pose_gt_3d[0].copy()

        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)

        return {
            "img": img_tensor,
            "gt_pose": torch.from_numpy(pose_gt_3d).float(),
            "gt_vertices": torch.from_numpy(vertices_gt_3d).float(),
            "origin_3d": torch.from_numpy(hand_central_3d).float(),
            "cam_k": torch.from_numpy(cam_k_new).float(),
            "dataset_idx": sample["img_relative_path"],
            "gt_pose_2d": torch.from_numpy(gt_pose_2d).float(),
            "crop_center": torch.from_numpy(center).float(),
            "crop_scale": torch.tensor(crop_side, dtype=torch.float32),
            "crop_bbox": torch.from_numpy(bbox).float(),
        }


# FreiHAND joint order: wrist, then thumb/index/middle/ring/little from base to tip.
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _draw_hand_keypoints(image, keypoints):
    """Draw FreiHAND 2D ground-truth joints on an RGB image."""
    canvas = image.copy()
    keypoints = np.asarray(keypoints, dtype=np.float32)
    valid = np.isfinite(keypoints).all(axis=1)
    h, w = canvas.shape[:2]

    for start, end in HAND_EDGES:
        if valid[start] and valid[end]:
            p1 = tuple(np.round(keypoints[start]).astype(int))
            p2 = tuple(np.round(keypoints[end]).astype(int))
            cv2.line(canvas, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
    for joint_idx, point in enumerate(keypoints):
        if not valid[joint_idx]:
            continue
        x, y = np.round(point).astype(int)
        if -4 <= x < w + 4 and -4 <= y < h + 4:
            cv2.circle(canvas, (x, y), 4, (255, 80, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _load_visualization_views(dataset, dataset_index):
    """Return matching baseline, geometry-only, and full-augmentation views."""
    sample = dataset._sample_for_index(dataset_index)
    img_path = os.path.join(dataset.data_root, sample["img_relative_path"])
    ann_path = os.path.join(dataset.data_root, sample["ann_relative_path"])
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    (
        pose_gt_3d,
        vertices_gt_3d,
        cam_k_original,
        gt_pose_2d,
    ) = dataset._load_annotation(ann_path)
    projected_2d = dataset._project_3d_to_pixel(pose_gt_3d, cam_k_original)
    bbox = dataset._bbox_from_projected_joints(pose_gt_3d, projected_2d, image.shape)
    center, crop_side, crop_x1, crop_y1 = dataset._crop_params_from_bbox(bbox, image.shape)

    baseline, crop_affine, _ = dataset._crop_resize_image(image, crop_x1, crop_y1, crop_side)
    if dataset.use_annotation_uv and gt_pose_2d is not None:
        baseline_keypoints = dataset._transform_keypoints(gt_pose_2d, crop_affine)
    else:
        baseline_keypoints = dataset._project_3d_to_pixel(pose_gt_3d, cam_k_original)

    if not dataset.augmentation_enabled:
        return sample["img_relative_path"], [
            ("standard crop", baseline, baseline_keypoints),
            ("validation input (no augmentation)", baseline.copy(), baseline_keypoints.copy()),
        ]

    (
        geometry_image,
        _,
        _,
        _,
        geometry_keypoints,
        _,
        _,
        _,
        _,
        _,
        angle,
    ) = dataset._apply_training_geometry(
        image.copy(),
        pose_gt_3d,
        vertices_gt_3d,
        cam_k_original,
        gt_pose_2d,
        center,
        crop_side,
    )
    full_image = dataset._apply_training_appearance(geometry_image.copy(), geometry_keypoints)
    return sample["img_relative_path"], [
        ("standard crop", baseline, baseline_keypoints),
        (f"geometry: rotation {angle:+.1f} deg", geometry_image, geometry_keypoints),
        ("full training augmentation", full_image, geometry_keypoints),
    ]


def _make_comparison_panel(views):
    panels = []
    for title, image, keypoints in views:
        panel = _draw_hand_keypoints(image, keypoints)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(panel, title, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def export_visualization_samples(dataset, count, output_dir, seed):
    """Export matching before/after augmentation panels for randomly selected samples."""
    if count > len(dataset):
        raise ValueError(f"Requested {count} samples, but dataset has only {len(dataset)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = random.Random(seed).sample(range(len(dataset)), count)
    manifest = []
    for sample_number, dataset_index in enumerate(indices, start=1):
        source_path, views = _load_visualization_views(dataset, dataset_index)
        visual = _make_comparison_panel(views)
        label = f"#{sample_number:02d}  index={dataset_index}  {source_path}"
        cv2.rectangle(visual, (0, visual.shape[0] - 20), (visual.shape[1], visual.shape[0]), (0, 0, 0), -1)
        cv2.putText(visual, label, (5, visual.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        filename = f"{sample_number:02d}_idx_{dataset_index:05d}.jpg"
        cv2.imwrite(str(output_dir / filename), cv2.cvtColor(visual, cv2.COLOR_RGB2BGR))
        manifest.append({"file": filename, "dataset_index": dataset_index, "source": source_path})

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return indices


def main():
    parser = argparse.ArgumentParser(
        description="Export augmented FreiHAND training and validation samples for visual inspection."
    )
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "freihand_graphormer.yaml"),
        help="Experiment YAML configuration path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "dataset_visualization_samples"),
        help="Directory that will contain train_augmented/ and val/ images.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling and augmentation random seed.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    data_root = Path(data_cfg["root"])
    if not data_root.is_absolute():
        data_root = (ROOT_DIR / data_root).resolve()
    common_kwargs = {
        "data_root": str(data_root),
        "img_size": data_cfg["image_size"],
        "use_annotation_uv": data_cfg.get("use_annotation_uv", True),
    }

    # Seed all random sources used by the augmentation pipeline for reproducible exports.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = FreiHANDExperimentDataset(
        split="train",
        augmentation=data_cfg.get("augmentation", {}),
        grouped_background_sampling=data_cfg.get("grouped_background_sampling", False),
        background_variant_weights=data_cfg.get("background_variant_weights"),
        **common_kwargs,
    )
    val_dataset = FreiHANDExperimentDataset(
        split=data_cfg.get("val_split", "evaluation"), **common_kwargs
    )

    output_dir = Path(args.output_dir)
    train_indices = export_visualization_samples(
        train_dataset, count=50, output_dir=output_dir / "train_augmented", seed=args.seed
    )
    val_indices = export_visualization_samples(
        val_dataset, count=15, output_dir=output_dir / "val", seed=args.seed + 1
    )
    print(f"Exported {len(train_indices)} augmented train samples to {output_dir / 'train_augmented'}")
    print(f"Exported {len(val_indices)} validation samples to {output_dir / 'val'}")


if __name__ == "__main__":
    main()
