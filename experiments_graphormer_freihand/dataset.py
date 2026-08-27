import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[0]
DATASETS_DIR = ROOT_DIR / "datasets"


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
        cam_k = np.asarray(ann["K"], dtype=np.float32).reshape(3, 3)

        gt_pose_2d = None
        if "uv" in ann and ann["uv"] is not None:
            gt_pose_2d = np.asarray(ann["uv"], dtype=np.float32).reshape(21, 2)

        return pose_gt_3d, cam_k, gt_pose_2d

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
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_root, sample["img_relative_path"])
        ann_path = os.path.join(self.data_root, sample["ann_relative_path"])

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_gt_3d, cam_k_original, gt_pose_2d = self._load_annotation(ann_path)
        hand_central_3d = pose_gt_3d[0].copy()

        projected_2d = self._project_3d_to_pixel(pose_gt_3d, cam_k_original)
        bbox = self._bbox_from_projected_joints(pose_gt_3d, projected_2d, img.shape)
        center, crop_side, crop_x1, crop_y1 = self._crop_params_from_bbox(bbox, img.shape)
        img, crop_affine, resize_scale = self._crop_resize_image(img, crop_x1, crop_y1, crop_side)
        cam_k_new = self._transform_camera_intrinsics(cam_k_original, crop_x1, crop_y1, resize_scale)

        if self.use_annotation_uv and gt_pose_2d is not None:
            gt_pose_2d = self._transform_keypoints(gt_pose_2d, crop_affine)
        else:
            gt_pose_2d = self._project_3d_to_pixel(pose_gt_3d, cam_k_new)

        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)

        return {
            "img": img_tensor,
            "gt_pose": torch.from_numpy(pose_gt_3d).float(),
            "origin_3d": torch.from_numpy(hand_central_3d).float(),
            "cam_k": torch.from_numpy(cam_k_new).float(),
            "dataset_idx": sample["img_relative_path"],
            "gt_pose_2d": torch.from_numpy(gt_pose_2d).float(),
            "crop_center": torch.from_numpy(center).float(),
            "crop_scale": torch.tensor(crop_side, dtype=torch.float32),
            "crop_bbox": torch.from_numpy(bbox).float(),
        }
