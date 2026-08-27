from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class DexYCBSampledDataset(Dataset):
    """
    Dataset reader for the materialized DexYCB subset.

    Supported layouts:
    1. data_root/<split>/<subject>/<sequence>/<camera>/color_xxxxxx.jpg
    2. data_root/<subject>/<sequence>/<camera>/color_xxxxxx.jpg
    """

    def __init__(self, data_root, split, img_size=336):
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.data_root = Path(data_root)
        self.split = split
        self.img_size = int(img_size)

        if not self.data_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")

        self.intrinsics_map = self._load_intrinsics()
        self.samples = self._load_samples()

    def _load_intrinsics(self):
        intrinsics_dir = self.data_root / "calibration" / "intrinsics"
        if not intrinsics_dir.is_dir():
            raise FileNotFoundError(f"Intrinsics directory not found: {intrinsics_dir}")

        intrinsics_map = {}
        for intr_path in sorted(intrinsics_dir.glob("*.yml")):
            serial = intr_path.name.split("_")[0]
            with open(intr_path, "r", encoding="utf-8") as f:
                intr_data = yaml.load(f, Loader=yaml.FullLoader)

            color_intr = intr_data["color"]
            intrinsics_map[serial] = np.array(
                [
                    [color_intr["fx"], 0.0, color_intr["ppx"]],
                    [0.0, color_intr["fy"], color_intr["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        if not intrinsics_map:
            raise RuntimeError(f"No intrinsics found under: {intrinsics_dir}")
        return intrinsics_map

    def _find_split_file(self):
        candidates = [
            self.data_root / "splits" / f"{self.split}.txt",
            self.data_root / f"{self.split}.txt",
        ]
        for split_file in candidates:
            if split_file.is_file():
                return split_file
        raise FileNotFoundError(
            f"Split file for '{self.split}' not found under {self.data_root}. "
            f"Tried: {', '.join(str(p) for p in candidates)}"
        )

    def _resolve_img_path(self, relative_path):
        rel_path = Path(relative_path)
        candidates = [
            self.data_root / rel_path,
            self.data_root / self.split / rel_path,
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Image not found for split '{self.split}': {relative_path}"
        )

    @staticmethod
    def _to_label_relative_path(img_relative_path):
        rel_str = str(img_relative_path).replace("/color_", "/labels_")
        return Path(rel_str).with_suffix(".npz")

    def _resolve_label_path(self, img_relative_path):
        label_relative_path = self._to_label_relative_path(img_relative_path)
        candidates = [
            self.data_root / label_relative_path,
            self.data_root / self.split / label_relative_path,
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"Label not found for split '{self.split}': {label_relative_path}"
        )

    def _load_samples(self):
        split_file = self._find_split_file()
        samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue

                img_path = self._resolve_img_path(rel_path)
                label_path = self._resolve_label_path(rel_path)
                samples.append(
                    {
                        "rel_path": rel_path,
                        "img_path": img_path,
                        "label_path": label_path,
                    }
                )

        if not samples:
            raise RuntimeError(f"No samples found in split file: {split_file}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_relative_path = sample["rel_path"]
        img_path = sample["img_path"]
        label_path = sample["label_path"]

        camera_serial = Path(img_relative_path).parts[-2]
        if camera_serial not in self.intrinsics_map:
            raise KeyError(f"Missing intrinsics for camera serial: {camera_serial}")

        cam_k_original = self.intrinsics_map[camera_serial].copy()

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        old_h, old_w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))

        with np.load(label_path) as label_data:
            pose_gt_3d = np.asarray(label_data["joint_3d"], dtype=np.float32).reshape(21, 3)
            gt_pose_2d = np.asarray(label_data["joint_2d"], dtype=np.float32).reshape(21, 2)

        hand_central_3d = pose_gt_3d[0].copy()

        scale_x = self.img_size / float(old_w)
        scale_y = self.img_size / float(old_h)

        cam_k_new = cam_k_original.copy()
        cam_k_new[0, 0] *= scale_x
        cam_k_new[0, 2] *= scale_x
        cam_k_new[1, 1] *= scale_y
        cam_k_new[1, 2] *= scale_y

        gt_pose_2d = gt_pose_2d.copy()
        gt_pose_2d[:, 0] *= scale_x
        gt_pose_2d[:, 1] *= scale_y

        img = img.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)

        return {
            "img": img_tensor,
            "gt_pose": torch.from_numpy(pose_gt_3d).float(),
            "origin_3d": torch.from_numpy(hand_central_3d).float(),
            "cam_k": torch.from_numpy(cam_k_new).float(),
            "dataset_idx": img_relative_path,
            "gt_pose_2d": torch.from_numpy(gt_pose_2d).float(),
        }
