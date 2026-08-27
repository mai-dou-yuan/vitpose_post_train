import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent


from experiments_graphormer_dexycb_light_fastvit.mano import MANORightModel


# Keep these explicit, matching NVlabs/dex-ycb-toolkit.  In particular, do not
# discover subjects by walking data_root: it also contains local sampled data.
DEXYCB_SUBJECTS = (
    "20200709-subject-01",
    "20200813-subject-02",
    "20200820-subject-03",
    "20200903-subject-04",
    "20200908-subject-05",
    "20200918-subject-06",
    "20200928-subject-07",
    "20201002-subject-08",
    "20201015-subject-09",
    "20201022-subject-10",
)
DEXYCB_SERIALS = (
    "836212060125",
    "839512060362",
    "840412060917",
    "841412060263",
    "932122060857",
    "932122060861",
    "932122061900",
    "932122062010",
)
FORBIDDEN_DATA_PARTS = {
    "sampled_30k_3p8k_seed42",
    "splits_30k_3p8k_seed42",
}


def official_dexycb_selection(setup, split):
    """Return official subject/camera/sequence indices for DexYCB s0-s3."""
    setup = str(setup).lower()
    split = DexYCBLightFastViTDataset.SPLIT_ALIASES.get(str(split).lower(), split)
    if setup not in {"s0", "s1", "s2", "s3"}:
        raise ValueError(f"Unsupported DexYCB setup: {setup}")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported DexYCB split: {split}")

    if setup == "s0":
        subject_indices = (
            range(10)
            if split == "train"
            else (range(2) if split == "val" else range(2, 10))
        )
        serial_indices = range(8)
        sequence_indices = [i for i in range(100) if (i % 5 != 4) == (split == "train")]
    elif setup == "s1":
        subject_indices = {"train": (0, 1, 2, 3, 4, 5, 9), "val": (6,), "test": (7, 8)}[split]
        serial_indices = range(8)
        sequence_indices = range(100)
    elif setup == "s2":
        subject_indices = range(10)
        serial_indices = {"train": range(6), "val": (6,), "test": (7,)}[split]
        sequence_indices = range(100)
    else:
        subject_indices = range(10)
        serial_indices = range(8)
        held_out = {"val": (3, 19), "test": (7, 11, 15)}
        if split == "train":
            sequence_indices = [i for i in range(100) if i // 5 not in (3, 7, 11, 15, 19)]
        else:
            sequence_indices = [i for i in range(100) if i // 5 in held_out[split]]

    return {
        "subjects": tuple(DEXYCB_SUBJECTS[i] for i in subject_indices),
        "serials": tuple(DEXYCB_SERIALS[i] for i in serial_indices),
        "sequence_indices": tuple(sequence_indices),
    }


class DexYCBLightFastViTDataset(Dataset):
    """
    Official DexYCB setup reader with FreiHAND-compatible crops and supervision.

    DexYCB stores 3D joints and MANO pose parameters in metres.  Mesh vertices
    are reconstructed from the official per-subject betas and ``pose_m`` so the
    mesh loss and MPVPE metrics use genuine DexYCB MANO supervision.
    """

    SPLIT_ALIASES = {
        "train": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "test",
    }

    def __init__(
        self,
        data_root="dexycb",
        mano_root="dexycb",
        split="train",
        setup="s0",
        img_size=518,
        use_annotation_uv=True,
        crop_padding=1.5,
        min_crop_size=32.0,
        augmentation=None,
        min_joints_in_frame=12,
        require_hand_segmentation=True,
        require_hand_in_crop=True,
        visibility_filter_workers=16,
        index_cache_dir=None,
        visibility_cache_dir=None,
    ):
        super().__init__()

        split_key = self.SPLIT_ALIASES.get(split)
        if split_key is None:
            raise ValueError(f"Unsupported split: {split}")

        self.data_root = Path(data_root)
        self.mano_root = Path(mano_root)
        self.split = split_key
        self.setup = str(setup).lower()
        self.img_size = int(img_size)
        self.use_annotation_uv = use_annotation_uv
        self.crop_padding = float(crop_padding)
        self.min_crop_size = float(min_crop_size)
        self.min_joints_in_frame = int(min_joints_in_frame)
        self.require_hand_segmentation = bool(require_hand_segmentation)
        self.require_hand_in_crop = bool(require_hand_in_crop)
        self.visibility_filter_workers = max(int(visibility_filter_workers), 1)
        cache_value = index_cache_dir or visibility_cache_dir
        self.index_cache_dir = Path(cache_value) if cache_value else None
        # Compatibility for callers that inspect this old attribute.
        self.visibility_cache_dir = self.index_cache_dir
        if not 0 <= self.min_joints_in_frame <= 21:
            raise ValueError("min_joints_in_frame must be between 0 and 21")
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"DexYCB data root not found: {self.data_root}")
        if not self.mano_root.is_dir():
            raise FileNotFoundError(f"DexYCB MANO metadata root not found: {self.mano_root}")
        self.selection = official_dexycb_selection(self.setup, self.split)
        self.intrinsics_map = self._load_intrinsics()
        self.mano_model = MANORightModel()
        self._sequence_metadata = {}
        candidates, index_audit, index_signature = self._build_samples()
        self.samples, filter_audit = self._filter_samples(candidates, index_signature)
        self.audit = {**index_audit, **filter_audit}
        self.augmentation = dict(augmentation or {})
        self.augmentation_enabled = bool(
            self.split == "train" and self.augmentation.get("enabled", False)
        )
        self.num_pose_groups = len(self.samples)

    def _load_intrinsics(self):
        intrinsics_dir = self.data_root / "calibration" / "intrinsics"
        if not intrinsics_dir.is_dir():
            raise FileNotFoundError(f"DexYCB intrinsics not found: {intrinsics_dir}")
        intrinsics = {}
        for serial in self.selection["serials"]:
            intrinsics_path = intrinsics_dir / f"{serial}_640x480.yml"
            if not intrinsics_path.is_file():
                raise FileNotFoundError(f"DexYCB intrinsics not found: {intrinsics_path}")
            with intrinsics_path.open("r", encoding="utf-8") as intrinsics_file:
                color = yaml.load(intrinsics_file, Loader=yaml.FullLoader)["color"]
            intrinsics[serial] = np.asarray(
                [
                    [color["fx"], 0.0, color["ppx"]],
                    [0.0, color["fy"], color["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
        if not intrinsics:
            raise RuntimeError(f"No DexYCB intrinsics found under: {intrinsics_dir}")
        return intrinsics

    @staticmethod
    def _atomic_json_dump(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_path, path)

    def _metadata_signature(self):
        paths = []
        for subject in self.selection["subjects"]:
            subject_dir = self.data_root / subject
            sequences = sorted(path for path in subject_dir.iterdir() if path.is_dir())
            if len(sequences) != 100:
                raise RuntimeError(
                    f"Expected 100 DexYCB sequences under {subject_dir}, "
                    f"got {len(sequences)}"
                )
            paths.extend(
                sequences[index] / "meta.yml"
                for index in self.selection["sequence_indices"]
            )
        paths.extend(
            self.data_root / "calibration" / "intrinsics" / f"{serial}_640x480.yml"
            for serial in self.selection["serials"]
        )
        digest = hashlib.sha256()
        digest.update(str(self.data_root.resolve()).encode())
        digest.update(self.setup.encode())
        digest.update(self.split.encode())
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            stat = path.stat()
            fingerprint = (
                f"{path.relative_to(self.data_root)}:{stat.st_size}:"
                f"{stat.st_mtime_ns}\n"
            )
            digest.update(fingerprint.encode())
        return digest.hexdigest()

    def _cache_path(self, kind):
        if self.index_cache_dir is None:
            return None
        return self.index_cache_dir / f"{kind}_{self.setup}_{self.split}.json"

    def _sample_from_relative(self, relative):
        path = Path(relative)
        if FORBIDDEN_DATA_PARTS.intersection(path.parts):
            raise RuntimeError(f"Forbidden sampled DexYCB path entered index: {relative}")
        label_relative = Path(str(path).replace("/color_", "/labels_")).with_suffix(".npz")
        return {
            "img_relative_path": path.as_posix(),
            "img_path": self.data_root / path,
            "ann_path": self.data_root / label_relative,
            "subject": path.parts[0],
            "sequence": path.parts[1],
            "camera_serial": path.parts[2],
            "frame_id": int(path.stem.rsplit("_", 1)[1]),
            "mano_side": "right",
        }

    def _build_samples(self):
        signature = self._metadata_signature()
        cache_path = self._cache_path("index")
        if cache_path is not None and cache_path.is_file():
            with cache_path.open("r", encoding="utf-8") as cache_file:
                cache = json.load(cache_file)
            if cache.get("version") == 3 and cache.get("metadata_signature") == signature:
                samples = [self._sample_from_relative(path) for path in cache["valid_paths"]]
                return samples, cache["audit"], signature

        samples = []
        counts = Counter()
        side_counts = Counter()
        for subject in self.selection["subjects"]:
            subject_dir = self.data_root / subject
            sequences = sorted(path for path in subject_dir.iterdir() if path.is_dir())
            for sequence_index in self.selection["sequence_indices"]:
                sequence_dir = sequences[sequence_index]
                meta_path = sequence_dir / "meta.yml"
                try:
                    with meta_path.open("r", encoding="utf-8") as meta_file:
                        metadata = yaml.load(meta_file, Loader=yaml.FullLoader)
                    num_frames = int(metadata["num_frames"])
                    sides = metadata.get("mano_sides", [])
                    side = sides[0] if len(sides) == 1 else "invalid"
                    calibration = metadata["mano_calib"][0]
                    mano_path = self.mano_root / "calibration" / f"mano_{calibration}" / "mano.yml"
                    if not mano_path.is_file():
                        raise FileNotFoundError(mano_path)
                    with mano_path.open("r", encoding="utf-8") as mano_file:
                        betas = np.asarray(yaml.load(mano_file, Loader=yaml.FullLoader)["betas"])
                    if betas.reshape(-1).shape != (10,):
                        raise ValueError(f"Invalid MANO betas in {mano_path}")
                except Exception:
                    counts["invalid_sequence_metadata_or_mano"] += len(self.selection["serials"])
                    continue

                sequence_candidates = num_frames * len(self.selection["serials"])
                counts["official_candidates"] += sequence_candidates
                side_counts[side] += sequence_candidates
                if side != "right":
                    counts["non_right_hand"] += sequence_candidates
                    continue
                for serial in self.selection["serials"]:
                    camera_dir = sequence_dir / serial
                    for frame_id in range(num_frames):
                        sample_dir = Path(subject) / sequence_dir.name / serial
                        relative = sample_dir / f"color_{frame_id:06d}.jpg"
                        label_relative = sample_dir / f"labels_{frame_id:06d}.npz"
                        image_path = self.data_root / relative
                        label_path = self.data_root / label_relative
                        if not image_path.is_file() or image_path.stat().st_size == 0:
                            counts["missing_rgb"] += 1
                            continue
                        if not label_path.is_file() or label_path.stat().st_size == 0:
                            counts["missing_label"] += 1
                            continue
                        samples.append(self._sample_from_relative(relative.as_posix()))

        if not samples:
            raise RuntimeError(f"No right-hand DexYCB samples for {self.setup}/{self.split}")
        audit = {
            "setup": self.setup,
            "split": self.split,
            "official_candidates": counts["official_candidates"],
            "indexed_candidates": len(samples),
            "side_candidates": dict(side_counts),
            "index_rejections": {
                name: count
                for name, count in counts.items()
                if name != "official_candidates"
            },
        }
        if cache_path is not None:
            self._atomic_json_dump(cache_path, {
                "version": 3,
                "metadata_signature": signature,
                "valid_paths": [sample["img_relative_path"] for sample in samples],
                "audit": audit,
            })
        return samples, audit, signature

    def _visibility_from_annotation(self, sample):
        stats = {
            "annotation_valid": False,
            "joints_in_frame": 0,
            "bbox_intersects": False,
            "has_hand_segmentation": False,
            "hand_in_crop": False,
        }
        try:
            with np.load(sample["ann_path"]) as annotation:
                required = {"joint_3d", "joint_2d", "pose_m", "seg"}
                if not required.issubset(annotation.files):
                    return {**stats, "accepted": False}
                joints_3d = np.asarray(annotation["joint_3d"], dtype=np.float32).reshape(21, 3)
                joints_2d = np.asarray(annotation["joint_2d"], dtype=np.float32).reshape(21, 2)
                pose_m = np.asarray(annotation["pose_m"], dtype=np.float32).reshape(-1, 51)
                segmentation = np.asarray(annotation["seg"])
            if pose_m.shape[0] < 1 or segmentation.ndim < 2:
                return {**stats, "accepted": False}
        except Exception:
            return {**stats, "accepted": False}

        height, width = segmentation.shape[:2]
        valid = (
            np.isfinite(joints_2d).all(axis=1)
            & np.isfinite(joints_3d).all(axis=1)
            & ~(joints_2d == -1.0).all(axis=1)
            & ~(joints_3d == -1.0).all(axis=1)
            & (joints_3d[:, 2] > 1e-4)
        )
        in_frame = (
            valid
            & (joints_2d[:, 0] >= 0.0)
            & (joints_2d[:, 0] < width)
            & (joints_2d[:, 1] >= 0.0)
            & (joints_2d[:, 1] < height)
        )
        has_hand_segmentation = bool(np.any(segmentation == 255))
        bbox_intersects = False
        hand_in_crop = False
        try:
            projected = self._project_3d_to_pixel(
                joints_3d, self.intrinsics_map[sample["camera_serial"]]
            )
            bbox = self._bbox_from_projected_joints(
                joints_3d, projected, segmentation.shape, require_image_intersection=True
            )
            _, crop_side, crop_x1, crop_y1 = self._crop_params_from_bbox(
                bbox, segmentation.shape
            )
            bbox_intersects = True
            x1 = max(int(math.floor(crop_x1)), 0)
            y1 = max(int(math.floor(crop_y1)), 0)
            x2 = min(int(math.ceil(crop_x1 + crop_side)), width)
            y2 = min(int(math.ceil(crop_y1 + crop_side)), height)
            hand_in_crop = bool(x2 > x1 and y2 > y1 and np.any(segmentation[y1:y2, x1:x2] == 255))
        except (KeyError, ValueError):
            pass

        stats.update({
            "annotation_valid": True,
            "joints_in_frame": int(in_frame.sum()),
            "bbox_intersects": bbox_intersects,
            "has_hand_segmentation": has_hand_segmentation,
            "hand_in_crop": hand_in_crop,
        })
        stats["accepted"] = bool(
            stats["joints_in_frame"] >= self.min_joints_in_frame
            and bbox_intersects
            and (has_hand_segmentation or not self.require_hand_segmentation)
            and (hand_in_crop or not self.require_hand_in_crop)
        )
        return stats

    def _filter_signature(self, index_signature):
        payload = {
            "version": 3,
            "index_signature": index_signature,
            "setup": self.setup,
            "split": self.split,
            "crop_padding": self.crop_padding,
            "min_crop_size": self.min_crop_size,
            "min_joints_in_frame": self.min_joints_in_frame,
            "require_hand_segmentation": self.require_hand_segmentation,
            "require_hand_in_crop": self.require_hand_in_crop,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _filter_samples(self, samples, index_signature):
        cache_path = self._cache_path("filter")
        filter_signature = self._filter_signature(index_signature)
        if cache_path is not None and cache_path.is_file():
            with cache_path.open("r", encoding="utf-8") as cache_file:
                cache = json.load(cache_file)
            cache_matches = (
                cache.get("version") == 3
                and cache.get("filter_signature") == filter_signature
            )
            if cache_matches:
                valid_paths = set(cache["valid_paths"])
                filtered = [
                    sample
                    for sample in samples
                    if sample["img_relative_path"] in valid_paths
                ]
                print(
                    f"DexYCB visibility filter ({self.split}, cached): "
                    f"{len(filtered)}/{len(samples)} samples retained"
                )
                return filtered, cache["audit"]

        def inspect(sample):
            stats = self._visibility_from_annotation(sample)
            return sample, stats

        inspected = []
        with ThreadPoolExecutor(max_workers=self.visibility_filter_workers) as executor:
            for progress, result in enumerate(executor.map(inspect, samples), start=1):
                inspected.append(result)
                if progress % 10000 == 0 or progress == len(samples):
                    print(
                        f"DexYCB filter progress ({self.setup}/{self.split}): "
                        f"{progress}/{len(samples)}"
                    )
        filtered = [sample for sample, stats in inspected if stats["accepted"]]
        if not filtered:
            raise RuntimeError(
                f"DexYCB visibility filtering removed every {self.split} sample"
            )

        # These diagnostic counts intentionally overlap: one rejected frame can
        # violate several quality criteria at once.
        rejection_counts = {
            "invalid_annotation": sum(
                not stats["annotation_valid"] for _, stats in inspected
            ),
            "too_few_joints": sum(
                stats["joints_in_frame"] < self.min_joints_in_frame
                for _, stats in inspected
            ),
            "bbox_no_intersection": sum(
                not stats["bbox_intersects"] for _, stats in inspected
            ),
            "empty_hand_segmentation": sum(
                not stats["has_hand_segmentation"] for _, stats in inspected
            ),
            "hand_missing_from_crop": sum(
                not stats["hand_in_crop"] for _, stats in inspected
            ),
        }
        audit = {
            "filtered_candidates": len(filtered),
            "filter_rejections": rejection_counts,
            "subjects": dict(Counter(sample["subject"] for sample in filtered)),
            "sequences": len({(sample["subject"], sample["sequence"]) for sample in filtered}),
            "cameras": dict(Counter(sample["camera_serial"] for sample in filtered)),
            "forbidden_paths": sum(
                bool(FORBIDDEN_DATA_PARTS.intersection(Path(sample["img_relative_path"]).parts))
                for sample in filtered
            ),
        }
        if cache_path is not None:
            cache = {
                "version": 3,
                "filter_signature": filter_signature,
                "audit": audit,
                "valid_paths": [
                    sample["img_relative_path"] for sample in filtered
                ],
            }
            self._atomic_json_dump(cache_path, cache)
        print(
            f"DexYCB visibility filter ({self.split}): "
            f"{len(filtered)}/{len(samples)} samples retained; "
            f"rejections={rejection_counts}"
        )
        return filtered, audit

    def _mano_betas(self, sample):
        sequence_key = (sample["subject"], sample["sequence"])
        if sequence_key not in self._sequence_metadata:
            meta_path = self.mano_root / sequence_key[0] / sequence_key[1] / "meta.yml"
            if not meta_path.is_file():
                raise FileNotFoundError(f"DexYCB sequence metadata not found: {meta_path}")
            with meta_path.open("r", encoding="utf-8") as meta_file:
                metadata = yaml.load(meta_file, Loader=yaml.FullLoader)
            sides = metadata.get("mano_sides", [])
            if sides != ["right"]:
                raise ValueError(
                    f"Only DexYCB right-hand samples are supported, got {sides} in {meta_path}"
                )
            calibration = metadata["mano_calib"][0]
            mano_path = self.mano_root / "calibration" / f"mano_{calibration}" / "mano.yml"
            with mano_path.open("r", encoding="utf-8") as mano_file:
                betas = np.asarray(
                    yaml.load(mano_file, Loader=yaml.FullLoader)["betas"],
                    dtype=np.float32,
                )
            self._sequence_metadata[sequence_key] = betas.reshape(10)
        return self._sequence_metadata[sequence_key]

    def _load_annotation(self, sample):
        with np.load(sample["ann_path"]) as annotation:
            pose_gt_3d = np.asarray(annotation["joint_3d"], dtype=np.float32).reshape(21, 3)
            gt_pose_2d = np.asarray(annotation["joint_2d"], dtype=np.float32).reshape(21, 2)
            pose_m = np.asarray(annotation["pose_m"], dtype=np.float32).reshape(-1, 51)[0]
        vertices_gt_3d = self.mano_model(pose_m, self._mano_betas(sample))
        cam_k = self.intrinsics_map[sample["camera_serial"]].copy()
        return pose_gt_3d, vertices_gt_3d, cam_k, gt_pose_2d

    def _project_3d_to_pixel(self, joints_3d, cam_k):
        homogeneous = np.asarray(joints_3d, dtype=np.float32) @ cam_k.T
        depth = homogeneous[:, 2:3]
        safe_depth = np.where(np.abs(depth) > 1e-8, depth, np.nan)
        return (homogeneous[:, :2] / safe_depth).astype(np.float32)

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

    def _bbox_from_projected_joints(
        self,
        joints_3d,
        keypoints_2d,
        image_shape,
        require_image_intersection=True,
    ):
        h, w = image_shape[:2]
        valid = self._valid_projected_keypoints(joints_3d, keypoints_2d)
        if not np.any(valid):
            raise ValueError("Cannot compute a hand bbox without valid projected joints")

        pts = keypoints_2d[valid]
        x1 = float(np.min(pts[:, 0]))
        y1 = float(np.min(pts[:, 1]))
        x2 = float(np.max(pts[:, 0]))
        y2 = float(np.max(pts[:, 1]))

        if not np.all(np.isfinite([x1, y1, x2, y2])):
            raise ValueError("Projected hand bbox contains non-finite coordinates")
        intersects = x2 >= 0.0 and y2 >= 0.0 and x1 < w and y1 < h
        if require_image_intersection and not intersects:
            raise ValueError(
                "Projected hand bbox does not intersect the source image: "
                f"bbox={(x1, y1, x2, y2)}, image_size={(w, h)}"
            )

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
    ):
        rotation_probability = float(self.augmentation.get("rotation_probability", 0.6))
        max_rotation_degrees = float(self.augmentation.get("max_rotation_degrees", 30.0))
        angle = (
            random.uniform(-max_rotation_degrees, max_rotation_degrees)
            if random.random() < rotation_probability
            else 0.0
        )

        # OpenCV's positive image angle is counter-clockwise in image space.
        # DexYCB camera coordinates use x-right/y-down, hence the negative
        # angle for the equivalent camera-Z rotation.
        theta = math.radians(-angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        rotation_3d = np.array(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pose_gt_3d = (pose_gt_3d @ rotation_3d.T).astype(np.float32)
        vertices_gt_3d = (vertices_gt_3d @ rotation_3d.T).astype(np.float32)

        # Rotate first in the virtual full-image camera, then recenter the crop
        # around the rotated hand.  The composed warp samples directly from the
        # original RGB, so even a rotated bbox outside the 640x480 canvas is not
        # collapsed to a degenerate border crop.
        rotated_projected_2d = self._project_3d_to_pixel(
            pose_gt_3d, cam_k_original
        )
        rotated_bbox = self._bbox_from_projected_joints(
            pose_gt_3d,
            rotated_projected_2d,
            img.shape,
            require_image_intersection=False,
        )
        center, crop_side, _, _ = self._crop_params_from_bbox(
            rotated_bbox, img.shape
        )
        center, crop_side = self._sample_augmented_crop(center, crop_side)
        crop_affine, crop_x1, crop_y1, resize_scale = self._crop_affine(
            center, crop_side
        )
        crop_h = np.eye(3, dtype=np.float32)
        crop_h[:2] = crop_affine
        cam_k_new = crop_h @ cam_k_original

        rotation_h = cam_k_original @ rotation_3d @ np.linalg.inv(cam_k_original)
        image_h = crop_h @ rotation_h
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
            rotated_bbox,
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
        return self.samples[idx]

    def __getitem__(self, idx):
        sample = self._sample_for_index(idx)
        img_path = sample["img_path"]

        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (
            pose_gt_3d,
            vertices_gt_3d,
            cam_k_original,
            gt_pose_2d,
        ) = self._load_annotation(sample)

        projected_2d = self._project_3d_to_pixel(pose_gt_3d, cam_k_original)
        bbox = self._bbox_from_projected_joints(pose_gt_3d, projected_2d, img.shape)
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
                bbox,
            ) = self._apply_training_geometry(
                img,
                pose_gt_3d,
                vertices_gt_3d,
                cam_k_original,
                gt_pose_2d,
            )
            img = self._apply_training_appearance(img, gt_pose_2d)
        else:
            center, crop_side, crop_x1, crop_y1 = self._crop_params_from_bbox(
                bbox, img.shape
            )
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
        joint_valid = (
            np.isfinite(pose_gt_3d).all(axis=1)
            & np.isfinite(gt_pose_2d).all(axis=1)
            & (pose_gt_3d[:, 2] > 1e-4)
            & ~(pose_gt_3d == -1.0).all(axis=1)
            & ~(gt_pose_2d == -1.0).all(axis=1)
        )
        vertex_valid = np.isfinite(vertices_gt_3d).all(axis=1) & joint_valid[0]

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
            "joint_valid": torch.from_numpy(joint_valid),
            "vertex_valid": torch.from_numpy(vertex_valid),
            "is_right": torch.tensor(True),
            "crop_center": torch.from_numpy(center).float(),
            "crop_scale": torch.tensor(crop_side, dtype=torch.float32),
            "crop_bbox": torch.from_numpy(bbox).float(),
        }


# DexYCB/MANO order: wrist, then thumb/index/middle/ring/little base to tip.
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def _draw_hand_keypoints(image, keypoints):
    """Draw DexYCB 2D ground-truth joints on an RGB image."""
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
    img_path = sample["img_path"]
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    (
        pose_gt_3d,
        vertices_gt_3d,
        cam_k_original,
        gt_pose_2d,
    ) = dataset._load_annotation(sample)
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
        _,
    ) = dataset._apply_training_geometry(
        image.copy(),
        pose_gt_3d,
        vertices_gt_3d,
        cam_k_original,
        gt_pose_2d,
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
        description="Export augmented DexYCB training and validation samples for visual inspection."
    )
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "dexycb_graphormer.yaml"),
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
    mano_root = Path(data_cfg["mano_root"])
    if not mano_root.is_absolute():
        mano_root = (ROOT_DIR / mano_root).resolve()
    common_kwargs = {
        "data_root": str(data_root),
        "mano_root": str(mano_root),
        "setup": data_cfg.get("setup", "s0"),
        "img_size": data_cfg["image_size"],
        "use_annotation_uv": data_cfg.get("use_annotation_uv", True),
        "crop_padding": data_cfg.get("crop_padding", 1.5),
        "min_crop_size": data_cfg.get("min_crop_size", 32.0),
        "min_joints_in_frame": data_cfg.get("min_joints_in_frame", 12),
        "require_hand_segmentation": data_cfg.get(
            "require_hand_segmentation", True
        ),
        "require_hand_in_crop": data_cfg.get("require_hand_in_crop", True),
        "visibility_filter_workers": data_cfg.get(
            "visibility_filter_workers", 16
        ),
    }
    index_cache_dir = data_cfg.get("index_cache_dir") or data_cfg.get(
        "visibility_cache_dir"
    )
    if index_cache_dir:
        index_cache_dir = Path(index_cache_dir)
        if not index_cache_dir.is_absolute():
            index_cache_dir = (ROOT_DIR / index_cache_dir).resolve()
        common_kwargs["index_cache_dir"] = str(index_cache_dir)

    # Seed all random sources used by the augmentation pipeline for reproducible exports.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = DexYCBLightFastViTDataset(
        split="train",
        augmentation=data_cfg.get("augmentation", {}),
        **common_kwargs,
    )
    val_dataset = DexYCBLightFastViTDataset(
        split=data_cfg.get("val_split", "val"), **common_kwargs
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
