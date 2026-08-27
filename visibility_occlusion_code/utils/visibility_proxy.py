"""Automatic joint visibility labels from calibrated 3D hand annotations.

The main path uses CPU analytic geometry rather than OpenGL/Open3D rendering:
for each target joint, cast a ray from the wrist-camera origin to the joint and
test whether another hand proxy primitive is hit first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


VISIBLE = 0
SELF_OCCLUDED = 1
OUT_OF_VIEW = 2
UNCERTAIN = 3

LABEL_NAMES: Dict[int, str] = {
    VISIBLE: "visible",
    SELF_OCCLUDED: "self_occluded",
    OUT_OF_VIEW: "out_of_view",
    UNCERTAIN: "uncertain",
}

HAND_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

PALM_BONES: Tuple[Tuple[int, int], ...] = (
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 5),
    (0, 17),
)

DEFAULT_ALL_CAPSULES: Tuple[Tuple[int, int], ...] = HAND_BONES + PALM_BONES


@dataclass(frozen=True)
class VisibilityConfig:
    image_size: Tuple[int, int]
    finger_radius: float = 5.0
    joint_radius: float = 5.0
    palm_radius: float = 10.0
    depth_margin: float = 10.0
    min_depth: float = 1e-2


@dataclass
class VisibilityResult:
    labels: np.ndarray
    projected_2d: np.ndarray
    in_view_mask: np.ndarray
    occluder_index: np.ndarray
    occluder_type: np.ndarray


def normalize_dist_coeffs(dist_coeffs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if dist_coeffs is None:
        return None
    coeffs = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1)
    if coeffs.size == 0 or not np.any(coeffs):
        return None
    if coeffs.size < 5:
        raise ValueError(f"Expected at least 5 distortion coefficients, got {coeffs.shape}")
    return coeffs[:5]


def project_points(
    points_3d: np.ndarray,
    cam_k: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project camera-coordinate 3D points to distorted 2D pixels."""
    points = np.asarray(points_3d, dtype=np.float32)
    camera_matrix = np.asarray(cam_k, dtype=np.float32)
    coeffs = normalize_dist_coeffs(dist_coeffs)

    if coeffs is None:
        z = np.maximum(points[:, 2:3], 1e-8)
        points_homo = points @ camera_matrix.T
        return points_homo[:, :2] / z

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, coeffs)
    return projected.reshape(-1, 2)


def in_view_mask(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    image_size: Tuple[int, int],
    min_depth: float = 1e-2,
) -> np.ndarray:
    height, width = image_size
    points = np.asarray(points_3d)
    projected = np.asarray(points_2d)
    return (
        (points[:, 2] > min_depth)
        & np.isfinite(projected[:, 0])
        & np.isfinite(projected[:, 1])
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )


def _adjacency(num_joints: int = 21) -> Dict[int, set]:
    adjacency = {idx: set() for idx in range(num_joints)}
    for start, end in HAND_BONES:
        adjacency[start].add(end)
        adjacency[end].add(start)
    return adjacency


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


def _ray_sphere_front_t(
    direction: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Optional[float]:
    """Return first positive ray/sphere intersection depth, if any."""
    projection = float(np.dot(direction, center))
    center_norm2 = float(np.dot(center, center))
    radius2 = radius * radius
    discriminant = projection * projection - (center_norm2 - radius2)
    if discriminant < 0:
        return None
    sqrt_disc = float(np.sqrt(discriminant))
    t0 = projection - sqrt_disc
    t1 = projection + sqrt_disc
    if t0 > 0:
        return t0
    if t1 > 0:
        return t1
    return None


def _ray_segment_closest(
    direction: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> Tuple[float, float, float]:
    """Closest distance between ray ``t * direction`` and segment ``start + u*(end-start)``.

    Returns ``(distance, ray_t, segment_u)``.
    """
    segment = end - start
    seg_len2 = float(np.dot(segment, segment))
    if seg_len2 < 1e-12:
        t = max(0.0, float(np.dot(start, direction)))
        distance = float(np.linalg.norm(t * direction - start))
        return distance, t, 0.0

    a = 1.0
    b = float(np.dot(direction, segment))
    c = seg_len2
    d = -float(np.dot(direction, start))
    e = -float(np.dot(segment, start))
    denom = a * c - b * b

    if abs(denom) < 1e-12:
        u = np.clip(e / c, 0.0, 1.0)
    else:
        u = np.clip((a * e - b * d) / denom, 0.0, 1.0)

    closest_on_segment = start + u * segment
    t = max(0.0, float(np.dot(closest_on_segment, direction)))
    closest_on_ray = t * direction

    # Recompute segment parameter after clamping ray depth to handle endpoint cases.
    u = np.clip(float(np.dot(closest_on_ray - start, segment) / c), 0.0, 1.0)
    closest_on_segment = start + u * segment
    t = max(0.0, float(np.dot(closest_on_segment, direction)))
    closest_on_ray = t * direction
    distance = float(np.linalg.norm(closest_on_ray - closest_on_segment))
    return distance, t, float(u)


def _ray_capsule_front_t(
    direction: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> Optional[float]:
    """Approximate first positive ray/capsule intersection depth."""
    distance, t, _ = _ray_segment_closest(direction, start, end)
    if distance > radius:
        return None
    offset = float(np.sqrt(max(radius * radius - distance * distance, 0.0)))
    return max(0.0, t - offset)


def classify_joint_visibility(
    joints_3d: np.ndarray,
    cam_k: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    config: VisibilityConfig,
    capsules: Iterable[Tuple[int, int]] = DEFAULT_ALL_CAPSULES,
) -> VisibilityResult:
    """Classify each joint as visible, self-occluded, or out-of-view."""
    joints = np.asarray(joints_3d, dtype=np.float64)
    if joints.ndim != 2 or joints.shape[1] != 3:
        raise ValueError(f"Expected joints shape [J, 3], got {joints.shape}")

    projected = project_points(joints.astype(np.float32), cam_k, dist_coeffs)
    in_view = in_view_mask(joints, projected, config.image_size, min_depth=config.min_depth)
    labels = np.full(joints.shape[0], OUT_OF_VIEW, dtype=np.int64)
    occluder_index = np.full(joints.shape[0], -1, dtype=np.int64)
    occluder_type = np.full(joints.shape[0], "", dtype=object)

    adjacency = _adjacency(joints.shape[0])
    capsule_list = tuple(capsules)
    palm_set = set(PALM_BONES)

    for joint_idx, joint in enumerate(joints):
        if not in_view[joint_idx]:
            continue

        target_depth = float(np.linalg.norm(joint))
        if target_depth <= config.min_depth:
            continue
        direction = _normalize(joint)
        labels[joint_idx] = VISIBLE

        for capsule_idx, (start_idx, end_idx) in enumerate(capsule_list):
            if joint_idx in (start_idx, end_idx):
                continue
            radius = config.palm_radius if (start_idx, end_idx) in palm_set else config.finger_radius
            front_t = _ray_capsule_front_t(
                direction,
                joints[start_idx],
                joints[end_idx],
                radius,
            )
            if front_t is not None and front_t < target_depth - config.depth_margin:
                labels[joint_idx] = SELF_OCCLUDED
                occluder_index[joint_idx] = capsule_idx
                occluder_type[joint_idx] = "capsule"
                break

        if labels[joint_idx] == SELF_OCCLUDED:
            continue

        for other_idx, center in enumerate(joints):
            if other_idx == joint_idx or other_idx in adjacency[joint_idx]:
                continue
            front_t = _ray_sphere_front_t(direction, center, config.joint_radius)
            if front_t is not None and front_t < target_depth - config.depth_margin:
                labels[joint_idx] = SELF_OCCLUDED
                occluder_index[joint_idx] = other_idx
                occluder_type[joint_idx] = "sphere"
                break

    return VisibilityResult(
        labels=labels,
        projected_2d=projected,
        in_view_mask=in_view,
        occluder_index=occluder_index,
        occluder_type=occluder_type,
    )


def classify_batch_visibility(
    poses_3d: np.ndarray,
    cam_ks: np.ndarray,
    dist_coeffs: Optional[np.ndarray],
    config: VisibilityConfig,
) -> Dict[str, np.ndarray]:
    poses = np.asarray(poses_3d)
    cameras = np.asarray(cam_ks)
    coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs)
    if poses.ndim != 3:
        raise ValueError(f"Expected poses shape [N, J, 3], got {poses.shape}")

    labels = []
    projected = []
    in_view = []
    occluder_index = []
    occluder_type = []

    for idx in range(poses.shape[0]):
        sample_coeffs = None if coeffs is None else coeffs[idx]
        result = classify_joint_visibility(poses[idx], cameras[idx], sample_coeffs, config)
        labels.append(result.labels)
        projected.append(result.projected_2d)
        in_view.append(result.in_view_mask)
        occluder_index.append(result.occluder_index)
        occluder_type.append(result.occluder_type)

    return {
        "visibility_label": np.stack(labels, axis=0),
        "projected_2d": np.stack(projected, axis=0),
        "in_view_mask": np.stack(in_view, axis=0),
        "occluder_index": np.stack(occluder_index, axis=0),
        "occluder_type": np.stack(occluder_type, axis=0),
    }


def visible_joint_ratio(labels: np.ndarray) -> np.ndarray:
    labels_array = np.asarray(labels)
    in_view = (labels_array == VISIBLE) | (labels_array == SELF_OCCLUDED)
    visible = labels_array == VISIBLE
    counts = in_view.sum(axis=-1)
    return np.divide(
        visible.sum(axis=-1),
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )
