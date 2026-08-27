"""Minimal NumPy MANO forward pass for DexYCB mesh supervision."""

from pathlib import Path
import pickle

import cv2
import numpy as np


DEFAULT_MANO_PATH = (
    Path(__file__).resolve().parent.parent
    / "mano_joints_package"
    / "data"
    / "MANO_RIGHT_C.pkl"
)


class MANORightModel:
    """Evaluate the full 48-D right-hand MANO pose used by DexYCB.

    The packaged MANO model and DexYCB annotations both use metres.  ``pose_m``
    stores a global axis-angle, 45 PCA coefficients and camera translation.
    """

    def __init__(self, model_path=DEFAULT_MANO_PATH):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MANO model not found: {model_path}")
        with model_path.open("rb") as model_file:
            data = pickle.load(model_file, encoding="latin1")

        self.v_template = np.asarray(data["v_template"], dtype=np.float32)
        self.faces = np.asarray(data["f"], dtype=np.int32).reshape(-1, 3)
        self.shapedirs = np.asarray(data["shapedirs"], dtype=np.float32)
        self.posedirs = np.asarray(data["posedirs"], dtype=np.float32)
        self.weights = np.asarray(data["weights"], dtype=np.float32)
        self.hand_mean = np.asarray(data["hands_mean"], dtype=np.float32).reshape(45)
        self.hand_components = np.asarray(
            data["hands_components"], dtype=np.float32
        ).reshape(45, 45)
        self.joint_regressor = np.asarray(
            data["J_regressor"].toarray()
            if hasattr(data["J_regressor"], "toarray")
            else data["J_regressor"],
            dtype=np.float32,
        )
        kintree = np.asarray(data["kintree_table"])
        joint_ids = [int(value) for value in kintree[1]]
        id_to_index = {joint_id: index for index, joint_id in enumerate(joint_ids)}
        self.parents = np.asarray(
            [-1]
            + [id_to_index[int(parent_id)] for parent_id in kintree[0, 1:]],
            dtype=np.int64,
        )

        if self.v_template.shape != (778, 3):
            raise ValueError(f"Unexpected MANO template shape: {self.v_template.shape}")
        if self.faces.shape != (1538, 3):
            raise ValueError(f"Unexpected MANO face shape: {self.faces.shape}")
        if self.posedirs.shape != (778, 3, 135):
            raise ValueError(f"Unexpected MANO posedirs shape: {self.posedirs.shape}")

    @staticmethod
    def _rotation_matrices(axis_angles):
        rotations = []
        for axis_angle in np.asarray(axis_angles, dtype=np.float64):
            rotation, _ = cv2.Rodrigues(axis_angle)
            rotations.append(rotation.astype(np.float32))
        return np.stack(rotations, axis=0)

    def __call__(self, pose_m, betas):
        pose_m = np.asarray(pose_m, dtype=np.float32).reshape(-1)
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)
        if pose_m.shape != (51,):
            raise ValueError(f"DexYCB pose_m must have shape [51], got {pose_m.shape}")
        if betas.shape != (10,):
            raise ValueError(f"DexYCB MANO betas must have shape [10], got {betas.shape}")

        full_pose = pose_m[:48].copy()
        # DexYCB stores all 45 MANO PCA coefficients.  Reconstruct the joint
        # axis-angles using the packaged component basis and non-flat mean.
        full_pose[3:] = pose_m[3:48] @ self.hand_components + self.hand_mean
        rotations = self._rotation_matrices(full_pose.reshape(16, 3))
        v_shaped = self.v_template + np.tensordot(
            self.shapedirs, betas, axes=([2], [0])
        )
        joints = self.joint_regressor @ v_shaped
        pose_feature = (rotations[1:] - np.eye(3, dtype=np.float32)).reshape(-1)
        v_posed = v_shaped + np.tensordot(
            self.posedirs, pose_feature, axes=([2], [0])
        )

        transforms = np.zeros((16, 4, 4), dtype=np.float32)
        transforms[:, 3, 3] = 1.0
        transforms[0, :3, :3] = rotations[0]
        transforms[0, :3, 3] = joints[0]
        for joint_index in range(1, 16):
            parent = int(self.parents[joint_index])
            relative = np.eye(4, dtype=np.float32)
            relative[:3, :3] = rotations[joint_index]
            relative[:3, 3] = joints[joint_index] - joints[parent]
            transforms[joint_index] = transforms[parent] @ relative

        transforms[:, :3, 3] -= np.einsum(
            "bij,bj->bi", transforms[:, :3, :3], joints
        )
        vertex_transforms = np.tensordot(self.weights, transforms, axes=([1], [0]))
        homogeneous = np.concatenate(
            [v_posed, np.ones((v_posed.shape[0], 1), dtype=np.float32)], axis=1
        )
        vertices = np.einsum("vij,vj->vi", vertex_transforms, homogeneous)[:, :3]
        return (vertices + pose_m[48:51]).astype(np.float32)
