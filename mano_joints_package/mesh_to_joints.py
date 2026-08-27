"""Convert 778 MANO mesh vertices to SimpleHand's 21 hand joints."""

from functools import lru_cache
from pathlib import Path
import pickle
from typing import Union

import numpy as np
import torch
from torch import nn


DEFAULT_MANO_PATH = Path(__file__).resolve().parent / "data" / "MANO_RIGHT_C.pkl"

# Thumb, index, middle, ring and little fingertip vertices.
FINGERTIP_VERTEX_INDICES = (744, 320, 443, 555, 672)

# Convert MANO's 16 joints followed by the 5 fingertips to SimpleHand order.
JOINT_REORDER_INDICES = (
    0,
    13, 14, 15, 16,
    1, 2, 3, 17,
    4, 5, 6, 18,
    10, 11, 12, 19,
    7, 8, 9, 20,
)


def _validate_vertices(vertices: torch.Tensor) -> None:
    if not isinstance(vertices, torch.Tensor):
        raise TypeError(
            "vertices must be a torch.Tensor with shape [B, 778, 3], "
            f"got {type(vertices).__name__}"
        )
    if vertices.ndim != 3 or tuple(vertices.shape[1:]) != (778, 3):
        raise ValueError(
            "vertices must have shape [B, 778, 3], "
            f"got {tuple(vertices.shape)}"
        )


@lru_cache(maxsize=None)
def _load_joint_regressor_cpu(mano_path: str) -> torch.Tensor:
    path = Path(mano_path)
    if not path.is_file():
        raise FileNotFoundError(f"MANO parameter file not found: {path}")

    with path.open("rb") as mano_file:
        mano_data = pickle.load(mano_file)

    if "J_regressor" not in mano_data:
        raise KeyError(f"MANO parameter file has no 'J_regressor': {path}")

    regressor = torch.tensor(
        np.array(mano_data["J_regressor"]), dtype=torch.float32
    )
    if tuple(regressor.shape) != (16, 778):
        raise ValueError(
            "MANO J_regressor must have shape [16, 778], "
            f"got {tuple(regressor.shape)}"
        )
    return regressor


def load_joint_regressor(
    mano_path: Union[str, Path] = DEFAULT_MANO_PATH,
) -> torch.Tensor:
    """Load a detached CPU copy of MANO's [16, 778] joint regressor."""
    return _load_joint_regressor_cpu(str(Path(mano_path).resolve())).clone()


def vertices2joints(
    joint_regressor: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """Regress 16 MANO joints from vertices: [B,778,3] -> [B,16,3]."""
    _validate_vertices(vertices)
    if not isinstance(joint_regressor, torch.Tensor):
        raise TypeError("joint_regressor must be a torch.Tensor")
    if tuple(joint_regressor.shape) != (16, 778):
        raise ValueError(
            "joint_regressor must have shape [16, 778], "
            f"got {tuple(joint_regressor.shape)}"
        )
    return torch.einsum("bik,ji->bjk", vertices, joint_regressor)


def get_fingertips(vertices: torch.Tensor) -> torch.Tensor:
    """Read the five fingertip positions directly from mesh vertices."""
    _validate_vertices(vertices)
    return vertices[:, FINGERTIP_VERTEX_INDICES]


def remap_joints_and_fingertips(
    joints: torch.Tensor, fingertips: torch.Tensor
) -> torch.Tensor:
    """Concatenate 16 MANO joints and 5 fingertips, then use SimpleHand order."""
    if joints.ndim != 3 or tuple(joints.shape[1:]) != (16, 3):
        raise ValueError(
            "joints must have shape [B, 16, 3], "
            f"got {tuple(joints.shape)}"
        )
    if fingertips.ndim != 3 or tuple(fingertips.shape[1:]) != (5, 3):
        raise ValueError(
            "fingertips must have shape [B, 5, 3], "
            f"got {tuple(fingertips.shape)}"
        )
    if joints.shape[0] != fingertips.shape[0]:
        raise ValueError("joints and fingertips must have the same batch size")

    joints_with_tips = torch.cat((joints, fingertips), dim=1)
    return joints_with_tips[:, JOINT_REORDER_INDICES]


def mesh_to_joints(
    vertices: torch.Tensor,
    mano_path: Union[str, Path] = DEFAULT_MANO_PATH,
) -> torch.Tensor:
    """Convert [B, 778, 3] vertices to [B, 21, 3] SimpleHand joints."""
    _validate_vertices(vertices)
    regressor = load_joint_regressor(mano_path).to(
        device=vertices.device, dtype=vertices.dtype
    )
    joints = vertices2joints(regressor, vertices)
    fingertips = get_fingertips(vertices)
    return remap_joints_and_fingertips(joints, fingertips)


class MeshToJoints(nn.Module):
    """PyTorch module for the same 778-vertex to 21-joint conversion."""

    def __init__(
        self,
        mano_path: Union[str, Path] = DEFAULT_MANO_PATH,
    ) -> None:
        super().__init__()
        self.register_buffer("J_regressor", load_joint_regressor(mano_path))

    def forward(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertices: Predicted MANO mesh vertices with shape [B, 778, 3].
        Returns:
            3D hand joints in SimpleHand order with shape [B, 21, 3].
        """
        _validate_vertices(vertices)
        regressor = self.J_regressor.to(dtype=vertices.dtype)
        joints = vertices2joints(regressor, vertices)
        fingertips = get_fingertips(vertices)
        return remap_joints_and_fingertips(joints, fingertips)
