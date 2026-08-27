"""Standalone simpleHand mesh regressor with no project-internal imports."""

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .layers import (
    AttentionBlock,
    LinearUpsample,
    MultiScaleCrossStageAttention,
)


class MeshRegressor(nn.Module):
    """Regress 778 mesh vertices from 21 pre-sampled joint tokens."""

    def __init__(
        self,
        depths: Sequence[int] = (1, 1, 1),
        token_nums: Sequence[int] = (21, 84, 336),
        dims: Sequence[int] = (256, 128, 64),
        block_types: Sequence[str] = ("attention", "attention", "attention"),
        first_prenorms: Sequence[bool] = (True, True, True),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        configs = (depths, token_nums, dims, block_types, first_prenorms)
        if any(len(config) != 3 for config in configs):
            raise ValueError("all stage configurations must contain exactly 3 items")
        if tuple(token_nums) != (21, 84, 336):
            raise ValueError("token_nums must be [21, 84, 336]")
        if tuple(dims) != (256, 128, 64):
            raise ValueError("dims must be [256, 128, 64]")
        if any(block_type != "attention" for block_type in block_types):
            raise ValueError("the portable package supports block type 'attention' only")

        self.proj_1 = nn.Linear(256, 256)
        self.encoder_1 = self.build_encoder(
            256, depths[0], dropout, first_prenorms[0]
        )
        self.upsample_1 = LinearUpsample(21, 84)

        self.proj_2 = nn.Linear(256, 128)
        self.encoder_2 = self.build_encoder(
            128, depths[1], dropout, first_prenorms[1]
        )
        self.upsample_2 = LinearUpsample(84, 336)

        self.proj_3 = nn.Linear(128, 64)
        self.encoder_3 = self.build_encoder(
            64, depths[2], dropout, first_prenorms[2]
        )
        self.upsample_3 = LinearUpsample(336, 778)
        self.cross_stage_attn = MultiScaleCrossStageAttention()
        self.pred_final = nn.Linear(64, 3)

        self.pos_emb_1 = nn.Parameter(torch.zeros(1, 21, 256))
        self.pos_emb_2 = nn.Parameter(torch.zeros(1, 84, 256))
        self.pos_emb_3 = nn.Parameter(torch.zeros(1, 336, 128))
        nn.init.trunc_normal_(self.pos_emb_1, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_2, std=0.02)
        nn.init.trunc_normal_(self.pos_emb_3, std=0.02)

    @staticmethod
    def build_encoder(
        dim: int, depth: int, dropout: float, first_prenorm: bool
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                AttentionBlock(
                    dim,
                    dim,
                    drop_path=dropout,
                    dropout=dropout,
                    pre_norm=first_prenorm if index == 0 else True,
                )
                for index in range(depth)
            ]
        )

    def load_from_mesh_head(self, mesh_head: nn.Module) -> None:
        """Copy compatible regressor weights from a simpleHand MeshHead."""
        self._load_compatible_state(mesh_head.state_dict())

    def load_from_checkpoint(
        self,
        checkpoint: Mapping,
        prefix: str = "mesh_head.",
    ) -> None:
        """Load regressor weights from a full simpleHand checkpoint mapping."""
        state = checkpoint.get("state_dict", checkpoint)
        extracted = {
            key[len(prefix):]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not extracted:
            raise ValueError(f"checkpoint contains no keys with prefix {prefix!r}")
        self._load_compatible_state(extracted)

    def _load_compatible_state(self, source_state: Mapping[str, torch.Tensor]) -> None:
        target_state = self.state_dict()
        optional_prefix = "cross_stage_attn."
        missing = [
            key
            for key in target_state
            if key not in source_state and not key.startswith(optional_prefix)
        ]
        mismatched = [
            key
            for key in target_state
            if key in source_state and source_state[key].shape != target_state[key].shape
        ]
        if missing or mismatched:
            raise ValueError(
                "incompatible regressor weights; "
                f"missing keys: {missing}, shape-mismatched keys: {mismatched}"
            )
        compatible_state = {
            key: source_state[key]
            for key in target_state
            if key in source_state
        }
        self.load_state_dict(compatible_state, strict=False)

    def forward(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_tokens: [B, 21, 256]
        Returns:
            vertices: [B, 778, 3]
        """
        if not isinstance(joint_tokens, torch.Tensor):
            raise TypeError(
                "joint_tokens must be a torch.Tensor with shape [B, 21, 256], "
                f"got {type(joint_tokens).__name__}"
            )
        if joint_tokens.ndim != 3 or tuple(joint_tokens.shape[1:]) != (21, 256):
            raise ValueError(
                "joint_tokens must have shape [B, 21, 256], "
                f"got {tuple(joint_tokens.shape)}"
            )

        x21 = self.encoder_1(self.proj_1(joint_tokens + self.pos_emb_1))
        x84 = self.upsample_1(x21)
        x84_encoded = self.encoder_2(self.proj_2(x84 + self.pos_emb_2))
        x336 = self.upsample_2(x84_encoded)
        x336_encoded = self.encoder_3(self.proj_3(x336 + self.pos_emb_3))
        x778 = self.upsample_3(x336_encoded)
        x778 = self.cross_stage_attn(x21, x84, x336, x778)
        return self.pred_final(x778)
