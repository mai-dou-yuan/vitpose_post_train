"""Layers required by the standalone mesh regressor.

These implementations preserve the module structure and parameter names used by
the corresponding simpleHand/timm layers, while depending only on PyTorch.
"""

from typing import Type

import torch
import torch.nn as nn


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """Apply stochastic depth per sample, as used by timm's DropPath."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """PyTorch-only equivalent of the timm MLP used by simpleHand."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class AttentionBlock(nn.Module):
    """Attention/FFN residual block copied from simpleHand's MeshHead path."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        mlp_ratio: float = 4.0,
        nhead: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.pre_norm = pre_norm

        self.norm1 = norm_layer(dim_out) if pre_norm else nn.Identity()
        self.token_mixer = nn.MultiheadAttention(
            dim_out, nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            act_layer=act_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim != self.dim_out:
            x = self.proj(x)
        x_norm = self.norm1(x)
        x = x + self.drop_path(
            self.token_mixer(x_norm, x_norm, x_norm, need_weights=False)[0]
        )
        return x + self.drop_path(self.mlp(self.norm2(x)))


class LinearUpsample(nn.Module):
    """Linearly upsample along the token dimension."""

    def __init__(self, node_num_in: int, node_num_out: int) -> None:
        super().__init__()
        self.upsample = nn.Linear(node_num_in, node_num_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.upsample(x)
        return x.permute(0, 2, 1)


class MultiScaleCrossStageAttention(nn.Module):
    """Fuse 21/84/336-token history into the final 778 vertex tokens."""

    def __init__(
        self,
        d_attn: int = 64,
        num_heads: int = 4,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        if d_attn != 64:
            raise ValueError("d_attn must be 64 to match the vertex token width")
        if d_attn % num_heads != 0:
            raise ValueError("d_attn must be divisible by num_heads")

        self.proj21 = nn.Linear(256, d_attn)
        self.proj84 = nn.Linear(256, d_attn)
        self.proj336 = nn.Linear(128, d_attn)

        self.stage_embed21 = nn.Parameter(torch.zeros(1, 1, d_attn))
        self.stage_embed84 = nn.Parameter(torch.zeros(1, 1, d_attn))
        self.stage_embed336 = nn.Parameter(torch.zeros(1, 1, d_attn))

        self.norm_q = nn.LayerNorm(d_attn)
        self.norm_kv = nn.LayerNorm(d_attn)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(d_attn)
        self.ffn = nn.Sequential(
            nn.Linear(d_attn, d_attn * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_attn * ffn_ratio, d_attn),
        )

    def forward(
        self,
        x21: torch.Tensor,
        x84: torch.Tensor,
        x336: torch.Tensor,
        x778: torch.Tensor,
    ) -> torch.Tensor:
        h21 = self.proj21(x21) + self.stage_embed21
        h84 = self.proj84(x84) + self.stage_embed84
        h336 = self.proj336(x336) + self.stage_embed336
        history = torch.cat((h21, h84, h336), dim=1)

        kv = self.norm_kv(history)
        cross_feature = self.cross_attn(
            query=self.norm_q(x778),
            key=kv,
            value=kv,
            need_weights=False,
        )[0]
        x778 = x778 + cross_feature
        return x778 + self.ffn(self.norm_ffn(x778))
