from typing import Tuple

import torch
import torch.nn as nn


class RelationAwareAuxiliaryHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        distance_embed_dim: int = 32,
        hidden_dim: int = 256,
        max_distance: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.distance_embedding = nn.Embedding(max_distance + 1, distance_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim + distance_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, src_tokens: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        dist_embed = self.distance_embedding(distances)
        features = torch.cat([src_tokens, dist_embed], dim=-1)
        return self.mlp(features)

    def predict_from_pairs(
        self,
        joint_tokens: torch.Tensor,
        source_indices: torch.Tensor,
        distances: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = joint_tokens.size(0)
        expanded_indices = source_indices.view(1, -1, 1).expand(batch_size, -1, joint_tokens.size(-1))
        src_tokens = torch.gather(joint_tokens, 1, expanded_indices)
        return self.forward(src_tokens, distances.view(1, -1).expand(batch_size, -1))
