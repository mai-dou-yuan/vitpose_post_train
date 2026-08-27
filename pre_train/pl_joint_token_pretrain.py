import math
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_vit import ViTFeatureExtractor
from pl_system_v6_graphormer import HandGraphormerLayer, PoseRefinementLayer


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class JointTopologyPrior(nn.Module):
    """Builds an explicit hand-structure prior for the 21 joint queries."""

    def __init__(self, num_joints: int = 21, d_model: int = 512):
        super().__init__()
        if num_joints != 21:
            raise ValueError("JointTopologyPrior currently assumes the 21-joint hand layout.")

        spd = self._build_shortest_path_distance(num_joints)
        self.register_buffer("spd", spd.float())

        finger_ids = torch.tensor(
            [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
            dtype=torch.long,
        )
        depth_ids = torch.tensor(
            [0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            dtype=torch.long,
        )
        self.register_buffer("finger_ids", finger_ids)
        self.register_buffer("depth_ids", depth_ids)

        self.spd_proj = nn.Sequential(
            nn.Linear(num_joints, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.finger_embed = nn.Embedding(6, d_model)
        self.depth_embed = nn.Embedding(5, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _build_shortest_path_distance(num_joints: int) -> torch.Tensor:
        spd = torch.full((num_joints, num_joints), float("inf"))
        for i in range(num_joints):
            spd[i, i] = 0
        for i, j in HAND_EDGES:
            spd[i, j] = 1
            spd[j, i] = 1

        for k in range(num_joints):
            for i in range(num_joints):
                for j in range(num_joints):
                    spd[i, j] = min(spd[i, j], spd[i, k] + spd[k, j])
        return spd

    def forward(self) -> torch.Tensor:
        prior = (
            self.spd_proj(self.spd)
            + self.finger_embed(self.finger_ids)
            + self.depth_embed(self.depth_ids)
        )
        return self.norm(prior).unsqueeze(0)


class SimpleJointRegressionHead(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        last = self.net[-1]
        nn.init.normal_(last.weight, mean=0.0, std=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JointTokenPretrainModule(pl.LightningModule):
    """
    Lightweight joint-query pretraining.

    This module intentionally skips upsample_heads/fuse_block. Joint tokens cross-attend
    directly to frozen ViT patch tokens, so the exported prior is focused on query
    semantics instead of the full pose decoder.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        num_joints: int = 21,
        local_model_dir: Optional[str] = None,
        feature_dim: int = 768,
        d_model: int = 512,
        vit_layers: Optional[List[int]] = None,
        num_refine_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_patch_tokens: int = 4096,
        topology_alpha_init: float = 0.0,
        rel_loss_weight: float = 0.5,
        weight_decay: float = 0.04,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        if vit_layers is None:
            vit_layers = [-1]
        self.save_hyperparameters()

        self.num_joints = num_joints
        self.d_model = d_model
        self.vit_layers = vit_layers
        self.num_refine_layers = num_refine_layers
        self.rel_loss_weight = rel_loss_weight
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs

        self.vitmodel = ViTFeatureExtractor(
            model_name_or_path=local_model_dir,
            layers_to_extract=vit_layers,
            freeze_backbone=True,
        )
        for param in self.vitmodel.parameters():
            param.requires_grad = False

        self.patch_proj = nn.Linear(feature_dim * len(vit_layers), d_model)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, max_patch_tokens, d_model))
        nn.init.normal_(self.patch_pos_embed, std=0.02)

        self.joint_tokens = nn.Parameter(torch.empty(1, num_joints, d_model))
        self.joint_token_pos = nn.Parameter(torch.empty(1, num_joints, d_model))
        nn.init.normal_(self.joint_tokens, std=0.02)
        nn.init.normal_(self.joint_token_pos, std=0.02)

        self.topology_prior = JointTopologyPrior(num_joints=num_joints, d_model=d_model)
        self.topology_alpha = nn.Parameter(torch.tensor(float(topology_alpha_init)))

        self.layers_sa = nn.ModuleList(
            [
                HandGraphormerLayer(
                    d_model=d_model,
                    n_head=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    num_joints=num_joints,
                )
                for _ in range(num_refine_layers)
            ]
        )
        self.layers_ca = nn.ModuleList(
            [
                PoseRefinementLayer(
                    d_model=d_model,
                    n_head=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                )
                for _ in range(num_refine_layers)
            ]
        )
        self.head = SimpleJointRegressionHead(d_model=d_model, hidden_dim=d_model // 2, dropout=dropout)

    def _extract_patch_tokens(self, imgs: torch.Tensor) -> torch.Tensor:
        self.vitmodel.eval()
        with torch.no_grad():
            features_dict = self.vitmodel(imgs)

        patch_tokens = []
        for layer_id in self.vit_layers:
            feat = features_dict[layer_id]
            patch_tokens.append(feat[:, 1:, :])
        return torch.cat(patch_tokens, dim=-1)

    def _get_patch_pos(self, num_patches: int, device: torch.device) -> torch.Tensor:
        if num_patches > self.patch_pos_embed.size(1):
            raise ValueError(
                f"num_patches={num_patches} exceeds max_patch_tokens="
                f"{self.patch_pos_embed.size(1)}. Increase --max-patch-tokens."
            )
        return self.patch_pos_embed[:, :num_patches, :].to(device)

    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = imgs.size(0)
        patch_tokens = self._extract_patch_tokens(imgs)
        memory = self.patch_proj(patch_tokens)
        memory_pos = self._get_patch_pos(memory.size(1), memory.device).expand(batch_size, -1, -1)

        curr_tokens = self.joint_tokens.expand(batch_size, -1, -1)
        query_pos = self.joint_token_pos.expand(batch_size, -1, -1)
        topo = self.topology_prior().to(curr_tokens.dtype).expand(batch_size, -1, -1)
        curr_tokens = curr_tokens + self.topology_alpha * topo

        for sa_layer, ca_layer in zip(self.layers_sa, self.layers_ca):
            curr_tokens = sa_layer(x=curr_tokens, pos=query_pos)
            curr_tokens = ca_layer(
                tgt=curr_tokens,
                memory=memory,
                query_pos=query_pos,
                memory_pos=memory_pos,
            )

        pred_pose = self.head(curr_tokens)
        return {"pose3d": pred_pose, "joint_tokens_refined": curr_tokens}

    def _compute_loss(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        abs_loss = F.smooth_l1_loss(pred_pose, gt_pose)
        pred_rel = pred_pose - pred_pose[:, 0:1, :]
        gt_rel = gt_pose - gt_pose[:, 0:1, :]
        rel_loss = F.smooth_l1_loss(pred_rel, gt_rel)
        loss = abs_loss + self.rel_loss_weight * rel_loss
        return {"loss": loss, "loss_abs": abs_loss, "loss_rel": rel_loss}

    @staticmethod
    def _compute_mpjpe(pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        return torch.norm(pred_pose - gt_pose, dim=-1).mean()

    def training_step(self, batch, batch_idx):
        results = self(batch["img"])
        losses = self._compute_loss(results["pose3d"], batch["gt_pose"])
        mpjpe = self._compute_mpjpe(results["pose3d"], batch["gt_pose"])

        self.log("train_loss", losses["loss"], prog_bar=True)
        self.log("train_loss_abs", losses["loss_abs"])
        self.log("train_loss_rel", losses["loss_rel"])
        self.log("train_mpjpe_3d", mpjpe, prog_bar=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        results = self(batch["img"])
        losses = self._compute_loss(results["pose3d"], batch["gt_pose"])
        mpjpe = self._compute_mpjpe(results["pose3d"], batch["gt_pose"])

        self.log("val_loss", losses["loss"], prog_bar=True)
        self.log("val_loss_abs", losses["loss_abs"])
        self.log("val_loss_rel", losses["loss_rel"])
        self.log("val_mpjpe_3d", mpjpe, prog_bar=True)

    def test_step(self, batch, batch_idx):
        results = self(batch["img"])
        losses = self._compute_loss(results["pose3d"], batch["gt_pose"])
        mpjpe = self._compute_mpjpe(results["pose3d"], batch["gt_pose"])

        self.log("test_loss", losses["loss"], prog_bar=True)
        self.log("test_mpjpe_3d", mpjpe, prog_bar=True)

    def export_joint_query_prior(self) -> Dict[str, torch.Tensor]:
        prior = {
            "joint_tokens": self.joint_tokens.detach().cpu(),
            "joint_token_pos": self.joint_token_pos.detach().cpu(),
            "topology_alpha": self.topology_alpha.detach().cpu(),
        }
        for idx, layer in enumerate(self.layers_sa):
            if hasattr(layer, "spatial_bias_table"):
                prior[f"layers_sa.{idx}.spatial_bias_table.weight"] = (
                    layer.spatial_bias_table.weight.detach().cpu()
                )
            if hasattr(layer, "same_finger_bias"):
                prior[f"layers_sa.{idx}.same_finger_bias.weight"] = (
                    layer.same_finger_bias.weight.detach().cpu()
                )
        return prior

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.weight_decay,
        )

        max_epochs = self.trainer.max_epochs or 30
        warmup_epochs = min(self.warmup_epochs, max(1, max_epochs - 1))
        cosine_epochs = max(1, max_epochs - warmup_epochs)
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.001,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
