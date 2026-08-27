import math
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pl_system_v6_graphormer import PoseLightningModule
from torch.utils.checkpoint import checkpoint

from .auxiliary_head import RelationAwareAuxiliaryHead
from .kinematic_tree import AUXILIARY_PAIRS, MAX_TOPOLOGY_DISTANCE


class PoseLightningMTPModule(PoseLightningModule):
    def __init__(
        self,
        lr=1e-3,
        num_joints=21,
        local_model_dir=None,
        feature_dim=768,
        layers=[3, 6, -1],
        upsample_dim=512,
        num_refine_layers=3,
        use_gradient_checkpointing=True,
        enable_auxiliary_loss=True,
        lambda_aux=0.2,
        gamma=0.5,
        distance_embed_dim=32,
        auxiliary_hidden_dim=256,
    ):
        super().__init__(
            lr=lr,
            num_joints=num_joints,
            local_model_dir=local_model_dir,
            feature_dim=feature_dim,
            layers=layers,
            upsample_dim=upsample_dim,
            num_refine_layers=num_refine_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.enable_auxiliary_loss = enable_auxiliary_loss
        self.lambda_aux = lambda_aux
        self.gamma = gamma
        self.auxiliary_head = RelationAwareAuxiliaryHead(
            token_dim=self.upsample_dim,
            distance_embed_dim=distance_embed_dim,
            hidden_dim=auxiliary_hidden_dim,
            max_distance=MAX_TOPOLOGY_DISTANCE,
            dropout=0.1,
        )

        pair_tensor = torch.tensor(AUXILIARY_PAIRS, dtype=torch.long)
        self.register_buffer("aux_source_indices", pair_tensor[:, 0], persistent=False)
        self.register_buffer("aux_target_indices", pair_tensor[:, 1], persistent=False)
        self.register_buffer("aux_distances", pair_tensor[:, 2], persistent=False)
        self.register_buffer(
            "aux_distance_weights",
            torch.pow(torch.full((len(AUXILIARY_PAIRS),), float(self.gamma)), pair_tensor[:, 2].float()),
            persistent=False,
        )
        self.save_hyperparameters(ignore=["auxiliary_head"])

    def forward(self, x, hand_back):
        results: Dict[str, torch.Tensor] = {}
        batch_size = x.shape[0]

        features_dict = self.vitmodel(x)
        extracted_features = []
        for layer_id in self.layers:
            feat = features_dict[layer_id]
            patch_tokens = feat[:, 1:, :].transpose(1, 2)
            _, channels, num_patches = patch_tokens.shape
            height = width = int(math.sqrt(num_patches))
            extracted_features.append(patch_tokens.view(batch_size, channels, height, width))

        upsampled_features = [head(feat) for head, feat in zip(self.upsample_heads, extracted_features)]
        global_feature_map = self.fuse_block(upsampled_features)
        pos_embed_map = self.pos_embed_layer(global_feature_map)

        curr_tokens = self.joint_tokens.expand(batch_size, -1, -1)
        query_pos = self.joint_token_pos.expand(batch_size, -1, -1)

        all_stage_preds = []
        all_stage_logvars = []

        for i in range(self.num_refine_layers):
            if self.training and self.use_gradient_checkpointing:
                curr_tokens = checkpoint(self.layers_sa[i], curr_tokens, query_pos, use_reentrant=False)
                curr_tokens = checkpoint(
                    self.layers_ca[i],
                    curr_tokens,
                    global_feature_map,
                    query_pos,
                    pos_embed_map,
                    use_reentrant=False,
                )
            else:
                curr_tokens = self.layers_sa[i](x=curr_tokens, pos=query_pos)
                curr_tokens = self.layers_ca[i](
                    tgt=curr_tokens,
                    memory=global_feature_map,
                    query_pos=query_pos,
                    memory_pos=pos_embed_map,
                )

            raw_pred = self.pose_3d_head_PR(curr_tokens)
            all_stage_preds.append(raw_pred[..., :3])
            all_stage_logvars.append(raw_pred[..., 3:])

        results["pose3d"] = all_stage_preds[-1]
        results["pose3d_logvar"] = all_stage_logvars[-1]
        results["all_stage_pose3d"] = all_stage_preds
        results["all_stage_logvars"] = all_stage_logvars

        # Auxiliary prediction is training-only to keep inference cost unchanged.
        if self.training and self.enable_auxiliary_loss:
            results["aux_predictions"] = self.auxiliary_head.predict_from_pairs(
                joint_tokens=curr_tokens,
                source_indices=self.aux_source_indices,
                distances=self.aux_distances,
            )

        return results

    def _compute_auxiliary_loss(self, aux_predictions: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        batch_size = gt_pose.size(0)
        gather_indices = self.aux_target_indices.view(1, -1, 1).expand(batch_size, -1, gt_pose.size(-1))
        gt_targets = torch.gather(gt_pose, 1, gather_indices)
        pair_l1 = F.l1_loss(aux_predictions, gt_targets, reduction="none").mean(dim=-1)
        weighted = pair_l1 * self.aux_distance_weights.view(1, -1)
        return weighted.mean()

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_pose = batch["gt_pose"]
        hand_back = batch["hand_back"]

        results = self(imgs, hand_back)
        loss_main = 0.0
        for pred_mu, pred_logvar in zip(results["all_stage_pose3d"], results["all_stage_logvars"]):
            loss_main = loss_main + self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)

        aux_loss = torch.zeros((), device=gt_pose.device)
        if self.enable_auxiliary_loss and "aux_predictions" in results:
            aux_loss = self._compute_auxiliary_loss(results["aux_predictions"], gt_pose)

        total_loss = loss_main + self.lambda_aux * aux_loss

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_loss_main", loss_main, prog_bar=False)
        self.log("train_loss_aux", aux_loss, prog_bar=False)

        with torch.no_grad():
            mpjpe_3d = self._compute_mpjpe_3d(results["pose3d"], gt_pose)
            pa_mpjpe_3d = self._compute_pa_mpjpe_3d(results["pose3d"], gt_pose)
            self.log("train_mpjpe_3d", mpjpe_3d, prog_bar=True)
            self.log("train_pa_mpjpe_3d", pa_mpjpe_3d, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_pose = batch["gt_pose"]
        hand_back = batch["hand_back"]
        results = self(imgs, hand_back)

        val_loss = self._compute_gnll_loss(results["pose3d"], results["pose3d_logvar"], gt_pose)
        val_mpjpe_3d = self._compute_mpjpe_3d(results["pose3d"], gt_pose)
        val_pa_mpjpe_3d = self._compute_pa_mpjpe_3d(results["pose3d"], gt_pose)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mpjpe_3d", val_mpjpe_3d, prog_bar=True)
        self.log("val_pa_mpjpe_3d", val_pa_mpjpe_3d, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_pose = batch["gt_pose"]
        hand_back = batch["hand_back"]
        results = self(imgs, hand_back)

        test_loss = self._compute_gnll_loss(results["pose3d"], results["pose3d_logvar"], gt_pose)
        test_mpjpe_3d = self._compute_mpjpe_3d(results["pose3d"], gt_pose)
        test_pa_mpjpe_3d = self._compute_pa_mpjpe_3d(results["pose3d"], gt_pose)

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_mpjpe_3d", test_mpjpe_3d, prog_bar=True)
        self.log("test_pa_mpjpe_3d", test_pa_mpjpe_3d, prog_bar=True)
