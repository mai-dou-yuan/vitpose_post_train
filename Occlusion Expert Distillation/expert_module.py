from typing import Any, Dict

import pytorch_lightning as pl
import torch

from losses import build_self_occ_mask, build_visible_mask, compute_expert_gnll_components
from metrics import compute_pose_metrics
from model_wrappers import GraphormerOcclusionWrapper


class OcclusionExpertModule(pl.LightningModule):
    def __init__(
        self,
        model: GraphormerOcclusionWrapper,
        *,
        lr: float,
        w_occ: float = 1.0,
        w_vis: float = 0.1,
        gnll_warmup_epochs: int = 0,
        scheduler_warmup_epochs: int = 5,
        weight_decay: float = 0.04,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.model.set_gnll_warmup(gnll_warmup_epochs)
        self.lr = lr
        self.w_occ = w_occ
        self.w_vis = w_vis
        self.gnll_warmup_epochs = gnll_warmup_epochs
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.save_hyperparameters(ignore=["model"])

    def forward(self, imgs: torch.Tensor, hand_back: torch.Tensor) -> Dict[str, Any]:
        return self.model(imgs, hand_back)

    def _shared_eval(self, batch, stage_prefix: str) -> torch.Tensor:
        outputs = self(batch["img"], batch["hand_back"])
        gt_pose = batch["gt_pose"]
        occ_mask = build_self_occ_mask(batch)
        vis_mask = build_visible_mask(batch)

        loss_total = gt_pose.sum() * 0.0
        occ_raw_total = gt_pose.sum() * 0.0
        vis_raw_total = gt_pose.sum() * 0.0
        for pred_mu, pred_logvar in zip(outputs["all_stage_pose3d"], outputs["all_stage_logvars"]):
            components = compute_expert_gnll_components(
                pred_mu=pred_mu,
                pred_logvar=pred_logvar,
                gt=gt_pose,
                occ_mask=occ_mask,
                vis_mask=vis_mask,
                current_epoch=self.current_epoch,
                warmup_epochs=self.gnll_warmup_epochs,
                w_occ=self.w_occ,
                w_vis=self.w_vis,
            )
            loss_total = loss_total + components["total"]
            occ_raw_total = occ_raw_total + components["occ_raw"]
            vis_raw_total = vis_raw_total + components["vis_raw"]
        num_stages = len(outputs["all_stage_pose3d"])
        loss_total = loss_total / num_stages
        occ_raw_total = occ_raw_total / num_stages
        vis_raw_total = vis_raw_total / num_stages

        metrics = compute_pose_metrics(outputs["pose3d"], gt_pose, batch)
        self.log(f"{stage_prefix}_loss", loss_total, prog_bar=(stage_prefix != "test"))
        for name, value in metrics.items():
            self.log(f"{stage_prefix}_{name}", value, prog_bar=name in {"overall_mpjpe", "visible_mpjpe", "self_occ_mpjpe"})
        if stage_prefix == "val":
            self.log("val_occ_gnll_loss", occ_raw_total, prog_bar=False)
            self.log("val_vis_gnll_loss", vis_raw_total, prog_bar=False)
        return loss_total

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, "test")

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

        max_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 100
        warmup_epochs = min(self.scheduler_warmup_epochs, max(max_epochs - 1, 0))
        if warmup_epochs <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=self.min_lr)
        else:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(max_epochs - warmup_epochs, 1),
                eta_min=self.min_lr,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
            },
        }
