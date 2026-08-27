from typing import Any, Dict

import pytorch_lightning as pl
import torch

from losses import (
    build_self_occ_mask,
    compute_gnll_loss_masked,
    diagonal_gaussian_kl_occ_stable,
    token_distill_occ_safe,
)
from metrics import compute_pose_metrics
from model_wrappers import GraphormerOcclusionWrapper


class OcclusionDistillationModule(pl.LightningModule):
    def __init__(
        self,
        student: GraphormerOcclusionWrapper,
        expert: GraphormerOcclusionWrapper,
        *,
        lr: float,
        gnll_warmup_epochs: int = 0,
        scheduler_warmup_epochs: int = 5,
        weight_decay: float = 0.04,
        min_lr: float = 1e-6,
        lambda_kl: float = 0.5,
        lambda_token: float = 0.2,
        kd_warmup_epochs: int = 5,
        lambda_kl_warmup: float = 0.1,
        lambda_token_warmup: float = 0.05,
    ):
        super().__init__()
        self.student = student
        self.student.set_gnll_warmup(gnll_warmup_epochs)
        self.expert = expert
        self.expert.set_gnll_warmup(0)
        self.expert.eval()
        for param in self.expert.parameters():
            param.requires_grad = False

        self.lr = lr
        self.gnll_warmup_epochs = gnll_warmup_epochs
        self.scheduler_warmup_epochs = scheduler_warmup_epochs
        self.weight_decay = weight_decay
        self.min_lr = min_lr
        self.lambda_kl = lambda_kl
        self.lambda_token = lambda_token
        self.kd_warmup_epochs = kd_warmup_epochs
        self.lambda_kl_warmup = lambda_kl_warmup
        self.lambda_token_warmup = lambda_token_warmup
        self.save_hyperparameters(ignore=["student", "expert"])

    def forward(self, imgs: torch.Tensor, hand_back: torch.Tensor) -> Dict[str, Any]:
        return self.student(imgs, hand_back)

    def _current_kd_weights(self):
        if self.current_epoch < self.kd_warmup_epochs:
            return self.lambda_kl_warmup, self.lambda_token_warmup
        return self.lambda_kl, self.lambda_token

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        hand_back = batch["hand_back"]
        gt_pose = batch["gt_pose"]
        occ_mask = build_self_occ_mask(batch)

        student_out = self.student(imgs, hand_back)
        with torch.no_grad():
            expert_out = self.expert(imgs, hand_back)

        loss_unc = gt_pose.sum() * 0.0
        for pred_mu, pred_logvar in zip(student_out["all_stage_pose3d"], student_out["all_stage_logvars"]):
            loss_unc = loss_unc + compute_gnll_loss_masked(
                pred_mu=pred_mu,
                pred_logvar=pred_logvar,
                gt=gt_pose,
                mask=None,
                current_epoch=self.current_epoch,
                warmup_epochs=self.gnll_warmup_epochs,
            )
        loss_unc = loss_unc / len(student_out["all_stage_pose3d"])

        loss_kl = gt_pose.sum() * 0.0
        for mu_e, logvar_e, mu_s, logvar_s in zip(
            expert_out["all_stage_pose3d"],
            expert_out["all_stage_logvars"],
            student_out["all_stage_pose3d"],
            student_out["all_stage_logvars"],
        ):
            loss_kl = loss_kl + diagonal_gaussian_kl_occ_stable(
                mu_e=mu_e,
                logvar_e=logvar_e,
                mu_s=mu_s,
                logvar_s=logvar_s,
                occ_mask=occ_mask,
            )
        loss_kl = loss_kl / len(student_out["all_stage_pose3d"])

        loss_token = token_distill_occ_safe(
            token_s=student_out["joint_token"],
            token_e=expert_out["joint_token"],
            occ_mask=occ_mask,
        )

        lambda_kl, lambda_token = self._current_kd_weights()
        loss_total = loss_unc + lambda_kl * loss_kl + lambda_token * loss_token

        metrics = compute_pose_metrics(student_out["pose3d"], gt_pose, batch)
        self.log("train_loss", loss_total, prog_bar=True)
        self.log("train_loss_unc", loss_unc, prog_bar=False)
        self.log("train_loss_kl_occ", loss_kl, prog_bar=False)
        self.log("train_loss_token_occ", loss_token, prog_bar=False)
        self.log("train_lambda_kl", torch.tensor(lambda_kl, device=self.device), prog_bar=False)
        self.log("train_lambda_token", torch.tensor(lambda_token, device=self.device), prog_bar=False)
        for name, value in metrics.items():
            self.log(f"train_{name}", value, prog_bar=name in {"overall_mpjpe", "visible_mpjpe", "self_occ_mpjpe"})
        return loss_total

    def _shared_eval(self, batch, stage_prefix: str) -> torch.Tensor:
        outputs = self.student(batch["img"], batch["hand_back"])
        gt_pose = batch["gt_pose"]

        loss_unc = gt_pose.sum() * 0.0
        for pred_mu, pred_logvar in zip(outputs["all_stage_pose3d"], outputs["all_stage_logvars"]):
            loss_unc = loss_unc + compute_gnll_loss_masked(
                pred_mu=pred_mu,
                pred_logvar=pred_logvar,
                gt=gt_pose,
                mask=None,
                current_epoch=self.current_epoch,
                warmup_epochs=self.gnll_warmup_epochs,
            )
        loss_unc = loss_unc / len(outputs["all_stage_pose3d"])

        metrics = compute_pose_metrics(outputs["pose3d"], gt_pose, batch)
        self.log(f"{stage_prefix}_loss", loss_unc, prog_bar=(stage_prefix != "test"))
        for name, value in metrics.items():
            self.log(f"{stage_prefix}_{name}", value, prog_bar=name in {"overall_mpjpe", "visible_mpjpe", "self_occ_mpjpe"})
        return loss_unc

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, "test")

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.student.parameters())
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
