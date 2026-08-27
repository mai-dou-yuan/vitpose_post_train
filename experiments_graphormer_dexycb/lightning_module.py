import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pl_system_v6_graphormer import PoseLightningModule


class DexYCBPoseLightningModule(PoseLightningModule):
    """
    Thin wrapper around the original Graphormer lightning module.

    The original module expects `hand_back` and `dist_coeffs` in the batch, but
    its current forward path only uses the RGB image tensor. This wrapper keeps
    the original model weights and losses while removing the DexYCB-incompatible
    batch contract from the training loop.
    """

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_pose = batch["gt_pose"]
        results = self(imgs, imgs)

        loss_3d_pose = 0.0
        for pred_mu, pred_logvar in zip(results["all_stage_pose3d"], results["all_stage_logvars"]):
            loss_3d_pose = loss_3d_pose + self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)

        self.log("train_loss", loss_3d_pose, prog_bar=True)
        with torch.no_grad():
            mpjpe_3d = self._compute_mpjpe_3d(results["pose3d"], gt_pose)
            pa_mpjpe_3d = self._compute_pa_mpjpe_3d(results["pose3d"], gt_pose)
            self.log("train_mpjpe_3d", mpjpe_3d, prog_bar=True)
            self.log("train_pa_mpjpe_3d", pa_mpjpe_3d, prog_bar=True)

        return loss_3d_pose

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_pose = batch["gt_pose"]
        results = self(imgs, imgs)

        pred_mu = results["pose3d"]
        pred_logvar = results["pose3d_logvar"]

        val_loss = self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)
        val_mpjpe_3d = self._compute_mpjpe_3d(pred_mu, gt_pose)
        val_pa_mpjpe_3d = self._compute_pa_mpjpe_3d(pred_mu, gt_pose)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_mpjpe_3d", val_mpjpe_3d, prog_bar=True)
        self.log("val_pa_mpjpe_3d", val_pa_mpjpe_3d, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs = batch["img"]
        gt_pose = batch["gt_pose"]
        results = self(imgs, imgs)

        pred_mu = results["pose3d"]
        pred_logvar = results["pose3d_logvar"]

        test_loss = self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)
        test_mpjpe_3d = self._compute_mpjpe_3d(pred_mu, gt_pose)
        test_pa_mpjpe_3d = self._compute_pa_mpjpe_3d(pred_mu, gt_pose)
        test_root_rigid_mpjpe_3d = self._compute_root_rigid_mpjpe_3d(pred_mu, gt_pose)

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_mpjpe_3d", test_mpjpe_3d, prog_bar=True)
        self.log("test_pa_mpjpe_3d", test_pa_mpjpe_3d, prog_bar=True)
        self.log("test_root_rigid_mpjpe_3d", test_root_rigid_mpjpe_3d, prog_bar=True)
