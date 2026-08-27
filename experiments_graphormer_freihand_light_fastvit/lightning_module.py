import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiments_graphormer_freihand_light_fastvit.pl_system_v6_graphormer import (
    PoseLightningModule,
)


class FreiHANDPoseLightningModule(PoseLightningModule):
    """
    FreiHAND-specific wrapper around the original Graphormer LightningModule.

    The upstream training_step/validation_step/test_step assume extra fields
    from other datasets. This wrapper narrows the batch contract to FreiHAND's
    RGB, 2D/3D joints, MANO vertices and intrinsics.
    """
    root_joint_idx = 0
    hand_bones = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    )

    def _make_root_relative(self, joints):
        root = joints[:, self.root_joint_idx : self.root_joint_idx + 1, :]
        return joints - root

    def _project_3d_to_2d(self, joints_3d, cam_k, eps=1e-6):
        points_homo = torch.matmul(joints_3d, cam_k.transpose(1, 2))
        z = points_homo[..., 2:3].clamp_min(eps)
        return points_homo[..., :2] / z

    def _joint_valid_mask(self, batch, gt_joints_3d):
        valid = torch.isfinite(gt_joints_3d).all(dim=-1)
        for key in ("joint_valid", "joints_valid", "joint_mask", "visibility", "valid"):
            if key not in batch:
                continue
            batch_valid = batch[key].to(device=gt_joints_3d.device)
            if batch_valid.ndim == 1:
                batch_valid = batch_valid[:, None]
            if batch_valid.ndim == 3 and batch_valid.shape[-1] == 1:
                batch_valid = batch_valid.squeeze(-1)
            if batch_valid.ndim != 2:
                raise ValueError(f"Unsupported {key} mask shape: {tuple(batch_valid.shape)}")
            valid = valid & batch_valid.bool().expand_as(valid)
            break
        return valid

    def _compute_joint_2d_loss(
        self, pred_joints_3d, gt_joints_2d, cam_k, image_size, valid_mask=None
    ):
        """Project token-head joints and apply the reference branch's 2D loss."""
        if pred_joints_3d.ndim != 3 or tuple(pred_joints_3d.shape[1:]) != (21, 3):
            raise ValueError(
                "Transformer joint predictions must have shape [B, 21, 3], got "
                f"{tuple(pred_joints_3d.shape)}"
            )
        if gt_joints_2d.shape != pred_joints_3d.shape[:-1] + (2,):
            raise ValueError(
                "2D joint targets must have shape [B, 21, 2], got "
                f"{tuple(gt_joints_2d.shape)}"
            )
        gt_joints_2d = gt_joints_2d.to(
            device=pred_joints_3d.device, dtype=pred_joints_3d.dtype
        )
        cam_k = cam_k.to(device=pred_joints_3d.device, dtype=pred_joints_3d.dtype)
        pred_joints_2d = self._project_3d_to_2d(pred_joints_3d, cam_k)
        valid = (
            torch.isfinite(pred_joints_2d).all(dim=-1)
            & torch.isfinite(gt_joints_2d).all(dim=-1)
        )
        if valid_mask is not None:
            valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)

        if not torch.any(valid):
            safe_prediction = torch.where(
                torch.isfinite(pred_joints_2d),
                pred_joints_2d,
                torch.zeros_like(pred_joints_2d),
            )
            return safe_prediction.sum() * 0.0

        image_scale = float(max(image_size))
        loss = F.smooth_l1_loss(
            pred_joints_2d / image_scale,
            gt_joints_2d / image_scale,
            reduction="none",
            beta=1.0,
        ).mean(dim=-1)
        return loss[valid].mean()

    def _normalized_stage_weights(self, num_stages, reference_tensor):
        """Return increasing deep-supervision weights on the prediction device."""
        if num_stages < 1:
            raise ValueError("Deep supervision requires at least one prediction stage")
        if num_stages > len(self.stage_supervision_weights):
            raise ValueError(
                f"Configured {len(self.stage_supervision_weights)} stage weights, "
                f"but the model returned {num_stages} prediction stages"
            )
        weights = reference_tensor.new_tensor(
            self.stage_supervision_weights[-num_stages:]
        )
        return weights / weights.sum()

    def _compute_initial_2d_metrics(
        self, pred_normalized, gt_joints_2d, image_size, valid_mask=None
    ):
        """Compute the supervised loss and pixel EPE for the initial 2D head."""
        if pred_normalized.shape != gt_joints_2d.shape:
            raise ValueError(
                "Initial 2D prediction and target shapes must match, got "
                f"{tuple(pred_normalized.shape)} and {tuple(gt_joints_2d.shape)}"
            )
        image_h, image_w = image_size
        pixel_scale = pred_normalized.new_tensor(
            (float(image_w) - 1.0, float(image_h) - 1.0)
        ).clamp_min(1.0)
        gt_joints_2d = gt_joints_2d.to(
            device=pred_normalized.device, dtype=pred_normalized.dtype
        )
        gt_normalized = gt_joints_2d / pixel_scale
        valid = (
            torch.isfinite(pred_normalized).all(dim=-1)
            & torch.isfinite(gt_normalized).all(dim=-1)
            & (gt_normalized >= 0.0).all(dim=-1)
            & (gt_normalized <= 1.0).all(dim=-1)
        )
        if valid_mask is not None:
            valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)

        if not torch.any(valid):
            safe_prediction = torch.where(
                torch.isfinite(pred_normalized),
                pred_normalized,
                torch.zeros_like(pred_normalized),
            )
            zero = safe_prediction.sum() * 0.0
            return {
                "loss": zero,
                "epe_px": zero.detach(),
            }

        per_joint_loss = F.l1_loss(
            pred_normalized,
            gt_normalized,
            reduction="none",
        ).mean(dim=-1)
        pred_pixels = pred_normalized * pixel_scale
        epe_px = torch.linalg.vector_norm(pred_pixels - gt_joints_2d, dim=-1)
        return {
            "loss": per_joint_loss[valid].mean(),
            "epe_px": epe_px[valid].mean().detach(),
        }

    def _log_initial_2d_metrics(self, prefix, metrics, batch_size):
        self.log(f"{prefix}_initial_2d_loss", metrics["loss"], prog_bar=False)
        self.log(
            f"{prefix}_initial_2d_epe_px",
            metrics["epe_px"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )

    def _compute_mpjpe_3d(self, pred, gt, valid_mask=None):
        pred_rel = self._make_root_relative(pred)
        gt_rel = self._make_root_relative(gt)
        return super()._compute_mpjpe_3d(pred_rel, gt_rel, valid_mask)

    def _compute_pa_mpjpe_3d(self, pred, gt):
        pred_rel = self._make_root_relative(pred)
        gt_rel = self._make_root_relative(gt)
        return super()._compute_pa_mpjpe_3d(pred_rel, gt_rel)

    def _compute_root_rigid_mpjpe_3d(self, pred, gt):
        pred_rel = self._make_root_relative(pred)
        gt_rel = self._make_root_relative(gt)
        return super()._compute_root_rigid_mpjpe_3d(pred_rel, gt_rel)

    def _compute_mpvpe(self, pred_vertices, gt_vertices, valid_mask=None):
        """Mean per-vertex Euclidean error in the supervised coordinate frame."""
        return super()._compute_mpjpe_3d(
            pred_vertices, gt_vertices, valid_mask=valid_mask
        )

    def _compute_pa_mpvpe(self, pred_vertices, gt_vertices):
        """MPVPE after a per-sample similarity Procrustes alignment."""
        return super()._compute_pa_mpjpe_3d(pred_vertices, gt_vertices)

    def _compute_scale_diagnostics(self, pred, gt, valid_mask=None, eps=1e-8):
        pred_rel = self._make_root_relative(pred)
        gt_rel = self._make_root_relative(gt)
        if valid_mask is None:
            valid_mask = torch.ones_like(pred_rel[..., 0], dtype=torch.bool)
        valid = valid_mask.to(device=pred.device, dtype=torch.bool)
        valid_xyz = valid.unsqueeze(-1).expand_as(pred_rel)

        pred_masked = torch.where(valid_xyz, pred_rel, torch.zeros_like(pred_rel))
        gt_masked = torch.where(valid_xyz, gt_rel, torch.zeros_like(gt_rel))
        numerator = (pred_masked * gt_masked).sum(dim=(1, 2))
        denominator = pred_masked.square().sum(dim=(1, 2)).clamp_min(eps)
        scale = numerator / denominator
        pred_scale_aligned = pred_rel * scale[:, None, None]
        joint_error = torch.linalg.vector_norm(pred_scale_aligned - gt_rel, dim=-1)
        n_mpjpe = joint_error.masked_select(valid).mean()

        coordinate_error = (pred_rel - gt_rel).abs()
        axis_mae = []
        for axis in range(3):
            axis_mae.append(coordinate_error[..., axis].masked_select(valid).mean())

        pred_rms = torch.sqrt(
            pred_masked.square().sum(dim=(1, 2))
            / valid_xyz.sum(dim=(1, 2)).clamp_min(1)
        )
        gt_rms = torch.sqrt(
            gt_masked.square().sum(dim=(1, 2))
            / valid_xyz.sum(dim=(1, 2)).clamp_min(1)
        )
        scale_ratio = (pred_rms / gt_rms.clamp_min(eps)).mean()

        pred_bone_lengths = torch.stack(
            [torch.linalg.vector_norm(pred_rel[:, a] - pred_rel[:, b], dim=-1)
             for a, b in self.hand_bones],
            dim=1,
        )
        gt_bone_lengths = torch.stack(
            [torch.linalg.vector_norm(gt_rel[:, a] - gt_rel[:, b], dim=-1)
             for a, b in self.hand_bones],
            dim=1,
        )
        bone_valid = torch.stack(
            [valid[:, a] & valid[:, b] for a, b in self.hand_bones], dim=1
        )
        bone_length_error = (
            pred_bone_lengths - gt_bone_lengths
        ).abs().masked_select(bone_valid).mean()

        return {
            "n_mpjpe_3d": n_mpjpe,
            "mae_x_3d": axis_mae[0],
            "mae_y_3d": axis_mae[1],
            "mae_z_3d": axis_mae[2],
            "scale_ratio": scale_ratio,
            "bone_length_error": bone_length_error,
        }

    def _log_scale_diagnostics(self, prefix, pred, gt, valid_mask):
        diagnostics = self._compute_scale_diagnostics(pred, gt, valid_mask)
        for name, value in diagnostics.items():
            self.log(f"{prefix}_{name}", value, prog_bar=False)

    @staticmethod
    def _masked_l1_loss(pred, target, valid_mask=None):
        if pred.shape != target.shape:
            raise ValueError(
                f"L1 prediction and target shapes must match, got "
                f"{tuple(pred.shape)} and {tuple(target.shape)}"
            )
        finite = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target).all(dim=-1)
        if valid_mask is not None:
            finite = finite & valid_mask.to(device=finite.device, dtype=torch.bool)
        if not torch.any(finite):
            safe_prediction = torch.where(
                torch.isfinite(pred), pred, torch.zeros_like(pred)
            )
            return safe_prediction.sum() * 0.0
        per_point_loss = F.l1_loss(pred, target, reduction="none").mean(dim=-1)
        return per_point_loss[finite].mean()

    def _mesh_supervision_targets(self, batch):
        if "gt_vertices" not in batch:
            raise KeyError(
                "FreiHAND mesh supervision requires batch['gt_vertices'] with "
                "shape [B, 778, 3]"
            )
        gt_pose = batch["gt_pose"]
        gt_vertices = batch["gt_vertices"].to(
            device=gt_pose.device, dtype=gt_pose.dtype
        )
        if gt_vertices.ndim != 3 or tuple(gt_vertices.shape[1:]) != (778, 3):
            raise ValueError(
                f"gt_vertices must have shape [B, 778, 3], got "
                f"{tuple(gt_vertices.shape)}"
            )
        gt_wrist = gt_pose[:, self.root_joint_idx : self.root_joint_idx + 1, :]
        return self._make_root_relative(gt_pose), gt_vertices - gt_wrist

    def _compute_losses(self, batch, results, image_size):
        gt_pose_relative, gt_vertices_relative = self._mesh_supervision_targets(batch)
        joint_valid = self._joint_valid_mask(batch, batch["gt_pose"])
        vertex_valid = torch.isfinite(gt_vertices_relative).all(dim=-1)
        initial_2d_metrics = self._compute_initial_2d_metrics(
            results["initial_pose2d_normalized"],
            batch["gt_pose_2d"],
            image_size,
            joint_valid,
        )
        stage_predictions = results["all_stage_pose3d"]
        if len(stage_predictions) != self.num_refine_layers:
            raise ValueError(
                f"Expected {self.num_refine_layers} Transformer stage predictions, "
                f"got {len(stage_predictions)}"
            )
        for stage_idx, prediction in enumerate(stage_predictions, start=1):
            if prediction.shape != gt_pose_relative.shape:
                raise ValueError(
                    f"Stage {stage_idx} joint prediction must have shape "
                    f"{tuple(gt_pose_relative.shape)}, got {tuple(prediction.shape)}"
                )
        stage_weights = self._normalized_stage_weights(
            len(stage_predictions), stage_predictions[0]
        )
        compute_base_mpjpe = super()._compute_mpjpe_3d
        stage_joint_3d_losses = [
            compute_base_mpjpe(prediction, gt_pose_relative, joint_valid)
            for prediction in stage_predictions
        ]
        joint_3d_loss = (
            torch.stack(stage_joint_3d_losses) * stage_weights
        ).sum()

        gt_wrist = batch["gt_pose"][
            :, self.root_joint_idx : self.root_joint_idx + 1, :
        ].to(device=stage_predictions[0].device, dtype=stage_predictions[0].dtype)
        stage_joint_2d_losses = [
            self._compute_joint_2d_loss(
                prediction + gt_wrist,
                batch["gt_pose_2d"],
                batch["cam_k"],
                image_size,
                joint_valid,
            )
            for prediction in stage_predictions
        ]
        joint_2d_loss = (
            torch.stack(stage_joint_2d_losses) * stage_weights
        ).sum()

        vertex_loss = self._masked_l1_loss(
            results["pred_vertices"], gt_vertices_relative, vertex_valid
        )
        total_loss = (
            self.joint_3d_loss_weight * joint_3d_loss
            + self.joint_2d_loss_weight * joint_2d_loss
            + self.initial_2d_loss_weight * initial_2d_metrics["loss"]
            + self.vertices_loss_weight * vertex_loss
        )
        return {
            "joint_2d_loss": joint_2d_loss,
            "joint_3d_loss": joint_3d_loss,
            "main_joint_3d_loss": stage_joint_3d_losses[-1],
            "vertex_loss": vertex_loss,
            "total_loss": total_loss,
            "stage_joint_2d_losses": stage_joint_2d_losses,
            "stage_joint_3d_losses": stage_joint_3d_losses,
            "stage_weights": stage_weights,
            "initial_2d_metrics": initial_2d_metrics,
            "gt_pose_relative": gt_pose_relative,
            "gt_vertices_relative": gt_vertices_relative,
            "joint_valid": joint_valid,
            "vertex_valid": vertex_valid,
        }

    def _log_losses(self, prefix, losses, batch_size):
        self.log(
            f"{prefix}_joint_2d_loss", losses["joint_2d_loss"], prog_bar=False
        )
        self.log(
            f"{prefix}_joint_3d_loss", losses["joint_3d_loss"], prog_bar=False
        )
        self.log(
            f"{prefix}_main_joint_3d_loss",
            losses["main_joint_3d_loss"],
            prog_bar=prefix != "test",
        )
        self.log(f"{prefix}_vertex_loss", losses["vertex_loss"], prog_bar=False)
        self.log(f"{prefix}_total_loss", losses["total_loss"], prog_bar=True)
        self._log_initial_2d_metrics(
            prefix, losses["initial_2d_metrics"], batch_size=batch_size
        )
        self.log(
            f"{prefix}_initial_2d_weighted_ratio",
            self.initial_2d_loss_weight
            * losses["initial_2d_metrics"]["loss"].detach()
            / losses["joint_3d_loss"].detach().clamp_min(1e-8),
            prog_bar=False,
        )
        self.log(
            f"{prefix}_joint_2d_weighted_ratio",
            self.joint_2d_loss_weight * losses["joint_2d_loss"].detach()
            / losses["joint_3d_loss"].detach().clamp_min(1e-8),
            prog_bar=False,
        )
        for stage_idx, (loss_2d, loss_3d) in enumerate(
            zip(
                losses["stage_joint_2d_losses"],
                losses["stage_joint_3d_losses"],
            ),
            start=1,
        ):
            suffix = "_epoch" if prefix == "train" else ""
            log_kwargs = {
                "on_step": False,
                "on_epoch": True,
                "prog_bar": False,
                "batch_size": batch_size,
            }
            self.log(
                f"{prefix}_stage{stage_idx}_joint_2d_loss{suffix}",
                loss_2d,
                **log_kwargs,
            )
            self.log(
                f"{prefix}_stage{stage_idx}_joint_3d_loss{suffix}",
                loss_3d,
                **log_kwargs,
            )

    def _compute_prediction_metrics(self, results, losses):
        metrics = {
            "mpjpe_3d": super()._compute_mpjpe_3d(
                results["pred_mesh_joints"],
                losses["gt_pose_relative"],
                losses["joint_valid"],
            ),
            "pa_mpjpe_3d": self._compute_pa_mpjpe_3d(
                results["pred_mesh_joints"], losses["gt_pose_relative"]
            ),
            "mpvpe": self._compute_mpvpe(
                results["pred_vertices"],
                losses["gt_vertices_relative"],
                losses["vertex_valid"],
            ),
            "pa_mpvpe": self._compute_pa_mpvpe(
                results["pred_vertices"], losses["gt_vertices_relative"]
            ),
        }
        return metrics

    def _log_prediction_metrics(self, prefix, metrics, batch_size):
        is_train = prefix == "train"
        for name in ("mpjpe_3d", "pa_mpjpe_3d", "mpvpe", "pa_mpvpe"):
            if is_train:
                # Keep the existing per-step joint metrics in the detailed log.
                if name in ("mpjpe_3d", "pa_mpjpe_3d"):
                    self.log(
                        f"train_{name}",
                        metrics[name],
                        on_step=True,
                        on_epoch=False,
                        prog_bar=name == "mpjpe_3d",
                        batch_size=batch_size,
                    )
                self.log(
                    f"train_{name}_epoch",
                    metrics[name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    f"{prefix}_{name}",
                    metrics[name],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=name in ("mpjpe_3d", "mpvpe"),
                    batch_size=batch_size,
                )

    def training_step(self, batch, batch_idx):
        imgs = batch["img"]
        results = self(
            imgs, imgs, cam_k=batch["cam_k"], root_3d=batch["origin_3d"]
        )
        losses = self._compute_losses(batch, results, imgs.shape[-2:])
        self.log("train_loss", losses["total_loss"], prog_bar=True)
        self._log_losses("train", losses, batch_size=imgs.shape[0])
        with torch.no_grad():
            prediction_metrics = self._compute_prediction_metrics(results, losses)
            self._log_prediction_metrics(
                "train", prediction_metrics, batch_size=imgs.shape[0]
            )
        return losses["total_loss"]

    def validation_step(self, batch, batch_idx):
        imgs = batch["img"]
        results = self(
            imgs, imgs, cam_k=batch["cam_k"], root_3d=batch["origin_3d"]
        )
        losses = self._compute_losses(batch, results, imgs.shape[-2:])
        pred_mesh_joints = results["pred_mesh_joints"]
        prediction_metrics = self._compute_prediction_metrics(results, losses)

        self.log("val_loss", losses["total_loss"], prog_bar=True)
        self._log_losses("val", losses, batch_size=imgs.shape[0])
        self._log_prediction_metrics(
            "val", prediction_metrics, batch_size=imgs.shape[0]
        )
        self._log_scale_diagnostics(
            "val",
            pred_mesh_joints,
            losses["gt_pose_relative"],
            losses["joint_valid"],
        )

    def test_step(self, batch, batch_idx):
        imgs = batch["img"]
        results = self(
            imgs, imgs, cam_k=batch["cam_k"], root_3d=batch["origin_3d"]
        )
        losses = self._compute_losses(batch, results, imgs.shape[-2:])
        pred_mesh_joints = results["pred_mesh_joints"]
        prediction_metrics = self._compute_prediction_metrics(results, losses)
        test_root_rigid_mpjpe_3d = self._compute_root_rigid_mpjpe_3d(
            pred_mesh_joints, losses["gt_pose_relative"]
        )

        self.log("test_loss", losses["total_loss"], prog_bar=True)
        self._log_losses("test", losses, batch_size=imgs.shape[0])
        self._log_prediction_metrics(
            "test", prediction_metrics, batch_size=imgs.shape[0]
        )
        self.log("test_root_rigid_mpjpe_3d", test_root_rigid_mpjpe_3d, prog_bar=True)
        self._log_scale_diagnostics(
            "test",
            pred_mesh_joints,
            losses["gt_pose_relative"],
            losses["joint_valid"],
        )
