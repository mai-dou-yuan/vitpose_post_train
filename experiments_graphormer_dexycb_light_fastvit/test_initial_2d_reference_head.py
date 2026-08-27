import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments_graphormer_dexycb_light_fastvit import pl_system_v6_graphormer as graphormer
from experiments_graphormer_dexycb_light_fastvit.checkpoint_utils import (
    is_legacy_initial_2d_checkpoint,
    load_pose_checkpoint_weights,
)
from experiments_graphormer_dexycb_light_fastvit.lightning_module import (
    DexYCBPoseLightningModule,
)


class DummyViTPoseBackbone(nn.Module):
    model_name = "dummy_vitpose"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.feature_dim = 32
        self.projection = nn.Conv2d(3, self.feature_dim, kernel_size=1)

    def forward(self, image):
        feature = self.projection(image)
        return F.interpolate(
            feature, size=(16, 12), mode="bilinear", align_corners=False
        )


def build_model(lightning_wrapper=False, **model_kwargs):
    model_class = DexYCBPoseLightningModule if lightning_wrapper else graphormer.PoseLightningModule
    with mock.patch.object(
        graphormer, "ViTPosePlusBBackbone", DummyViTPoseBackbone
    ):
        return model_class(
            lr=1e-4,
            vitpose_config_path="unused.py",
            vitpose_checkpoint_path="unused.pth",
            num_joints=21,
            upsample_dim=32,
            num_refine_layers=3,
            use_gradient_checkpointing=False,
            local_grid_size=3,
            local_grid_radius=1.0,
            **model_kwargs,
        )


class Initial2DCoordinateHeadTest(unittest.TestCase):
    def test_lr_warmup_epochs_is_configurable_and_can_be_disabled(self):
        warmup_model = build_model(lr_warmup_epochs=3)
        warmup_model._trainer = SimpleNamespace(max_epochs=10)
        warmup_config = warmup_model.configure_optimizers()
        self.assertEqual(warmup_model.lr_warmup_epochs, 3)
        self.assertIsInstance(
            warmup_config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.SequentialLR,
        )
        self.assertAlmostEqual(
            warmup_config["optimizer"].param_groups[0]["lr"], 1e-7
        )

        no_warmup_model = build_model(lr_warmup_epochs=0)
        no_warmup_model._trainer = SimpleNamespace(max_epochs=10)
        no_warmup_config = no_warmup_model.configure_optimizers()
        self.assertIsInstance(
            no_warmup_config["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )
        self.assertAlmostEqual(
            no_warmup_config["optimizer"].param_groups[0]["lr"], 1e-4
        )

    def test_shape_range_and_backward(self):
        torch.manual_seed(0)
        head = graphormer.Initial2DCoordinateHead(
            in_channels=256,
            num_joints=21,
            bottleneck_channels=16,
            pooled_size=(4, 3),
            hidden_dim=32,
        )
        feature_map = torch.randn(2, 256, 16, 12, requires_grad=True)
        coordinates = head(feature_map)

        self.assertEqual(coordinates.shape, (2, 21, 2))
        self.assertTrue(torch.all(coordinates >= 0.0))
        self.assertTrue(torch.all(coordinates <= 1.0))
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 11866)
        self.assertFalse(any("heatmap" in name for name, _ in head.named_modules()))

        coordinates.square().mean().backward()
        self.assertIsNotNone(feature_map.grad)
        self.assertTrue(torch.isfinite(feature_map.grad).all())
        self.assertGreater(feature_map.grad.abs().sum().item(), 0.0)

    def test_crop_pixel_conversion_preserves_xy_axis_order(self):
        coordinates = torch.tensor([[[0.75, 0.25]]])
        pixels = graphormer.normalized_crop_to_pixel(coordinates, (224, 224))
        torch.testing.assert_close(
            pixels, torch.tensor([[[167.25, 55.75]]]), atol=1e-5, rtol=0.0
        )

    def test_stage_zero_one_two_reference_flow_and_full_backward(self):
        torch.manual_seed(1)
        model = build_model().eval()
        captured_references = []
        pose_head_call_count = 0

        def capture_reference(_module, _args, kwargs):
            captured_references.append(kwargs.get("reference_points"))

        def count_pose_head(_module, _args, _output):
            nonlocal pose_head_call_count
            pose_head_call_count += 1

        hooks = [
            layer.register_forward_pre_hook(capture_reference, with_kwargs=True)
            for layer in model.layers_ca
        ]
        hooks.append(model.pose_3d_head_PR.register_forward_hook(count_pose_head))

        images = torch.rand(1, 3, 224, 224, requires_grad=True)
        cam_k = torch.tensor(
            [[[500.0, 0.0, 111.5], [0.0, 500.0, 111.5], [0.0, 0.0, 1.0]]]
        )
        root_3d = torch.tensor([[0.0, 0.0, 0.6]])
        results = model(images, cam_k=cam_k, root_3d=root_3d)
        for hook in hooks:
            hook.remove()

        self.assertEqual(pose_head_call_count, 3)
        self.assertEqual(len(results["all_stage_pose3d"]), 3)
        self.assertEqual(len(results["all_stage_vertices"]), 3)
        self.assertEqual(len(results["all_stage_mesh_joints"]), 3)
        self.assertEqual(results["pose3d"].shape, (1, 21, 3))
        self.assertEqual(results["pred_vertices"].shape, (1, 778, 3))
        torch.testing.assert_close(results["pose3d"], results["pred_mesh_joints"])
        self.assertEqual(results["initial_pose2d_normalized"].shape, (1, 21, 2))
        self.assertNotIn("initial_heatmaps", results)

        # Stage 0 uses the direct 2D prediction converted to 224x224 crop pixels.
        torch.testing.assert_close(captured_references[0], results["initial_pose2d"])
        torch.testing.assert_close(
            captured_references[0],
            results["initial_pose2d_normalized"] * 223.0,
        )

        # Stage 1 still projects Stage 0's root-relative 3D prediction.
        expected_stage1_reference = graphormer.project_camera_joints_to_image(
            results["all_stage_pose3d"][0] + root_3d[:, None, :],
            cam_k,
            (224, 224),
        )
        torch.testing.assert_close(captured_references[1], expected_stage1_reference)

        # Stage 2 remains Full CA and receives no reference points.
        self.assertEqual(
            [layer.attention_mode for layer in model.layers_ca],
            ["local", "local", "full"],
        )
        self.assertIsNone(captured_references[2])

        loss = results["pose3d"].square().mean()
        loss = loss + results["initial_pose2d_normalized"].square().mean()
        loss.backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())
        head_gradients = [
            parameter.grad for parameter in model.initial_2d_head.parameters()
        ]
        self.assertTrue(all(gradient is not None for gradient in head_gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in head_gradients))

    def test_initial_loss_pixel_epe_and_log_names(self):
        model = build_model(lightning_wrapper=True)
        gt_pixels = torch.tensor(
            [[[0.0, 0.0], [111.5, 111.5], [223.0, 223.0]]]
        )
        pred_normalized = gt_pixels / 223.0
        metrics = model._compute_initial_2d_metrics(
            pred_normalized,
            gt_pixels,
            (224, 224),
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        torch.testing.assert_close(metrics["loss"], torch.tensor(0.0))
        torch.testing.assert_close(metrics["epe_px"], torch.tensor(0.0))
        self.assertEqual(set(metrics), {"loss", "epe_px"})

        model.log = mock.Mock()
        model._log_initial_2d_metrics("val", metrics, batch_size=1)
        logged_names = [call.args[0] for call in model.log.call_args_list]
        self.assertEqual(
            logged_names,
            [
                "val_initial_2d_loss",
                "val_initial_2d_epe_px",
            ],
        )

    def test_training_loss_uses_stage_joints_and_mesh_vertices_only(self):
        torch.manual_seed(2)
        model = build_model(lightning_wrapper=True)
        cam_k = torch.tensor(
            [[[500.0, 0.0, 111.5], [0.0, 500.0, 111.5], [0.0, 0.0, 1.0]]]
        )
        gt_pose = torch.randn(1, 21, 3) * 0.02
        gt_pose[..., 2] += 0.6
        gt_pose_2d = model._project_3d_to_2d(gt_pose, cam_k)
        gt_relative = model._make_root_relative(gt_pose)
        stage_predictions = [
            (gt_relative + offset).requires_grad_()
            for offset in (0.03, 0.02, 0.01)
        ]
        gt_vertices = gt_pose[:, :1] + torch.randn(1, 778, 3) * 0.02
        gt_vertices_relative = gt_vertices - gt_pose[:, :1]
        pred_mesh_joints = (gt_relative + 0.01).requires_grad_()
        pred_vertices = (gt_vertices_relative + 0.02).requires_grad_()
        gt_normalized = gt_pose_2d / 223.0
        initial_prediction = (gt_normalized + 0.02).clamp(0.0, 1.0).requires_grad_()
        results = {
            "pose3d": pred_mesh_joints,
            "pred_mesh_joints": pred_mesh_joints,
            "pred_vertices": pred_vertices,
            "all_stage_pose3d": stage_predictions,
            "initial_pose2d_normalized": initial_prediction,
        }
        model.forward = mock.Mock(return_value=results)
        model.log = mock.Mock()
        batch = {
            "img": torch.rand(1, 3, 224, 224),
            "gt_pose": gt_pose,
            "gt_vertices": gt_vertices,
            "gt_pose_2d": gt_pose_2d,
            "cam_k": cam_k,
            "origin_3d": gt_pose[:, 0],
        }

        expected_metrics = model._compute_initial_2d_metrics(
            initial_prediction,
            gt_pose_2d,
            (224, 224),
            torch.ones(1, 21, dtype=torch.bool),
        )
        loss = model.training_step(batch, batch_idx=0)
        stage_weights = model._normalized_stage_weights(3, stage_predictions[0])
        expected_stage_3d_losses = torch.stack(
            [
                graphormer.PoseLightningModule._compute_mpjpe_3d(
                    model, prediction, gt_relative
                )
                for prediction in stage_predictions
            ]
        )
        expected_joint_3d_loss = (expected_stage_3d_losses * stage_weights).sum()
        expected_stage_2d_losses = torch.stack(
            [
                model._compute_joint_2d_loss(
                    prediction + gt_pose[:, :1],
                    gt_pose_2d,
                    cam_k,
                    (224, 224),
                    torch.ones(1, 21, dtype=torch.bool),
                )
                for prediction in stage_predictions
            ]
        )
        expected_joint_2d_loss = (expected_stage_2d_losses * stage_weights).sum()
        expected_vertex_loss = F.l1_loss(pred_vertices, gt_vertices_relative)
        torch.testing.assert_close(
            loss,
            model.joint_3d_loss_weight * expected_joint_3d_loss
            + model.joint_2d_loss_weight * expected_joint_2d_loss
            + model.initial_2d_loss_weight * expected_metrics["loss"]
            + model.vertices_loss_weight * expected_vertex_loss,
            atol=1e-6,
            rtol=1e-5,
        )
        logged_names = {call.args[0] for call in model.log.call_args_list}
        self.assertTrue(
            {
                "train_loss",
                "train_joint_2d_loss",
                "train_joint_3d_loss",
                "train_main_joint_3d_loss",
                "train_vertex_loss",
                "train_total_loss",
                "train_initial_2d_loss",
                "train_initial_2d_epe_px",
                "train_initial_2d_weighted_ratio",
                "train_joint_2d_weighted_ratio",
                "train_mpvpe_epoch",
                "train_pa_mpvpe_epoch",
                "train_stage1_joint_2d_loss_epoch",
                "train_stage1_joint_3d_loss_epoch",
                "train_stage3_joint_2d_loss_epoch",
                "train_stage3_joint_3d_loss_epoch",
            }.issubset(logged_names)
        )
        self.assertTrue(
            {
                "train_initial_2d_epe_normalized",
                "train_initial_2d_pck",
                "train_initial_2d_saturation_ratio",
                "train_stage_0_mpjpe_3d_epoch",
            }.isdisjoint(logged_names)
        )
        loss.backward()
        self.assertIsNotNone(initial_prediction.grad)
        self.assertGreater(initial_prediction.grad.abs().sum().item(), 0.0)
        self.assertIsNone(pred_mesh_joints.grad)
        self.assertIsNotNone(pred_vertices.grad)
        self.assertTrue(
            all(prediction.grad is not None for prediction in stage_predictions)
        )
        self.assertTrue(
            all(
                prediction.grad.abs().sum().item() > 0.0
                for prediction in stage_predictions
            )
        )

    def test_joint_losses_ignore_invalid_joints(self):
        model = build_model(lightning_wrapper=True)
        gt_pose = torch.zeros(1, 21, 3)
        gt_pose[..., 2] = 0.6
        gt_relative = model._make_root_relative(gt_pose)
        cam_k = torch.tensor(
            [[[500.0, 0.0, 111.5], [0.0, 500.0, 111.5], [0.0, 0.0, 1.0]]]
        )
        gt_pose_2d = model._project_3d_to_2d(gt_pose, cam_k)
        valid = torch.ones(1, 21, dtype=torch.bool)
        valid[:, 5] = False
        stage_predictions = [gt_relative.clone() for _ in range(3)]
        for prediction in stage_predictions:
            prediction[:, 5] = 1000.0
        gt_vertices = gt_pose[:, :1].expand(-1, 778, -1).clone()
        pred_vertices = torch.zeros_like(gt_vertices)
        pred_vertices[:, 5] = 1000.0
        vertex_valid = torch.ones(1, 778, dtype=torch.bool)
        vertex_valid[:, 5] = False
        results = {
            "pred_mesh_joints": torch.full_like(gt_pose, 1000.0),
            "pred_vertices": pred_vertices,
            "all_stage_pose3d": stage_predictions,
            "initial_pose2d_normalized": gt_pose_2d / 223.0,
        }
        batch = {
            "gt_pose": gt_pose,
            "gt_vertices": gt_vertices,
            "gt_pose_2d": gt_pose_2d,
            "cam_k": cam_k,
            "joint_valid": valid,
            "vertex_valid": vertex_valid,
        }

        losses = model._compute_losses(batch, results, (224, 224))

        torch.testing.assert_close(losses["joint_3d_loss"], torch.tensor(0.0))
        torch.testing.assert_close(losses["joint_2d_loss"], torch.tensor(0.0))
        torch.testing.assert_close(losses["vertex_loss"], torch.tensor(0.0))

    def test_vertex_metrics_use_euclidean_error_and_similarity_alignment(self):
        model = build_model(lightning_wrapper=True)
        torch.manual_seed(3)
        gt_vertices = torch.randn(2, 778, 3) * 0.02
        translation = torch.tensor([0.01, -0.02, 0.03])
        translated_vertices = gt_vertices + translation

        expected_mpvpe = torch.linalg.vector_norm(translation)
        torch.testing.assert_close(
            model._compute_mpvpe(translated_vertices, gt_vertices),
            expected_mpvpe,
        )
        torch.testing.assert_close(
            model._compute_pa_mpvpe(translated_vertices, gt_vertices),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0.0,
        )

    def test_legacy_checkpoint_reports_only_new_head_missing_keys(self):
        model = build_model(lightning_wrapper=True)
        legacy_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("initial_2d_head.")
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "legacy.ckpt"
            torch.save({"state_dict": legacy_state}, checkpoint_path)
            report = load_pose_checkpoint_weights(model, checkpoint_path)

        self.assertTrue(is_legacy_initial_2d_checkpoint(report))
        self.assertTrue(report["missing_keys"])
        self.assertEqual(
            report["missing_keys"], report["initial_2d_head_missing_keys"]
        )
        self.assertFalse(report["unexpected_keys"])

    def test_pre_attention_checkpoint_reports_new_mesh_attention_keys(self):
        model = build_model(lightning_wrapper=True)
        previous_state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("mesh_regressor.cross_stage_attn.")
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "pre_attention.ckpt"
            torch.save({"state_dict": previous_state}, checkpoint_path)
            report = load_pose_checkpoint_weights(model, checkpoint_path)

        self.assertTrue(is_legacy_initial_2d_checkpoint(report))
        self.assertTrue(report["mesh_attention_missing_keys"])
        self.assertEqual(
            report["missing_keys"], report["mesh_attention_missing_keys"]
        )
        self.assertEqual(
            report["missing_keys"], report["architecture_missing_keys"]
        )
        self.assertFalse(report["unexpected_keys"])

    def test_legacy_detection_rejects_unexpected_keys(self):
        report = {
            "missing_keys": ["initial_2d_head.spatial_encoder.0.weight"],
            "initial_2d_head_missing_keys": [
                "initial_2d_head.spatial_encoder.0.weight"
            ],
            "unexpected_keys": ["unrelated.weight"],
        }
        self.assertFalse(is_legacy_initial_2d_checkpoint(report))


if __name__ == "__main__":
    unittest.main()
