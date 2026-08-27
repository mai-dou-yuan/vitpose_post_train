import unittest

import torch

from experiments_graphormer_freihand_light_fastvit.pl_system_v6_graphormer import (
    PoseRefinementLayer,
    project_camera_joints_to_image,
    refinement_attention_mode,
)


class LocalCrossAttentionTest(unittest.TestCase):
    def test_attention_schedule_repeats(self):
        self.assertEqual(
            [refinement_attention_mode(i) for i in range(8)],
            ["local", "local", "full", "local", "local", "full", "local", "local"],
        )

    def test_projection_uses_camera_coordinates(self):
        joints = torch.tensor([[[0.1, 0.2, 1.0]]], requires_grad=True)
        cam_k = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]])
        points = project_camera_joints_to_image(joints, cam_k, (80, 100))
        torch.testing.assert_close(points, torch.tensor([[[60.0, 60.0]]]))
        points.sum().backward()
        self.assertIsNotNone(joints.grad)
        self.assertTrue(torch.isfinite(joints.grad).all())

    def test_local_shape_sample_count_and_backward(self):
        torch.manual_seed(0)
        layer = PoseRefinementLayer(
            d_model=32,
            n_head=4,
            dim_feedforward=64,
            dropout=0.0,
            attention_mode="local",
            local_grid_size=5,
            local_grid_radius=2.0,
        )
        tgt = torch.randn(2, 3, 32, requires_grad=True)
        memory = torch.randn(2, 32, 8, 6, requires_grad=True)
        memory_pos = torch.randn(2, 32, 8, 6)
        points = torch.tensor(
            [[[20.0, 20.0], [40.0, 30.0], [60.0, 50.0]]] * 2,
            requires_grad=True,
        )
        captured = {}

        def capture_kv(_module, args, kwargs):
            captured["query"] = kwargs["query"].shape
            captured["key"] = kwargs["key"].shape

        hook = layer.cross_attention.register_forward_pre_hook(
            capture_kv, with_kwargs=True
        )
        output = layer(
            tgt,
            memory,
            memory_pos=memory_pos,
            reference_points=points,
            image_size=(80, 100),
        )
        hook.remove()

        self.assertEqual(output.shape, tgt.shape)
        self.assertEqual(captured["query"], torch.Size([6, 1, 32]))
        self.assertEqual(captured["key"], torch.Size([6, 25, 32]))
        output.square().mean().backward()
        for tensor in (tgt, memory, points):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

        out_of_bounds = torch.tensor(
            [[[-1000.0, 1000.0], [float("nan"), 30.0], [50.0, float("inf")]]] * 2
        )
        safe_output = layer(
            tgt.detach(),
            memory.detach(),
            reference_points=out_of_bounds,
            image_size=(80, 100),
        )
        self.assertTrue(torch.isfinite(safe_output).all())

    def test_full_path_still_matches_flattened_memory(self):
        torch.manual_seed(1)
        layer = PoseRefinementLayer(
            d_model=32,
            n_head=4,
            dim_feedforward=64,
            dropout=0.0,
            attention_mode="full",
        ).eval()
        tgt = torch.randn(2, 3, 32)
        memory = torch.randn(2, 32, 4, 5)
        memory_pos = torch.randn(2, 32, 4, 5)
        output_map = layer(tgt, memory, memory_pos=memory_pos)
        output_tokens = layer(
            tgt,
            memory.permute(0, 2, 3, 1).flatten(1, 2),
            memory_pos=memory_pos.permute(0, 2, 3, 1).flatten(1, 2),
        )
        torch.testing.assert_close(output_map, output_tokens)


if __name__ == "__main__":
    unittest.main()
