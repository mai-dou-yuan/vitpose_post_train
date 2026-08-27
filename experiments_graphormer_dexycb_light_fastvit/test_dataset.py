import unittest
from pathlib import Path
import random

import numpy as np
import torch

from experiments_graphormer_dexycb_light_fastvit.dataset import (
    FORBIDDEN_DATA_PARTS,
    DexYCBLightFastViTDataset,
    official_dexycb_selection,
)
from mano_joints_package import mesh_to_joints


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT_DIR / "dexycb"
MANO_ROOT = ROOT_DIR / "dexycb"
CACHE_ROOT = (
    ROOT_DIR
    / "experiments_graphormer_dexycb_light_fastvit"
    / "cache"
    / "full_dexycb_s0"
)


class OfficialDexYCBSelectionTest(unittest.TestCase):
    def test_s0_matches_nvlabs_protocol(self):
        train = official_dexycb_selection("s0", "train")
        val = official_dexycb_selection("s0", "val")
        test = official_dexycb_selection("s0", "test")
        self.assertEqual(len(train["subjects"]), 10)
        self.assertEqual(len(train["serials"]), 8)
        self.assertEqual(train["sequence_indices"], tuple(i for i in range(100) if i % 5 != 4))
        self.assertEqual(val["subjects"], train["subjects"][:2])
        self.assertEqual(test["subjects"], train["subjects"][2:])
        self.assertEqual(val["sequence_indices"], tuple(i for i in range(100) if i % 5 == 4))

    def test_s1_s2_s3_are_configurable_and_match_official_sizes(self):
        expected = {
            "s1": ((7, 8, 100), (1, 8, 100), (2, 8, 100)),
            "s2": ((10, 6, 100), (10, 1, 100), (10, 1, 100)),
            "s3": ((10, 8, 75), (10, 8, 10), (10, 8, 15)),
        }
        for setup, setup_expected in expected.items():
            for split, split_expected in zip(("train", "val", "test"), setup_expected):
                selection = official_dexycb_selection(setup, split)
                actual = tuple(
                    len(selection[name])
                    for name in ("subjects", "serials", "sequence_indices")
                )
                self.assertEqual(actual, split_expected, (setup, split))


@unittest.skipUnless(DATA_ROOT.is_dir(), "local full DexYCB is unavailable")
class DexYCBLightFastViTDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = DexYCBLightFastViTDataset(
            data_root=DATA_ROOT,
            mano_root=MANO_ROOT,
            split="test",
            setup="s0",
            img_size=224,
            index_cache_dir=CACHE_ROOT,
        )

    def test_bbox_outside_image_is_rejected_before_clipping(self):
        joints_3d = np.tile([0.0, 0.0, 0.5], (21, 1)).astype(np.float32)
        projected_2d = np.stack(
            [np.linspace(700.0, 740.0, 21), np.linspace(520.0, 560.0, 21)],
            axis=1,
        ).astype(np.float32)

        with self.assertRaisesRegex(ValueError, "does not intersect"):
            self.dataset._bbox_from_projected_joints(
                joints_3d, projected_2d, (480, 640, 3)
            )

        raw_bbox = self.dataset._bbox_from_projected_joints(
            joints_3d,
            projected_2d,
            (480, 640, 3),
            require_image_intersection=False,
        )
        np.testing.assert_allclose(raw_bbox, [700.0, 520.0, 740.0, 560.0])

    def test_sample_shapes_units_validity_and_projection(self):
        sample = self.dataset[0]
        expected_shapes = {
            "img": (3, 224, 224),
            "gt_pose": (21, 3),
            "gt_pose_2d": (21, 2),
            "gt_vertices": (778, 3),
            "origin_3d": (3,),
            "cam_k": (3, 3),
            "joint_valid": (21,),
            "vertex_valid": (778,),
        }
        for name, shape in expected_shapes.items():
            self.assertEqual(tuple(sample[name].shape), shape)
        self.assertEqual(sample["joint_valid"].dtype, torch.bool)
        self.assertEqual(sample["vertex_valid"].dtype, torch.bool)
        self.assertGreaterEqual(int(sample["joint_valid"].sum()), 12)
        self.assertTrue(bool(sample["vertex_valid"].any()))
        self.assertTrue(bool(sample["is_right"]))
        self.assertGreater(sample["gt_pose"][:, 2].median().item(), 0.1)
        self.assertLess(sample["gt_pose"][:, 2].median().item(), 2.0)

        homogeneous = sample["gt_pose"] @ sample["cam_k"].T
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        projection_epe = torch.linalg.vector_norm(
            projected - sample["gt_pose_2d"], dim=-1
        )
        self.assertLess(projection_epe[sample["joint_valid"]].max().item(), 1e-3)
        self.assertFalse(
            bool(FORBIDDEN_DATA_PARTS.intersection(Path(sample["dataset_idx"]).parts))
        )
        visibility = self.dataset._visibility_from_annotation(self.dataset.samples[0])
        self.assertTrue(visibility["hand_in_crop"])

    def test_mano_pca_vertex_reconstruction_matches_dexycb_wrist(self):
        annotation = self.dataset._load_annotation(self.dataset.samples[0])
        gt_joints, gt_vertices = annotation[:2]
        mesh_joints = mesh_to_joints(torch.from_numpy(gt_vertices)[None])[0].numpy()
        root_error = np.linalg.norm(mesh_joints[0] - gt_joints[0])
        relative_error = np.linalg.norm(
            (mesh_joints - mesh_joints[0]) - (gt_joints - gt_joints[0]), axis=-1
        ).mean()

        # DexYCB's joints come from the posed MANO skeleton while the model's
        # metric joints are regressed from vertices, so a few millimetres are
        # expected; a wrong pose representation causes centimetre-scale error.
        self.assertLess(root_error, 0.002)
        self.assertLess(relative_error, 0.006)

    def test_training_geometry_keeps_2d_3d_and_intrinsics_synchronized(self):
        random.seed(3)
        np.random.seed(3)
        dataset = DexYCBLightFastViTDataset(
            data_root=DATA_ROOT,
            mano_root=MANO_ROOT,
            split="train",
            setup="s0",
            img_size=224,
            index_cache_dir=CACHE_ROOT,
            augmentation={
                "enabled": True,
                "center_jitter": 0.07,
                "scale_range": [0.85, 1.20],
                "rotation_probability": 1.0,
                "max_rotation_degrees": 30.0,
                "color_jitter_probability": 0.0,
                "degradation_probability": 0.0,
                "occlusion_probability": 0.0,
            },
        )
        sample = dataset[0]
        homogeneous = sample["gt_pose"] @ sample["cam_k"].T
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        projection_epe = torch.linalg.vector_norm(
            projected - sample["gt_pose_2d"], dim=-1
        )
        self.assertLess(projection_epe[sample["joint_valid"]].max().item(), 1e-3)


if __name__ == "__main__":
    unittest.main()
