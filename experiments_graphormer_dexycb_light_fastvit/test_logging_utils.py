import csv
import tempfile
import unittest
from pathlib import Path

from experiments_graphormer_dexycb_light_fastvit.logging_utils import (
    build_csv_loggers,
    is_summary_metric,
)


class DualCSVLoggerTest(unittest.TestCase):
    def test_summary_metric_filter(self):
        included = {
            "epoch",
            "train_mpjpe_3d_epoch",
            "train_pa_mpjpe_3d_epoch",
            "train_stage1_joint_3d_loss_epoch",
            "train_stage3_joint_3d_loss_epoch",
            "train_mpvpe_epoch",
            "train_pa_mpvpe_epoch",
            "val_mpjpe_3d",
            "val_pa_mpjpe_3d",
            "val_stage1_joint_3d_loss",
            "val_stage3_joint_3d_loss",
            "val_mpvpe",
            "val_pa_mpvpe",
            "test_mpvpe",
            "test_pa_mpvpe",
        }
        excluded = {
            "train_loss",
            "train_mpjpe_3d",
            "val_loss",
            "val_n_mpjpe_3d",
            "val_bone_length_error",
            "test_root_rigid_mpjpe_3d",
        }
        self.assertTrue(all(is_summary_metric(name) for name in included))
        self.assertFalse(any(is_summary_metric(name) for name in excluded))

    def test_loggers_share_version_and_summary_csv_is_filtered(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            detailed, summary = build_csv_loggers(tmp_dir)
            self.assertEqual(detailed.version, summary.version)

            metrics = {
                "epoch": 3,
                "val_loss": 1.25,
                "val_mpjpe_3d": 0.02,
                "val_pa_mpjpe_3d": 0.01,
                "val_stage1_joint_3d_loss": 0.03,
                "val_mpvpe": 0.025,
                "val_pa_mpvpe": 0.012,
            }
            detailed.log_metrics(metrics, step=7)
            summary.log_metrics(metrics, step=7)
            detailed.finalize("success")
            summary.finalize("success")

            with (Path(summary.log_dir) / "metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                row = next(csv.DictReader(handle))

            self.assertEqual(
                set(row),
                {
                    "epoch",
                    "step",
                    "val_mpjpe_3d",
                    "val_pa_mpjpe_3d",
                    "val_stage1_joint_3d_loss",
                    "val_mpvpe",
                    "val_pa_mpvpe",
                },
            )
            self.assertNotIn("val_loss", row)

            next_detailed, next_summary = build_csv_loggers(tmp_dir)
            self.assertEqual(next_detailed.version, detailed.version + 1)
            self.assertEqual(next_detailed.version, next_summary.version)


if __name__ == "__main__":
    unittest.main()
