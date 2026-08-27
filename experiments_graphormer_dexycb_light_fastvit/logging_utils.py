import re
from pathlib import Path

from pytorch_lightning.loggers import CSVLogger


DETAILED_LOG_NAME = "lightning_logs"
SUMMARY_LOG_NAME = "lightning_logs_summary"

_FINAL_SUMMARY_METRIC = re.compile(
    r"^(?:val|test)_(?:pa_)?(?:mpjpe_3d|mpvpe)$"
)
_STAGE_SUMMARY_METRIC = re.compile(
    r"^(?:val|test)_stage\d+_joint_3d_loss$"
)
_TRAIN_FINAL_SUMMARY_METRIC = re.compile(
    r"^train_(?:pa_)?(?:mpjpe_3d|mpvpe)_epoch$"
)
_TRAIN_STAGE_SUMMARY_METRIC = re.compile(
    r"^train_stage\d+_joint_3d_loss_epoch$"
)


def is_summary_metric(name):
    """Return whether a Lightning metric belongs in the compact CSV log."""
    return bool(
        name == "epoch"
        or _FINAL_SUMMARY_METRIC.fullmatch(name)
        or _STAGE_SUMMARY_METRIC.fullmatch(name)
        or _TRAIN_FINAL_SUMMARY_METRIC.fullmatch(name)
        or _TRAIN_STAGE_SUMMARY_METRIC.fullmatch(name)
    )


class SummaryCSVLogger(CSVLogger):
    """CSV logger that retains final metrics and per-stage 3D supervision."""

    def log_metrics(self, metrics, step=None):
        summary_metrics = {
            name: value for name, value in metrics.items() if is_summary_metric(name)
        }
        if summary_metrics:
            super().log_metrics(summary_metrics, step=step)


def _next_shared_version(save_dir, log_names):
    save_dir = Path(save_dir)
    versions = []
    for log_name in log_names:
        log_root = save_dir / log_name
        if not log_root.is_dir():
            continue
        for candidate in log_root.iterdir():
            if not candidate.is_dir() or not candidate.name.startswith("version_"):
                continue
            suffix = candidate.name[len("version_") :]
            if suffix.isdigit():
                versions.append(int(suffix))
    return max(versions, default=-1) + 1


def build_csv_loggers(default_root_dir):
    """Create paired detailed and compact CSV loggers with one version id."""
    version = _next_shared_version(
        default_root_dir, (DETAILED_LOG_NAME, SUMMARY_LOG_NAME)
    )
    detailed_logger = CSVLogger(
        save_dir=default_root_dir,
        name=DETAILED_LOG_NAME,
        version=version,
    )
    summary_logger = SummaryCSVLogger(
        save_dir=default_root_dir,
        name=SUMMARY_LOG_NAME,
        version=version,
    )
    return [detailed_logger, summary_logger]
