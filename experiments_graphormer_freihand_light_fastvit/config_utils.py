import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader
import yaml


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_graphormer_freihand_light_fastvit.dataset import FreiHANDExperimentDataset


def resolve_path(root_dir, path_value):
    if path_value in (None, ""):
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((root_dir / path).resolve())


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_config_paths(config, repo_root=None):
    repo_root = Path(repo_root or ROOT_DIR)
    config["data"]["root"] = resolve_path(repo_root, config["data"]["root"])
    config["model"]["vitpose_config_path"] = resolve_path(
        repo_root, config["model"]["vitpose_config_path"]
    )
    config["model"]["vitpose_checkpoint_path"] = resolve_path(
        repo_root, config["model"]["vitpose_checkpoint_path"]
    )
    config["model"]["ckpt_path"] = resolve_path(repo_root, config["model"].get("ckpt_path"))
    config["output"]["checkpoint_dir"] = resolve_path(repo_root, config["output"]["checkpoint_dir"])
    config["output"]["default_root_dir"] = resolve_path(repo_root, config["output"]["default_root_dir"])
    return config


def build_datasets(config):
    data_cfg = config["data"]
    common_kwargs = {
        "data_root": data_cfg["root"],
        "img_size": data_cfg["image_size"],
        "use_annotation_uv": data_cfg.get("use_annotation_uv", True),
    }
    train_kwargs = {
        **common_kwargs,
        "augmentation": data_cfg.get("augmentation", {}),
        "grouped_background_sampling": data_cfg.get(
            "grouped_background_sampling", False
        ),
        "background_variant_weights": data_cfg.get("background_variant_weights"),
    }
    return {
        "train": FreiHANDExperimentDataset(split="train", **train_kwargs),
        "val": FreiHANDExperimentDataset(split=data_cfg.get("val_split", "evaluation"), **common_kwargs),
        "test": FreiHANDExperimentDataset(split=data_cfg.get("test_split", "test"), **common_kwargs),
    }


def build_dataloader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    pin_memory=False,
    prefetch_factor=False,
):
    """Create a DataLoader with a CUDA-compatible memory-pinning setting.

    ``pin_memory`` is deliberately configurable rather than hard-coded.  Some
    PyTorch/CUDA combinations fail in the background pin-memory thread before
    the first validation batch is evaluated. ``prefetch_factor=False`` keeps
    PyTorch's default; a positive integer overrides it for multi-worker loaders.
    """
    if prefetch_factor is not False and (
        isinstance(prefetch_factor, bool)
        or not isinstance(prefetch_factor, int)
        or prefetch_factor <= 0
    ):
        raise ValueError("prefetch_factor must be false or a positive integer")

    persistent_workers = bool(num_workers and num_workers > 0)
    loader_kwargs = {}
    if persistent_workers and prefetch_factor is not False:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        **loader_kwargs,
    )


def ensure_dir(path_value):
    if path_value:
        os.makedirs(path_value, exist_ok=True)
