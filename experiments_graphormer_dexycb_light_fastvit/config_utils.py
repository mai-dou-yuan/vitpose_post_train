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

from experiments_graphormer_dexycb_light_fastvit.dataset import (
    DexYCBLightFastViTDataset,
)


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
    config["data"]["mano_root"] = resolve_path(
        repo_root, config["data"]["mano_root"]
    )
    config["data"]["index_cache_dir"] = resolve_path(
        repo_root,
        config["data"].get("index_cache_dir")
        or config["data"].get("visibility_cache_dir"),
    )
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


def build_dataset(config, split):
    data_cfg = config["data"]
    dataset_kwargs = {
        "data_root": data_cfg["root"],
        "mano_root": data_cfg["mano_root"],
        "setup": data_cfg.get("setup", "s0"),
        "img_size": data_cfg["image_size"],
        "use_annotation_uv": data_cfg.get("use_annotation_uv", True),
        "crop_padding": data_cfg.get("crop_padding", 1.5),
        "min_crop_size": data_cfg.get("min_crop_size", 32.0),
        "min_joints_in_frame": data_cfg.get("min_joints_in_frame", 12),
        "require_hand_segmentation": data_cfg.get(
            "require_hand_segmentation", True
        ),
        "require_hand_in_crop": data_cfg.get("require_hand_in_crop", True),
        "visibility_filter_workers": data_cfg.get(
            "visibility_filter_workers", 16
        ),
        "index_cache_dir": data_cfg.get("index_cache_dir"),
    }
    if split == "train":
        dataset_kwargs["augmentation"] = data_cfg.get("augmentation", {})
    return DexYCBLightFastViTDataset(split=split, **dataset_kwargs)


def build_datasets(config):
    data_cfg = config["data"]
    return {
        "train": build_dataset(config, "train"),
        "val": build_dataset(config, data_cfg.get("val_split", "val")),
        "test": build_dataset(config, data_cfg.get("test_split", "test")),
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
