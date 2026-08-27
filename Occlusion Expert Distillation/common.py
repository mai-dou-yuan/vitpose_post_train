import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml
from torch.utils.data import DataLoader


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset import Unrealego3DPoseDataset


DEFAULT_TRAIN_USER_IDS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14]
DEFAULT_VAL_USER_IDS = [4, 8, 13]
DEFAULT_TEST_USER_IDS = [4, 8, 13]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def build_dataset_splits(
    data_root: str,
    image_size: int,
    visibility_profile: Optional[Dict[str, float]] = None,
) -> Tuple[Unrealego3DPoseDataset, Unrealego3DPoseDataset, Unrealego3DPoseDataset]:
    train_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=True,
        target_user_ids=DEFAULT_TRAIN_USER_IDS,
        visibility_profile=visibility_profile,
    )
    val_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=False,
        target_user_ids=DEFAULT_VAL_USER_IDS,
        visibility_profile=visibility_profile,
    )
    test_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=False,
        target_user_ids=DEFAULT_TEST_USER_IDS,
        visibility_profile=visibility_profile,
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    *,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_all_dataloaders(config: Dict[str, Any]):
    data_cfg = config["data"]
    train_dataset, val_dataset, test_dataset = build_dataset_splits(
        data_root=data_cfg["root"],
        image_size=data_cfg["image_size"],
        visibility_profile=data_cfg.get("visibility_profile"),
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
        pin_memory=config["training"].get("pin_memory", False),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
        pin_memory=config["training"].get("pin_memory", False),
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
        pin_memory=config["training"].get("pin_memory", False),
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def apply_overrides(config: Dict[str, Any], overrides: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    for dotted_key, value in overrides:
        if value is None:
            continue
        node = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config
