import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_graphormer_dexycb.dataset import DexYCBSampledDataset
from experiments_graphormer_dexycb.lightning_module import DexYCBPoseLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Test Graphormer on DexYCB.")
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "dexycb_graphormer.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Checkpoint path for testing. Falls back to model.ckpt_path in config.",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    config = load_config(args.config)

    repo_root = ROOT_DIR
    config["data"]["root"] = resolve_path(repo_root, config["data"]["root"])
    config["model"]["local_model_dir"] = resolve_path(repo_root, config["model"]["local_model_dir"])
    config["model"]["ckpt_path"] = resolve_path(repo_root, config["model"].get("ckpt_path"))

    ckpt_path = args.ckpt_path or config["model"].get("ckpt_path")
    if not ckpt_path:
        raise ValueError("未提供 --ckpt-path，且 config['model']['ckpt_path'] 为空。")

    pl.seed_everything(config["seed"])

    print(f"正在加载 DexYCB 测试数据: {config['data']['root']}")
    test_dataset = DexYCBSampledDataset(
        data_root=config["data"]["root"],
        split="test",
        img_size=config["data"]["image_size"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    print(f"测试集加载完成: {len(test_dataset)}")

    system = DexYCBPoseLightningModule(
        lr=config["training"]["learning_rate"],
        local_model_dir=config["model"]["local_model_dir"],
        num_joints=21,
    )

    trainer = pl.Trainer(
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", 1),
        precision=config["training"].get("precision", 32),
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        default_root_dir=resolve_path(repo_root, config["output"]["default_root_dir"]),
    )
    trainer.test(model=system, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
