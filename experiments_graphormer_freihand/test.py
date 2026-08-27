import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_graphormer_freihand.config_utils import (
    build_dataloader,
    build_datasets,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_freihand.lightning_module import FreiHANDPoseLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Test Graphormer on FreiHAND.")
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "freihand_graphormer.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        "--checkpoint",
        dest="ckpt_path",
        default=None,
        help="Checkpoint path for testing. Falls back to model.ckpt_path in config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = normalize_config_paths(load_config(config_path))

    ckpt_path = args.ckpt_path or config["model"].get("ckpt_path")
    if not ckpt_path:
        raise ValueError("未提供 --ckpt-path/--checkpoint，且 config['model']['ckpt_path'] 为空。")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    pl.seed_everything(config["seed"])

    print(f"正在加载 FreiHAND 测试数据: {config['data']['root']}")
    datasets = build_datasets(config)
    test_dataset = datasets["test"]
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
    )

    print(f"测试集加载完成: {len(test_dataset)}")

    system = FreiHANDPoseLightningModule(
        lr=config["training"]["learning_rate"],
        local_model_dir=config["model"]["local_model_dir"],
        num_joints=21,
    )

    trainer = pl.Trainer(
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", 1),
        precision=config["training"].get("precision", 32),
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        default_root_dir=config["output"]["default_root_dir"],
    )
    trainer.test(model=system, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
