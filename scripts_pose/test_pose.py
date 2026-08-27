import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset import Unrealego3DPoseDataset
from pl_system_v6_graphormer import PoseLightningModule


DEFAULT_TEST_USER_IDS = [1, 2, 3]


def parse_args():
    parser = argparse.ArgumentParser(description="Test pose model.")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Checkpoint path for testing. Falls back to model.ckpt_path in config.",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config = load_config(args.config)

    ckpt_path = args.ckpt_path or config["model"].get("ckpt_path")
    if not ckpt_path:
        raise ValueError("未提供 --ckpt-path，且 config['model']['ckpt_path'] 为空。")

    data_root = config["data"]["root"]
    image_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    local_model_dir = config["model"]["local_model_dir"]

    pl.seed_everything(config["seed"])

    print(f"正在加载测试数据: {data_root}")
    test_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=False,
        target_user_ids=DEFAULT_TEST_USER_IDS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"测试集加载完成: {len(test_dataset)}")

    system = PoseLightningModule(
        lr=config["training"]["learning_rate"],
        local_model_dir=local_model_dir,
        num_joints=21,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        log_every_n_steps=20,
    )
    trainer.test(model=system, dataloaders=test_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
