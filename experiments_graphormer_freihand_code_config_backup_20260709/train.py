import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[0]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_graphormer_freihand.config_utils import (
    build_dataloader,
    build_datasets,
    ensure_dir,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_freihand.lightning_module import FreiHANDPoseLightningModule


def parse_args():
    parser = argparse.ArgumentParser(description="Train Graphormer on FreiHAND.")
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "freihand_graphormer.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Resume training from checkpoint. Falls back to model.ckpt_path in config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override checkpoint directory from config.",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run test set evaluation with the best checkpoint after training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = normalize_config_paths(load_config(config_path))

    batch_size = config["training"]["batch_size"]
    max_epochs = config["training"]["max_epochs"]
    num_workers = config["training"]["num_workers"]
    learning_rate = config["training"]["learning_rate"]

    checkpoint_dir = args.checkpoint_dir or config["output"]["checkpoint_dir"]
    resume_ckpt_path = args.ckpt_path or config["model"].get("ckpt_path") or None

    pl.seed_everything(config["seed"])

    print(f"正在加载 FreiHAND 数据: {config['data']['root']}")
    datasets = build_datasets(config)
    train_loader = build_dataloader(datasets["train"], batch_size, num_workers, shuffle=True)
    val_loader = build_dataloader(datasets["val"], batch_size, num_workers, shuffle=False)
    test_loader = build_dataloader(datasets["test"], batch_size, num_workers, shuffle=False)

    print(
        f"数据加载完成 -> 训练集: {len(datasets['train'])}, "
        f"验证集: {len(datasets['val'])}, 测试集: {len(datasets['test'])}"
    )

    system = FreiHANDPoseLightningModule(
        lr=learning_rate,
        local_model_dir=config["model"]["local_model_dir"],
        num_joints=21,
    )

    ensure_dir(checkpoint_dir)
    ensure_dir(config["output"]["default_root_dir"])

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="pose-{epoch:02d}-{val_mpjpe_3d:.4f}",
        save_top_k=config["output"].get("save_top_k", 3),
        monitor="val_mpjpe_3d",
        mode="min",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_mpjpe_3d",
        patience=config["training"]["early_stop_patience"],
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", 1),
        precision=config["training"].get("precision", 32),
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
        default_root_dir=config["output"]["default_root_dir"],
    )

    if resume_ckpt_path and not Path(resume_ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint 不存在: {resume_ckpt_path}")

    if resume_ckpt_path:
        print(f"从 checkpoint 恢复训练: {resume_ckpt_path}")
    else:
        print("未提供 checkpoint，开始重新训练。")

    trainer.fit(
        model=system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt_path,
    )

    if args.run_test:
        print("\n训练完成，开始测试最佳模型...")
        trainer.test(model=system, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
