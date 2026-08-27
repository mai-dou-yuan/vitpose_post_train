import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_graphormer_dexycb_light_fastvit.config_utils import (
    build_dataloader,
    build_datasets,
    ensure_dir,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_dexycb_light_fastvit.lightning_module import (
    DexYCBPoseLightningModule,
)
from experiments_graphormer_dexycb_light_fastvit.logging_utils import (
    build_csv_loggers,
)
from experiments_graphormer_dexycb_light_fastvit.checkpoint_utils import (
    is_legacy_initial_2d_checkpoint,
    load_pose_checkpoint_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Light FastViT Graphormer on DexYCB.")
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "configs" / "dexycb_graphormer.yaml"),
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
    pin_memory = config["training"].get("pin_memory", False)
    prefetch_factor = config["training"].get("prefetch_factor", False)

    checkpoint_dir = args.checkpoint_dir or config["output"]["checkpoint_dir"]
    resume_ckpt_path = args.ckpt_path or config["model"].get("ckpt_path") or None

    pl.seed_everything(config["seed"])

    print(f"正在加载 DexYCB 数据: {config['data']['root']}")
    datasets = build_datasets(config)
    train_loader = build_dataloader(
        datasets["train"],
        batch_size,
        num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    val_loader = build_dataloader(
        datasets["val"],
        batch_size,
        num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    test_loader = build_dataloader(
        datasets["test"],
        batch_size,
        num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    print(
        f"数据加载完成 -> 训练集: {len(datasets['train'])}, "
        f"验证集: {len(datasets['val'])}, 测试集: {len(datasets['test'])}"
    )

    initial_head_cfg = config["model"].get("initial_2d_head", {})
    system = DexYCBPoseLightningModule(
        lr=learning_rate,
        backbone_lr=config["training"].get("backbone_learning_rate", learning_rate),
        backbone_freeze_epochs=config["training"].get("backbone_freeze_epochs", 0),
        lr_warmup_epochs=config["training"].get("lr_warmup_epochs", 5),
        vitpose_config_path=config["model"]["vitpose_config_path"],
        vitpose_checkpoint_path=config["model"]["vitpose_checkpoint_path"],
        vitpose_dataset_source=config["model"].get("vitpose_dataset_source", 5),
        local_grid_size=config["model"].get("local_grid_size", 5),
        local_grid_radius=config["model"].get("local_grid_radius", 2.0),
        num_refine_layers=config["model"].get("num_refine_layers", 3),
        initial_2d_loss_weight=config["training"].get(
            "initial_2d_loss_weight", 0.1
        ),
        joint_2d_loss_weight=config["training"].get("joint_2d_loss_weight", 0.02),
        joint_3d_loss_weight=config["training"].get("joint_3d_loss_weight", 1.0),
        stage_supervision_weights=config["training"].get(
            "stage_supervision_weights", (0.1, 0.3, 1.0)
        ),
        vertices_loss_weight=config["training"].get("vertices_loss_weight", 10.0),
        initial_2d_bottleneck_channels=initial_head_cfg.get(
            "bottleneck_channels", 16
        ),
        initial_2d_pooled_size=(
            initial_head_cfg.get("pooled_height", 4),
            initial_head_cfg.get("pooled_width", 3),
        ),
        initial_2d_hidden_dim=initial_head_cfg.get("hidden_dim", 32),
        initial_2d_dropout=initial_head_cfg.get("dropout", 0.1),
        num_joints=21,
    )

    ensure_dir(checkpoint_dir)
    ensure_dir(config["output"]["default_root_dir"])
    loggers = build_csv_loggers(config["output"]["default_root_dir"])
    print(f"详细指标日志: {loggers[0].log_dir}/metrics.csv")
    print(f"精简指标日志: {loggers[1].log_dir}/metrics.csv")

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
        logger=loggers,
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        gradient_clip_val=config["training"].get("gradient_clip_val", 1.0),
        default_root_dir=config["output"]["default_root_dir"],
    )

    if resume_ckpt_path and not Path(resume_ckpt_path).exists():
        raise FileNotFoundError(f"checkpoint 不存在: {resume_ckpt_path}")

    trainer_resume_ckpt_path = resume_ckpt_path
    if resume_ckpt_path:
        load_report = load_pose_checkpoint_weights(system, resume_ckpt_path)
        if is_legacy_initial_2d_checkpoint(load_report):
            trainer_resume_ckpt_path = None
            print(
                "检测到旧架构 checkpoint：已非严格加载兼容权重并报告新增 2D/mesh "
                "模块的 missing keys；旧优化器参数组不完整，因此从新优化器开始训练。"
            )
        elif load_report["missing_keys"] or load_report["unexpected_keys"]:
            raise RuntimeError(
                "checkpoint 含预期新增模块以外的不兼容键，拒绝恢复训练。"
            )
        else:
            print(f"从 checkpoint 完整恢复训练: {resume_ckpt_path}")
    else:
        print("未提供 checkpoint，开始重新训练。")

    trainer.fit(
        model=system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=trainer_resume_ckpt_path,
    )

    if args.run_test:
        print("\n训练完成，开始测试最佳模型...")
        trainer.test(model=system, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
