import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from experiments_graphormer_freihand_light_fastvit.config_utils import (
    build_dataloader,
    build_datasets,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_freihand_light_fastvit.lightning_module import (
    FreiHANDPoseLightningModule,
)
from experiments_graphormer_freihand_light_fastvit.logging_utils import (
    build_csv_loggers,
)
from experiments_graphormer_freihand_light_fastvit.checkpoint_utils import (
    load_pose_checkpoint_weights,
)


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
        pin_memory=config["training"].get("pin_memory", False),
        prefetch_factor=config["training"].get("prefetch_factor", False),
    )

    print(f"测试集加载完成: {len(test_dataset)}")

    initial_head_cfg = config["model"].get("initial_2d_head", {})
    system = FreiHANDPoseLightningModule(
        lr=config["training"]["learning_rate"],
        backbone_lr=config["training"].get(
            "backbone_learning_rate", config["training"]["learning_rate"]
        ),
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
    load_report = load_pose_checkpoint_weights(system, ckpt_path)
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(
            "测试 checkpoint 必须包含完整的 Initial 2D 和 mesh 模块；"
            "旧架构 checkpoint 只能用于重新训练初始化，不能直接测试。"
        )

    loggers = build_csv_loggers(config["output"]["default_root_dir"])
    print(f"详细指标日志: {loggers[0].log_dir}/metrics.csv")
    print(f"精简指标日志: {loggers[1].log_dir}/metrics.csv")

    trainer = pl.Trainer(
        accelerator=config["training"].get("accelerator", "auto"),
        devices=config["training"].get("devices", 1),
        precision=config["training"].get("precision", 32),
        log_every_n_steps=config["training"].get("log_every_n_steps", 20),
        default_root_dir=config["output"]["default_root_dir"],
        logger=loggers,
    )
    trainer.test(model=system, dataloaders=test_loader)


if __name__ == "__main__":
    main()
