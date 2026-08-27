import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset import Unrealego3DPoseDataset
from pre_train.pl_joint_token_pretrain import JointTokenPretrainModule


DEFAULT_TRAIN_USER_IDS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14]
DEFAULT_VAL_USER_IDS = [4, 8, 13]
DEFAULT_TEST_USER_IDS = [4, 8, 13]


class JointPriorExportCallback(Callback):
    def __init__(self, output_path: str):
        super().__init__()
        self.output_path = output_path

    def on_fit_end(self, trainer, pl_module):
        self._save(pl_module)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._save(pl_module)

    def _save(self, pl_module):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        torch.save(pl_module.export_joint_query_prior(), self.output_path)


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain hand joint query tokens.")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT_DIR / "pre_train" / "checkpoints"),
        help="Directory to save pretraining checkpoints.",
    )
    parser.add_argument(
        "--prior-output",
        default=str(ROOT_DIR / "pre_train" / "joint_query_prior.pt"),
        help="Path to save lightweight joint token prior.",
    )
    parser.add_argument("--ckpt-path", default=None, help="Resume pretraining checkpoint.")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--num-refine-layers", type=int, default=2)
    parser.add_argument("--vit-layers", default="-1", help="Comma separated ViT layers, e.g. -1 or 3,6,-1.")
    parser.add_argument("--max-patch-tokens", type=int, default=4096)
    parser.add_argument("--topology-alpha-init", type=float, default=0.0)
    parser.add_argument("--rel-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pin_memory. Disabled by default to avoid CUDA pinning issues.",
    )
    parser.add_argument("--run-test", action="store_true")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(data_root, image_size):
    train_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=True,
        target_user_ids=DEFAULT_TRAIN_USER_IDS,
    )
    val_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=False,
        target_user_ids=DEFAULT_VAL_USER_IDS,
    )
    test_dataset = Unrealego3DPoseDataset(
        data_root=data_root,
        img_size=image_size,
        is_train=False,
        target_user_ids=DEFAULT_TEST_USER_IDS,
    )
    return train_dataset, val_dataset, test_dataset


def build_dataloader(dataset, batch_size, num_workers, shuffle, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    data_root = config["data"]["root"]
    image_size = config["data"]["image_size"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["training"]["num_workers"]
    local_model_dir = config["model"]["local_model_dir"]
    seed = config.get("seed", 42)
    vit_layers = parse_int_list(args.vit_layers)

    pl.seed_everything(seed)

    print(f"正在加载 joint token 预训练数据: {data_root}")
    train_dataset, val_dataset, test_dataset = build_dataset(data_root, image_size)
    train_loader = build_dataloader(
        train_dataset,
        batch_size,
        num_workers,
        shuffle=True,
        pin_memory=args.pin_memory,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size,
        num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size,
        num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
    )
    print(
        f"数据加载完成 -> 训练集: {len(train_dataset)}, "
        f"验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}"
    )

    system = JointTokenPretrainModule(
        lr=args.lr,
        num_joints=21,
        local_model_dir=local_model_dir,
        feature_dim=args.feature_dim,
        d_model=args.d_model,
        vit_layers=vit_layers,
        num_refine_layers=args.num_refine_layers,
        max_patch_tokens=args.max_patch_tokens,
        topology_alpha_init=args.topology_alpha_init,
        rel_loss_weight=args.rel_loss_weight,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="joint-token-{epoch:02d}-{val_mpjpe_3d:.4f}",
        save_top_k=3,
        monitor="val_mpjpe_3d",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_mpjpe_3d",
        patience=10,
        mode="min",
    )
    export_callback = JointPriorExportCallback(args.prior_output)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, export_callback],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    if args.ckpt_path and not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {args.ckpt_path}")

    trainer.fit(
        model=system,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.ckpt_path,
    )

    if checkpoint_callback.best_model_path:
        best_model = JointTokenPretrainModule.load_from_checkpoint(checkpoint_callback.best_model_path)
        torch.save(best_model.export_joint_query_prior(), args.prior_output)
        print(f"已从最佳 checkpoint 导出 joint query prior: {args.prior_output}")
    else:
        torch.save(system.export_joint_query_prior(), args.prior_output)
        print(f"已导出 joint query prior: {args.prior_output}")

    if args.run_test:
        print("开始测试最佳 joint token 预训练模型...")
        trainer.test(model=system, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
