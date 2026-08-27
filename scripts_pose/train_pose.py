import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.dataset import Unrealego3DPoseDataset
from pl_system_v6_graphormer import PoseLightningModule


DEFAULT_TRAIN_USER_IDS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14]
DEFAULT_VAL_USER_IDS = [4, 8, 13]
DEFAULT_TEST_USER_IDS = [4, 8, 13]


def parse_args():
    parser = argparse.ArgumentParser(description="Train pose model.")
    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "config.yaml"),
        help="Path to config yaml.",
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help="Resume training from checkpoint. Falls back to model.ckpt_path in config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(ROOT_DIR / "checkpoints"),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--joint-prior-path",
        # default=str(ROOT_DIR / "pre_train" / "joint_query_prior.pt"),
        default="",
        help="Path to the pretrained joint query prior. Set to an empty string to disable.",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run test set evaluation with the best checkpoint after training.",
    )
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


def build_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _copy_tensor_param(dst_tensor, src_tensor, name):
    if tuple(dst_tensor.shape) != tuple(src_tensor.shape):
        raise ValueError(
            f"{name} shape mismatch: model={tuple(dst_tensor.shape)} prior={tuple(src_tensor.shape)}"
        )
    dst_tensor.data.copy_(src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))


def load_joint_query_prior(model, prior_path):
    if not prior_path:
        print("未指定 joint query prior，使用随机初始化的 joint tokens。")
        return
    if not os.path.exists(prior_path):
        raise FileNotFoundError(f"joint query prior 不存在: {prior_path}")

    prior = torch.load(prior_path, map_location="cpu")
    loaded_items = []

    if "joint_tokens" in prior:
        _copy_tensor_param(model.joint_tokens, prior["joint_tokens"], "joint_tokens")
        loaded_items.append("joint_tokens")
    if "joint_token_pos" in prior:
        _copy_tensor_param(model.joint_token_pos, prior["joint_token_pos"], "joint_token_pos")
        loaded_items.append("joint_token_pos")

    for layer_idx in range(min(2, len(model.layers_sa))):
        layer = model.layers_sa[layer_idx]
        if not hasattr(layer, "spatial_bias_table") or not hasattr(layer, "same_finger_bias"):
            continue

        spd_key = f"layers_sa.{layer_idx}.spatial_bias_table.weight"
        finger_key = f"layers_sa.{layer_idx}.same_finger_bias.weight"

        if spd_key in prior:
            _copy_tensor_param(layer.spatial_bias_table.weight, prior[spd_key], spd_key)
            loaded_items.append(spd_key)
        if finger_key in prior:
            _copy_tensor_param(layer.same_finger_bias.weight, prior[finger_key], finger_key)
            loaded_items.append(finger_key)

    if not loaded_items:
        raise ValueError(f"{prior_path} 中没有可加载到 PoseLightningModule 的 joint prior 权重。")

    print(f"已加载 joint query prior: {prior_path}")
    print("载入参数: " + ", ".join(loaded_items))


def main():
    args = parse_args()
    config = load_config(args.config)

    data_root = config["data"]["root"]
    image_size = config["data"]["image_size"]

    batch_size = config["training"]["batch_size"]
    max_epochs = config["training"]["max_epochs"]
    num_workers = config["training"]["num_workers"]
    learning_rate = config["training"]["learning_rate"]

    local_model_dir = config["model"]["local_model_dir"]
    config_ckpt_path = config["model"].get("ckpt_path") or None
    resume_ckpt_path = args.ckpt_path or config_ckpt_path

    pl.seed_everything(config["seed"])

    print(f"正在加载训练数据: {data_root}")
    train_dataset, val_dataset, test_dataset = build_dataset(data_root, image_size)

    train_loader = build_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size, num_workers, shuffle=False)
    test_loader = build_dataloader(test_dataset, batch_size, num_workers, shuffle=False)

    print(
        f"数据加载完成 -> 训练集: {len(train_dataset)}, "
        f"验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}"
    )

    system = PoseLightningModule(
        lr=learning_rate,
        local_model_dir=local_model_dir,
        num_joints=21,
    )
    load_joint_query_prior(system, args.joint_prior_path)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="pose-{epoch:02d}-{val_mpjpe_3d:.4f}",
        save_top_k=3,
        monitor="val_mpjpe_3d",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_mpjpe_3d",
        patience=15,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    if resume_ckpt_path and not os.path.exists(resume_ckpt_path):
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
