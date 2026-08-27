"""Run one real DexYCB batch through forward, training_step and backward."""

from pathlib import Path

import torch

from experiments_graphormer_dexycb_light_fastvit.config_utils import (
    build_dataloader,
    build_datasets,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_dexycb_light_fastvit.lightning_module import (
    DexYCBPoseLightningModule,
)


THIS_DIR = Path(__file__).resolve().parent


def _move_batch(batch, device):
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def main():
    config = normalize_config_paths(
        load_config(THIS_DIR / "configs" / "dexycb_graphormer.yaml")
    )
    datasets = build_datasets(config)
    loaders = {
        split: build_dataloader(
            dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )
        for split, dataset in datasets.items()
    }
    batch = next(iter(loaders["train"]))

    model_cfg = config["model"]
    training_cfg = config["training"]
    initial_head_cfg = model_cfg.get("initial_2d_head", {})
    model = DexYCBPoseLightningModule(
        lr=training_cfg["learning_rate"],
        backbone_lr=training_cfg.get(
            "backbone_learning_rate", training_cfg["learning_rate"]
        ),
        backbone_freeze_epochs=0,
        lr_warmup_epochs=training_cfg.get("lr_warmup_epochs", 5),
        vitpose_config_path=model_cfg["vitpose_config_path"],
        vitpose_checkpoint_path=model_cfg["vitpose_checkpoint_path"],
        vitpose_dataset_source=model_cfg.get("vitpose_dataset_source", 5),
        local_grid_size=model_cfg.get("local_grid_size", 5),
        local_grid_radius=model_cfg.get("local_grid_radius", 2.0),
        num_refine_layers=model_cfg.get("num_refine_layers", 3),
        initial_2d_loss_weight=training_cfg.get("initial_2d_loss_weight", 0.1),
        joint_2d_loss_weight=training_cfg.get("joint_2d_loss_weight", 0.02),
        joint_3d_loss_weight=training_cfg.get("joint_3d_loss_weight", 1.0),
        stage_supervision_weights=training_cfg.get(
            "stage_supervision_weights", (0.1, 0.3, 1.0)
        ),
        vertices_loss_weight=training_cfg.get("vertices_loss_weight", 10.0),
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
        use_gradient_checkpointing=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).train()
    batch = _move_batch(batch, device)

    # A direct training_step has no Lightning Trainer/logger attached.
    model.log = lambda *args, **kwargs: None
    loss = model.training_step(batch, batch_idx=0)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f"Expected a finite scalar loss, got {loss}")
    loss.backward()

    watched_modules = {
        "backbone": model.vitmodel,
        "initial_2d_head": model.initial_2d_head,
        "joint_3d_head": model.pose_3d_head_PR,
        "mesh_regressor": model.mesh_regressor,
    }
    gradient_counts = {
        name: sum(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        )
        for name, module in watched_modules.items()
    }
    if not all(gradient_counts.values()):
        raise AssertionError(f"Missing finite gradients: {gradient_counts}")

    model.eval()
    with torch.no_grad():
        validation_batch = _move_batch(next(iter(loaders["val"])), device)
        test_batch = _move_batch(next(iter(loaders["test"])), device)
        model.validation_step(validation_batch, batch_idx=0)
        model.test_step(test_batch, batch_idx=0)

    print(f"device={device}")
    print(f"sample={batch['dataset_idx'][0]}")
    print(f"image_shape={tuple(batch['img'].shape)}")
    print(f"joint_shape={tuple(batch['gt_pose'].shape)}")
    print(f"vertex_shape={tuple(batch['gt_vertices'].shape)}")
    print(f"valid_joints={int(batch['joint_valid'].sum())}/21")
    print(f"loss={loss.detach().item():.8f}")
    print(f"stage_weights={model.stage_supervision_weights}")
    print(f"finite_gradient_tensors={gradient_counts}")
    print(f"validation_sample={validation_batch['dataset_idx'][0]}")
    print(f"test_sample={test_batch['dataset_idx'][0]}")
    print("validation_test_steps=PASSED")
    print("minimal_training_step=PASSED")


if __name__ == "__main__":
    main()
