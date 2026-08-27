"""Minimal offline verification for the trainable ViTPose++-B integration."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the Lightning module before config_utils loads the repository's local
# `datasets` package; transformers probes that package name during import.
from experiments_graphormer_dexycb_light_fastvit.lightning_module import (
    DexYCBPoseLightningModule,
)
from experiments_graphormer_dexycb_light_fastvit.config_utils import (
    load_config,
    normalize_config_paths,
)


def main():
    config_path = THIS_DIR / "configs" / "dexycb_graphormer.yaml"
    config = normalize_config_paths(load_config(config_path))
    model_cfg = config["model"]
    initial_head_cfg = model_cfg.get("initial_2d_head", {})

    model = DexYCBPoseLightningModule(
        lr=config["training"]["learning_rate"],
        vitpose_config_path=model_cfg["vitpose_config_path"],
        vitpose_checkpoint_path=model_cfg["vitpose_checkpoint_path"],
        vitpose_dataset_source=model_cfg.get("vitpose_dataset_source", 5),
        local_grid_size=model_cfg.get("local_grid_size", 5),
        local_grid_radius=model_cfg.get("local_grid_radius", 2.0),
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
        use_gradient_checkpointing=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    backbone_params = list(model.vitmodel.backbone.parameters())
    assert backbone_params, "ViTPose++-B has no parameters"
    assert all(param.requires_grad for param in backbone_params), "ViTPose++-B is frozen"

    model._set_backbone_trainable(False)
    model.vitmodel.eval()
    assert not any(param.requires_grad for param in backbone_params)
    assert not model.vitmodel.backbone.backbone.training
    model._set_backbone_trainable(True)
    model.train()
    assert all(param.requires_grad for param in backbone_params)
    assert model.vitmodel.backbone.backbone.training

    load_report = model.vitmodel.backbone.last_load_report
    assert load_report["key_ratio"] == 1.0
    assert load_report["numel_ratio"] == 1.0

    # configure_optimizers reads max_epochs from the attached Trainer. A small
    # stand-in is sufficient here because no Lightning training loop is started.
    model._trainer = SimpleNamespace(max_epochs=10)
    optimizer = model.configure_optimizers()["optimizer"]
    optimizer_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    missing_from_optimizer = [
        name
        for name, param in model.vitmodel.backbone.named_parameters()
        if id(param) not in optimizer_param_ids
    ]
    assert not missing_from_optimizer, (
        f"ViTPose parameters missing from optimizer: {missing_from_optimizer[:5]}"
    )

    captured = {}

    def capture_feature_shape(_module, _inputs, output):
        captured["feature_shape"] = tuple(output.shape)

    hook = model.vitmodel.register_forward_hook(capture_feature_shape)
    image_size = config["data"]["image_size"]
    images = torch.rand(1, 3, image_size, image_size, device=device)
    cam_k = torch.tensor(
        [[[500.0, 0.0, 111.5], [0.0, 500.0, 111.5], [0.0, 0.0, 1.0]]],
        device=device,
    )
    root_3d = torch.tensor([[0.0, 0.0, 0.5]], device=device)
    results = model(images, images, cam_k=cam_k, root_3d=root_3d)
    hook.remove()

    assert captured["feature_shape"] == (1, 768, 16, 12)
    assert results["pose3d"].shape == (1, 21, 3)
    assert results["initial_pose2d_normalized"].shape == (1, 21, 2)
    assert "initial_heatmaps" not in results
    assert bool((results["initial_pose2d_normalized"] >= 0.0).all())
    assert bool((results["initial_pose2d_normalized"] <= 1.0).all())
    assert torch.allclose(results["stage_reference_points"][0], results["initial_pose2d"])
    assert results["stage_reference_points"][2] is None
    forbidden_modules = [
        name
        for name, _ in model.vitmodel.named_modules()
        if any(part in name.lower() for part in ("neck", "heatmap", "keypoint_head"))
    ]
    assert not forbidden_modules, f"Unexpected 2D pose modules: {forbidden_modules}"

    loss = results["pose3d"].square().mean()
    loss.backward()

    params_with_grad = sum(param.grad is not None for param in backbone_params)
    assert params_with_grad > 0, "No ViTPose parameter received a gradient"
    initial_head_params = list(model.initial_2d_head.parameters())
    initial_head_param_count = sum(param.numel() for param in initial_head_params)
    assert initial_head_param_count <= 12_000
    assert all(param.grad is not None for param in initial_head_params)
    assert all(torch.isfinite(param.grad).all() for param in initial_head_params)

    print(f"config={model_cfg['vitpose_config_path']}")
    print(f"checkpoint={model_cfg['vitpose_checkpoint_path']}")
    print(
        f"checkpoint_load={load_report['loaded_keys']}/{load_report['total_keys']} "
        f"tensors, {load_report['numel_ratio']:.2%} values"
    )
    print(f"dataset_source={model_cfg.get('vitpose_dataset_source', 5)}")
    print(f"dataset_input_shape={tuple(images.shape)}")
    print(f"device={device}")
    print(f"feature_map_shape={captured['feature_shape']}")
    print(f"graphormer_output_shape={tuple(results['pose3d'].shape)}")
    print(f"initial_2d_shape={tuple(results['initial_pose2d_normalized'].shape)}")
    print(f"initial_2d_head_params={initial_head_param_count:,}")
    print("initial_2d_head=direct_spatial_grid_regression")
    print("stage0_reference=initial_2d_crop_pixels")
    print("stage1_reference=stage0_3d_projection")
    print("stage2_attention=full")
    print(f"vitpose_trainable_params={sum(p.numel() for p in backbone_params):,}")
    print(f"vitpose_tensors_with_grad={params_with_grad}/{len(backbone_params)}")
    print("vitpose_in_optimizer=True")
    print("freeze_unfreeze_modes=True")
    print("neck_heatmap_keypoint_modules=none")
    print("verification=PASSED")


if __name__ == "__main__":
    main()
