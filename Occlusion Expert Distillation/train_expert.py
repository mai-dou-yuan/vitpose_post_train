import argparse
import shutil
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import apply_overrides, build_all_dataloaders, ensure_dir, load_yaml_config
from evaluate import evaluate_checkpoint_pair
from expert_module import OcclusionExpertModule
from model_wrappers import load_occlusion_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train occlusion expert from a base checkpoint.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "configs" / "expert.yaml"))
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--w-occ", type=float, default=None)
    parser.add_argument("--w-vis", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--fast-dev-run", type=int, default=0)
    parser.add_argument("--compare-split", choices=["none", "val", "test"], default="val")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_overrides(
        config,
        [
            ("model.base_checkpoint", args.base_checkpoint),
            ("output.output_dir", args.output_dir),
            ("training.batch_size", args.batch_size),
            ("training.max_epochs", args.epochs),
            ("training.learning_rate", args.lr),
            ("training.w_occ", args.w_occ),
            ("training.w_vis", args.w_vis),
            ("training.num_workers", args.num_workers),
            ("seed", args.seed),
        ],
    )

    base_checkpoint = config["model"]["base_checkpoint"]
    output_dir = ensure_dir(config["output"]["output_dir"])
    checkpoint_dir = ensure_dir(str(Path(output_dir) / "checkpoints"))
    logger_dir = ensure_dir(str(Path(output_dir) / "logs"))

    pl.seed_everything(config["seed"])
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_all_dataloaders(config)
    print(
        f"[Expert] data_root={config['data']['root']} "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    print(f"[Expert] base_checkpoint={base_checkpoint}")

    model = load_occlusion_model_from_checkpoint(
        base_checkpoint,
        strict=False,
        gnll_warmup_epochs=config["training"]["gnll_warmup_epochs"],
    )
    module = OcclusionExpertModule(
        model=model,
        lr=config["training"]["learning_rate"],
        w_occ=config["training"]["w_occ"],
        w_vis=config["training"]["w_vis"],
        gnll_warmup_epochs=config["training"]["gnll_warmup_epochs"],
        scheduler_warmup_epochs=config["training"]["scheduler_warmup_epochs"],
        weight_decay=config["training"]["weight_decay"],
        min_lr=config["training"]["min_lr"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="occlusion_expert",
        save_top_k=1,
        save_last=True,
        monitor="val_self_occ_mpjpe",
        mode="min",
        auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(
        monitor="val_self_occ_mpjpe",
        patience=config["training"]["early_stop_patience"],
        mode="min",
    )
    logger = CSVLogger(save_dir=logger_dir, name="expert")

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["runtime"]["accelerator"],
        devices=config["runtime"]["devices"],
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        log_every_n_steps=config["runtime"]["log_every_n_steps"],
        gradient_clip_val=config["runtime"]["gradient_clip_val"],
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    final_export_path = str(Path(checkpoint_dir) / "occlusion_expert_final.ckpt")
    if best_path:
        shutil.copy2(best_path, final_export_path)
        print(f"[Expert] best checkpoint -> {best_path}")
        print(f"[Expert] exported checkpoint -> {final_export_path}")

    if args.run_test and best_path:
        trainer.test(module, dataloaders=test_loader, ckpt_path=best_path)

    if args.compare_split != "none" and best_path:
        compare_loader = val_loader if args.compare_split == "val" else test_loader
        device = module.device if module.device.type != "cpu" or trainer.num_devices >= 0 else model.device
        base_model = load_occlusion_model_from_checkpoint(
            base_checkpoint,
            strict=False,
            gnll_warmup_epochs=0,
        )
        trained_model = load_occlusion_model_from_checkpoint(
            best_path,
            strict=False,
            gnll_warmup_epochs=0,
        )
        _, _, table = evaluate_checkpoint_pair(
            base_model,
            trained_model,
            compare_loader,
            device=device,
            before_name="base",
            after_name="expert",
        )
        print(f"\n[Expert] compare on {args.compare_split} split")
        print(table)


if __name__ == "__main__":
    main()
