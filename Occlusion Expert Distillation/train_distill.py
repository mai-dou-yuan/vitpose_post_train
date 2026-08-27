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
from distill_module import OcclusionDistillationModule
from evaluate import evaluate_checkpoint_pair
from model_wrappers import load_occlusion_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Distill student from base checkpoint with a frozen occlusion expert.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "configs" / "distill.yaml"))
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--expert-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-kl", type=float, default=None)
    parser.add_argument("--lambda-token", type=float, default=None)
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
            ("model.expert_checkpoint", args.expert_checkpoint),
            ("output.output_dir", args.output_dir),
            ("training.batch_size", args.batch_size),
            ("training.max_epochs", args.epochs),
            ("training.learning_rate", args.lr),
            ("training.lambda_kl", args.lambda_kl),
            ("training.lambda_token", args.lambda_token),
            ("training.num_workers", args.num_workers),
            ("seed", args.seed),
        ],
    )

    base_checkpoint = config["model"]["base_checkpoint"]
    expert_checkpoint = config["model"]["expert_checkpoint"]
    output_dir = ensure_dir(config["output"]["output_dir"])
    checkpoint_dir = ensure_dir(str(Path(output_dir) / "checkpoints"))
    logger_dir = ensure_dir(str(Path(output_dir) / "logs"))

    pl.seed_everything(config["seed"])
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_all_dataloaders(config)
    print(
        f"[Distill] data_root={config['data']['root']} "
        f"train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}"
    )
    print(f"[Distill] base_checkpoint={base_checkpoint}")
    print(f"[Distill] expert_checkpoint={expert_checkpoint}")

    student = load_occlusion_model_from_checkpoint(
        base_checkpoint,
        strict=False,
        gnll_warmup_epochs=config["training"]["gnll_warmup_epochs"],
    )
    expert = load_occlusion_model_from_checkpoint(
        expert_checkpoint,
        strict=False,
        gnll_warmup_epochs=0,
    )
    module = OcclusionDistillationModule(
        student=student,
        expert=expert,
        lr=config["training"]["learning_rate"],
        gnll_warmup_epochs=config["training"]["gnll_warmup_epochs"],
        scheduler_warmup_epochs=config["training"]["scheduler_warmup_epochs"],
        weight_decay=config["training"]["weight_decay"],
        min_lr=config["training"]["min_lr"],
        lambda_kl=config["training"]["lambda_kl"],
        lambda_token=config["training"]["lambda_token"],
        kd_warmup_epochs=config["training"]["kd_warmup_epochs"],
        lambda_kl_warmup=config["training"]["lambda_kl_warmup"],
        lambda_token_warmup=config["training"]["lambda_token_warmup"],
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="student_distilled",
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
    logger = CSVLogger(save_dir=logger_dir, name="distill")

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
    final_export_path = str(Path(checkpoint_dir) / "student_distilled_final.ckpt")
    if best_path:
        shutil.copy2(best_path, final_export_path)
        print(f"[Distill] best checkpoint -> {best_path}")
        print(f"[Distill] exported checkpoint -> {final_export_path}")

    if args.run_test and best_path:
        trainer.test(module, dataloaders=test_loader, ckpt_path=best_path)

    if args.compare_split != "none" and best_path:
        compare_loader = val_loader if args.compare_split == "val" else test_loader
        device = module.device if module.device.type != "cpu" or trainer.num_devices >= 0 else student.device
        base_model = load_occlusion_model_from_checkpoint(
            base_checkpoint,
            strict=False,
            gnll_warmup_epochs=0,
        )
        distilled_model = load_occlusion_model_from_checkpoint(
            best_path,
            strict=False,
            gnll_warmup_epochs=0,
        )
        _, _, table = evaluate_checkpoint_pair(
            base_model,
            distilled_model,
            compare_loader,
            device=device,
            before_name="base",
            after_name="student",
        )
        print(f"\n[Distill] compare on {args.compare_split} split")
        print(table)


if __name__ == "__main__":
    main()
