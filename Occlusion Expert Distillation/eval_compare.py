import argparse
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common import apply_overrides, build_all_dataloaders, load_yaml_config
from evaluate import evaluate_checkpoint_pair
from model_wrappers import load_occlusion_model_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two checkpoints on val/test split.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "configs" / "distill.yaml"))
    parser.add_argument("--before-checkpoint", required=True, help="Reference checkpoint, usually the base model.")
    parser.add_argument("--after-checkpoint", required=True, help="Target checkpoint, usually expert or distilled student.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--before-name", default="before")
    parser.add_argument("--after-name", default="after")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_overrides(
        config,
        [
            ("training.batch_size", args.batch_size),
            ("training.num_workers", args.num_workers),
            ("seed", args.seed),
        ],
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)

    _, val_dataset, test_dataset, _, val_loader, test_loader = build_all_dataloaders(config)
    loader = val_loader if args.split == "val" else test_loader
    dataset = val_dataset if args.split == "val" else test_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    before_model = load_occlusion_model_from_checkpoint(
        args.before_checkpoint,
        strict=False,
        gnll_warmup_epochs=0,
    )
    after_model = load_occlusion_model_from_checkpoint(
        args.after_checkpoint,
        strict=False,
        gnll_warmup_epochs=0,
    )

    before_metrics, after_metrics, table = evaluate_checkpoint_pair(
        before_model,
        after_model,
        loader,
        device=device,
        before_name=args.before_name,
        after_name=args.after_name,
    )

    print(f"[Compare] split={args.split} size={len(dataset)} device={device}")
    print(f"[Compare] before_checkpoint={args.before_checkpoint}")
    print(f"[Compare] after_checkpoint={args.after_checkpoint}")
    print(table)


if __name__ == "__main__":
    main()
