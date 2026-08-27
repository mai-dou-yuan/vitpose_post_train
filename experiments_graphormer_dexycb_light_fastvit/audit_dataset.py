"""Build all official DexYCB splits and write an integrity audit."""

import argparse
import json
from pathlib import Path

from experiments_graphormer_dexycb_light_fastvit.config_utils import (
    build_datasets,
    load_config,
    normalize_config_paths,
)
from experiments_graphormer_dexycb_light_fastvit.dataset import FORBIDDEN_DATA_PARTS


THIS_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(THIS_DIR / "configs" / "dexycb_graphormer.yaml")
    )
    parser.add_argument("--setup", choices=("s0", "s1", "s2", "s3"), default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = normalize_config_paths(load_config(args.config))
    if args.setup is not None:
        config["data"]["setup"] = args.setup
    datasets = build_datasets(config)
    paths = {
        split: {sample["img_relative_path"] for sample in dataset.samples}
        for split, dataset in datasets.items()
    }
    sequences = {
        split: {(sample["subject"], sample["sequence"]) for sample in dataset.samples}
        for split, dataset in datasets.items()
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    path_overlaps = {f"{a}-{b}": len(paths[a] & paths[b]) for a, b in pairs}
    sequence_overlaps = {f"{a}-{b}": len(sequences[a] & sequences[b]) for a, b in pairs}
    forbidden = {
        split: sum(
            bool(FORBIDDEN_DATA_PARTS.intersection(Path(path).parts)) for path in split_paths
        )
        for split, split_paths in paths.items()
    }
    report = {
        "setup": config["data"].get("setup", "s0"),
        "data_root": config["data"]["root"],
        "splits": {split: dataset.audit for split, dataset in datasets.items()},
        "path_overlaps": path_overlaps,
        "sequence_overlaps": sequence_overlaps,
        "forbidden_sampled_paths": forbidden,
        "all_filtered_samples_are_right": all(
            sample["mano_side"] == "right"
            for dataset in datasets.values()
            for sample in dataset.samples
        ),
    }
    if any(path_overlaps.values()):
        raise AssertionError(f"DexYCB split path overlap: {path_overlaps}")
    if any(forbidden.values()):
        raise AssertionError(f"Forbidden sampled paths entered dataset: {forbidden}")
    if not report["all_filtered_samples_are_right"]:
        raise AssertionError("A non-right-hand sample entered the filtered dataset")

    output = Path(args.output).resolve() if args.output else (
        THIS_DIR / "cache" / f"full_dexycb_{report['setup']}" / "audit.json"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
    temporary.replace(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"audit_output={output}")


if __name__ == "__main__":
    main()
