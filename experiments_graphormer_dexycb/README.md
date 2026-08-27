# DexYCB Graphormer Experiment

这套目录是基于现有 `pl_system_v6_graphormer.py` 单独整理出的 DexYCB 实验入口。

## 目录说明

- `dataset.py`: 读取 `dexycb/sampled_30k_3p8k_seed42` 这类独立子数据集目录。
- `lightning_module.py`: 对根目录 `PoseLightningModule` 的轻量包装，移除对 `hand_back` 和 `dist_coeffs` batch 字段的依赖。
- `train.py`: 训练入口。
- `test.py`: 测试入口。
- `configs/dexycb_graphormer.yaml`: 默认配置。
- `scripts/run_train.sh`: 训练脚本。
- `scripts/run_test.sh`: 测试脚本。

## 依赖关系

- 仍然复用根目录的 `pl_system_v6_graphormer.py`
- 仍然复用根目录的 `models/` 和 `utils/`
- 不修改原有 `scripts_pose/` 和旧数据集代码

## 使用方式

建议先进入 `vit` 环境：

```bash
conda activate vit
```

训练：

```bash
bash experiments_graphormer_dexycb/scripts/run_train.sh
```

测试：

```bash
bash experiments_graphormer_dexycb/scripts/run_test.sh --ckpt-path /path/to/checkpoint.ckpt
```

## 默认数据路径

配置默认使用：

```text
dexycb/sampled_30k_3p8k_seed42
```

要求该目录下至少包含：

- `train.txt` / `val.txt` / `test.txt` 或 `splits/train.txt` 等
- `train/...`、`val/...`、`test/...` 图像与 `labels_*.npz`
- `calibration/intrinsics/*.yml`
