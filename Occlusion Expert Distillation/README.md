# Occlusion Expert Distillation

本目录实现了一套不改动仓库原训练框架的遮挡专家训练与蒸馏子系统。所有新增代码都限制在当前目录内，通过 wrapper 复用现有 `datasets/dataset.py` 和 `pl_system_v6_graphormer.py`。

## 代码结构

- `common.py`: 数据集 / DataLoader / YAML 配置加载
- `losses.py`: self-occlusion mask、masked GNLL、Expert weighted GNLL、stable Gaussian KL、token KD
- `metrics.py`: Overall / Visible / Self-occluded / Out-of-view / Fingertip / PCK 指标
- `model_wrappers.py`: 对原 `PoseLightningModule` 做无侵入包装，补齐 `joint_token`、`all_stage_tokens`
- `expert_module.py`: Occlusion Expert LightningModule
- `distill_module.py`: Student distillation LightningModule
- `evaluate.py`: checkpoint 对比评估与前后指标表格输出
- `eval_compare.py`: 独立的 checkpoint 对比评估入口
- `train_expert.py`: Expert 训练入口
- `train_distill.py`: Student 蒸馏入口
- `configs/expert.yaml`: Expert 默认配置
- `configs/distill.yaml`: Distill 默认配置

## 代码库调研结论

当前仓库中实际对齐的是这些接口：

- Base model: `pl_system_v6_graphormer.PoseLightningModule`
- 主训练入口: `scripts_pose/train_pose.py`
- forward 现状:
  - 已返回 `pose3d`
  - 已返回 `pose3d_logvar`
  - 已返回 `all_stage_pose3d`
  - 已返回 `all_stage_logvars`
  - 未返回 `joint_token`
  - 未返回 `all_stage_tokens`
- checkpoint 加载方式:
  - 原仓库通常通过 `PoseLightningModule.load_from_checkpoint(...)` 或 Lightning `ckpt_path=...` 恢复
  - 这里为了不改原文件，改为读取 checkpoint 的 `hyper_parameters/state_dict`，在 wrapper 中重建模型并 `load_state_dict(...)`
- dataset 字段真实形状:
  - `img`: `[B, 3, 336, 336]`
  - `hand_back`: `[B, 3, 336, 336]`
  - `gt_pose`: `[B, 21, 3]`
  - `origin_3d`: `[B, 3]`
  - `cam_k`: `[B, 3, 3]`
  - `dist_coeffs`: `[B, 1, 5]`
  - `gt_pose_2d`: `[B, 21, 2]`
  - `visibility_label`: `[B, 21]`
  - `in_view_mask`: `[B, 21]`
  - `visible_joint_ratio`: `[B, 1]`

## 与方案文档的兼容差异

方案文档假设原 forward 已能直接输出 `joint_token` 和 `all_stage_tokens`，但现有 `pl_system_v6_graphormer.py` 并没有这些字段。这里的兼容方案是：

- 在 `model_wrappers.py` 里继承原 `PoseLightningModule`
- 完整复用 backbone / transformer / head
- 在本目录内重写 `forward`
- 额外收集每一层 refinement 后的 token，并统一输出：
  - `pose3d`
  - `pose3d_logvar`
  - `joint_token`
  - `all_stage_pose3d`
  - `all_stage_logvars`
  - `all_stage_tokens`

这样不需要修改目录外代码，同时满足蒸馏所需张量。

## 训练逻辑

### Self-occlusion 定义

`occ_mask = (visibility_label != 0) & in_view_mask`

这里不会把 out-of-view joints 混入 self-occlusion mask。

### Expert

- 初始化来源: Base checkpoint
- `gnll_warmup_epochs = 0`
- loss:
  - `L_expert = w_occ * L_GNLL_occ + w_vis * L_GNLL_vis`
- 默认:
  - `w_occ = 1.0`
  - `w_vis = 0.1`

实现细节：

- `compute_expert_gnll_weighted(...)` 支持 `vis_mask`
- 实际传入的是 `visible_mask = (visibility_label == 0) & in_view_mask`
- 因此 Expert 的可见 joint 权重不会把 out-of-view joints 混进去

### Student

- 初始化来源: Base checkpoint
- Teacher 来源: 冻结后的 Expert checkpoint
- `gnll_warmup_epochs = 0`
- loss:
  - `L_student = L_GNLL_all_stage + lambda_KL * L_KL_occ_all_stage + lambda_token * L_token_occ_last_stage`
- 默认:
  - 正式阶段: `lambda_KL = 0.5`, `lambda_token = 0.2`
  - 前 5 epoch warmup: `lambda_KL = 0.1`, `lambda_token = 0.05`
- Gaussian KL 方向:
  - `KL(P_E || P_S)`

## 运行前依赖

- Python 环境: `conda activate vit`
- 数据集: `save_path_15`
- 本地 backbone: `dinov2-base-local`

## 运行注意

- 当前推荐直接在已激活的 `vit` 环境中使用 `python -u ...`
- `python -u` 会使用非缓冲输出，Lightning 日志和进度条能实时显示
- 如果使用 `conda run`，推荐加 `--no-capture-output`，否则可能看起来像“没有输出”
- 当前默认 `training.pin_memory: false`
- 原因是该环境上出现过 DataLoader pin-memory 线程触发的 `CUDA error: invalid argument`
- 如果你换到另一个更稳定的环境，想测试更高吞吐，可以手动改回 `true`

## 输入 / 输出 checkpoint

- Base checkpoint 默认路径:
  - `/home/duanmu/data/vitpose_v3/vitpose_v3/checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt`
- Expert 输出目录:
  - `/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/expert/checkpoints/`
- Expert 导出 checkpoint:
  - `/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/expert/checkpoints/occlusion_expert_final.ckpt`
- Student 输出目录:
  - `/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/distill/checkpoints/`
- Student 导出 checkpoint:
  - `/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/distill/checkpoints/student_distilled_final.ckpt`

## 推荐运行方式

在 `vit` 环境里直接运行：

```bash
python -u "Occlusion Expert Distillation/train_expert.py" ...
```

如果必须用 `conda run`，请使用：

```bash
conda run --no-capture-output -n vit python -u "Occlusion Expert Distillation/train_expert.py" ...
```

## 训练命令

### 1. Expert 训练
# --lr 3e-5 \
```bash
python -u "Occlusion Expert Distillation/train_expert.py" \
  --config "Occlusion Expert Distillation/configs/expert.yaml" \
  --base-checkpoint "checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt" \
  --output-dir "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/expert_run_01" \
  --epochs 50 \
  --batch-size 24 \
  --lr 6e-5 \
  --w-occ 10.0 \
  --w-vis 0.01 \
  --num-workers 4 \
  --seed 42 \
  --compare-split none
```

说明：

- `w_occ`: self-occluded joints 的 loss 权重
- `w_vis`: visible joints 的 loss 权重
- `compare-split none`: 训练结束后不自动做前后对比

### 2. Student 蒸馏

```bash
python -u "Occlusion Expert Distillation/train_distill.py" \
  --config "Occlusion Expert Distillation/configs/distill.yaml" \
  --base-checkpoint "checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt" \
  --expert-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/expert_run_01/checkpoints/occlusion_expert_final.ckpt" \
  --output-dir "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/distill_run_01" \
  --epochs 50 \
  --batch-size 24 \
  --lr 3e-5 \
  --lambda-kl 0.5 \
  --lambda-token 0.2 \
  --num-workers 4 \
  --seed 42 \
  --compare-split none
```

说明：

- `lambda_kl`: self-occluded joints 上的 Gaussian KL 蒸馏权重
- `lambda_token`: self-occluded joints 上的 last-stage token KD 权重

## 单独算指标命令

如果你希望把训练和算指标完全分开，用 `eval_compare.py`。

### 3. Expert 单独算指标

验证集：

```bash
python -u "Occlusion Expert Distillation/eval_compare.py" \
  --config "Occlusion Expert Distillation/configs/expert.yaml" \
  --before-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt" \
  --after-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/expert_run_01/checkpoints/occlusion_expert_final.ckpt" \
  --split val \
  --before-name base \
  --after-name expert \
  --batch-size 24 \
  --num-workers 4 \
  --seed 42
```

测试集：

```bash
python -u "Occlusion Expert Distillation/eval_compare.py" \
  --config "Occlusion Expert Distillation/configs/expert.yaml" \
  --before-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt" \
  --after-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/expert_run_01/checkpoints/occlusion_expert_final.ckpt" \
  --split test \
  --before-name base \
  --after-name expert \
  --batch-size 24 \
  --num-workers 4 \
  --seed 42
```

### 4. Student 单独算指标

验证集：

```bash
python -u "Occlusion Expert Distillation/eval_compare.py" \
  --config "Occlusion Expert Distillation/configs/distill.yaml" \
  --before-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt" \
  --after-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/distill_run_01/checkpoints/student_distilled_final.ckpt" \
  --split val \
  --before-name base \
  --after-name student \
  --batch-size 24 \
  --num-workers 4 \
  --seed 42
```

测试集：

```bash
python -u "Occlusion Expert Distillation/eval_compare.py" \
  --config "Occlusion Expert Distillation/configs/distill.yaml" \
  --before-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt" \
  --after-checkpoint "/home/duanmu/data/vitpose_v3/vitpose_v3/Occlusion Expert Distillation/outputs/distill_run_01/checkpoints/student_distilled_final.ckpt" \
  --split test \
  --before-name base \
  --after-name student \
  --batch-size 24 \
  --num-workers 4 \
  --seed 42
```

## 指标

训练 / 验证 / 测试统一记录：

- `overall_mpjpe`
- `visible_mpjpe`
- `self_occ_mpjpe`
- `out_of_view_mpjpe`
- `fingertip_mpjpe`
- `self_occ_fingertip_mpjpe`
- `pck_20`
- `pck_30`

建议优先观察：

- `self_occ_mpjpe`
- `visible_mpjpe`
- `overall_mpjpe`

对比输出会直接给出：

- `base`
- `expert` 或 `student`
- `delta = after - before`

其中：

- MPJPE 类指标越小越好，`delta < 0` 表示提升
- PCK 类指标越大越好，`delta > 0` 表示提升

## 限制项

- 本实现没有修改目录外原始 `PoseLightningModule`，因此只能在 wrapper 中复制 forward 流程来拿到 `all_stage_tokens`
- `hand_back` 当前仍沿用原模型签名，但实际 `pl_system_v6_graphormer.py` 的 forward 主路径没有使用该分支，这里保持接口兼容，不在本目录外做结构改动
- README 中所有 Base model checkpoint 示例现统一为 `checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt`
