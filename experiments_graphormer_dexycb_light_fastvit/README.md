# DexYCB Light FastViT Graphormer

独立的 DexYCB 训练、验证和测试实验。模型、stage supervision、loss、指标、
optimizer/scheduler、checkpoint 和双 CSV 日志沿用
`experiments_graphormer_freihand_light_fastvit/`；数据路径、DexYCB 标注和相机内参
读取方式与 NVIDIA 官方 `dex-ycb-toolkit` 的 setup 映射保持一致。

> 目录名称为兼容现有实验命名。当前主模板实际使用可训练的 ViTPose++-B backbone，
> 因此本目录也保持该实现，没有擅自换回旧 FastViT backbone。

## 数据与坐标

- 原始图像、label、metadata 和 calibration 均直接读取 `dexycb`，不复制数据。
- 默认使用官方 `s0` train/val/test；配置中的 `data.setup` 可切换 `s0` 至 `s3`。
- loader 只枚举官方 10 个 subject 和 8 个 camera，因此不会扫描本地生成的
  `sampled_30k_3p8k_seed42` 或 `splits_30k_3p8k_seed42`。
- `joint_3d`、`pose_m` 和 MANO 顶点都保持相机坐标系中的米单位。
- DexYCB `pose_m` 的 45 个手部参数按完整 MANO PCA basis 和 non-flat hand mean
  解码，再结合每个 subject 的 10 个 betas 重建 `[778,3]` 顶点。
- 原始 DexYCB 同时包含左右手；loader 根据 sequence `meta.yml` 仅索引
  `mano_sides == ["right"]`，左手被统计并排除，不做镜像。
- 构建 split 时仅保留原图内至少 `12/21` 个 joints、hand segmentation 非空，且
  未裁剪 bbox 与原图相交的样例；此外要求基础 crop 内仍有 segmentation id 255。
- `cache/full_dexycb_s0/` 分开保存官方候选索引和过滤结果。setup、split、metadata
  或过滤参数变化会使相应缓存失效；缓存目录已被 Git 忽略。
- 验证/测试直接根据投影关节构造方形 crop。训练增强先同步 camera-Z 旋转原图、
  3D joints 和 mesh，再按旋转后的 joints 重算 bbox，最后执行 center/scale jitter、
  crop 和 resize；2D joints 与相机内参始终同步更新。

每个 batch 的核心字段为：

- `img [B,3,H,W]`
- `gt_pose [B,21,3]`、`gt_pose_2d [B,21,2]`
- `gt_vertices [B,778,3]`
- `origin_3d [B,3]`、`cam_k [B,3,3]`
- `joint_valid [B,21]`、`vertex_valid [B,778]`、`is_right [B]`

## Loss 与指标

三个 Transformer refinement stages 的 token joint predictions 分别计算 joint 2D/3D
loss，stage 原始权重为 `[0.1, 0.3, 1.0]`，计算时归一化为
`[1/14, 3/14, 10/14]`。总 loss 为：

```text
1.0 * stage-weighted joint 3D
+ 0.02 * stage-weighted projected joint 2D
+ 0.1 * initial 2D coordinate-head loss
+ 10.0 * final-stage mesh vertex L1 loss
```

mesh loss 只作用于 `mesh_token_projection(curr_tokens)` 后、最终 stage 的 778 顶点。
mesh 回归得到的 joints 只用于 MPJPE/PA-MPJPE 指标，不再计算 joint loss，避免重复监督。

MPJPE、PA-MPJPE、MPVPE、PA-MPVPE 与 FreiHAND 模板完全同名并保持米单位；训练日志
同时记录 step/epoch 指标，验证和测试记录 epoch 聚合值。详细 CSV 保存全部 loss 和诊断，
summary CSV 只保留最终四项指标及各 stage 3D loss。

## 命令

```bash
conda run -n vit python -m experiments_graphormer_dexycb_light_fastvit.train \
  --config experiments_graphormer_dexycb_light_fastvit/configs/dexycb_graphormer.yaml

conda run -n vit python -m experiments_graphormer_dexycb_light_fastvit.test \
  --config experiments_graphormer_dexycb_light_fastvit/configs/dexycb_graphormer.yaml \
  --checkpoint /path/to/model.ckpt
```

也可在已激活的 `vit` 环境中使用 `scripts/run_train.sh` 和 `scripts/run_test.sh`。

## 数据审计与可视化

```bash
conda run -n vit python -m experiments_graphormer_dexycb_light_fastvit.audit_dataset \
  --config experiments_graphormer_dexycb_light_fastvit/configs/dexycb_graphormer.yaml

conda run -n vit python \
  experiments_graphormer_dexycb_light_fastvit/sample_visualization/visualize_random_samples.py \
  --setup s0 --split train --count 100 --seed 42
```

审计报告包含官方候选数、左右手候选数、过滤原因、subject/sequence/camera 分布、
split 路径交集和禁用目录检查。可视化每组保存原图裁剪框、未增强 crop、增强 crop、
crop 内手部分割和元数据 manifest。首次构建全量过滤缓存需要读取全部右手标注，之后
训练和审计直接复用缓存。

本地原始数据按默认 `s0` 和当前过滤参数审计后的数量为：train `187185`、val
`10314`、test `37223`。三者路径及 sequence 交集均为 0。审计中的各过滤原因是可
重叠诊断值（同一帧可能违反多项条件），不能直接相加作为总拒绝数。
