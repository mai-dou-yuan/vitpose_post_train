# FreiHAND Graphormer Joint Token 重设计分析

> 分析对象：`pl_system_v6_graphormer.py`、`lightning_module.py`、`Joint Token 设计.docx`、`outputs/lightning_logs/version_5/metrics.csv`。  
> 本文只给出分析、伪代码和实验设计，不包含任何模型代码、配置、日志或 Word 文档修改。

## 1. 执行摘要

本文用以下标签区分结论的证据等级：

- **[代码事实]**：可以从当前项目代码、配置或已有集成断言直接确认。
- **[日志事实]**：由 `version_5/metrics.csv` 的非空 epoch 级记录计算得到。
- **[数据事实]**：由当前 FreiHAND 文件和当前划分策略只读统计得到。
- **[推断]**：与事实一致，但尚不能证明因果。
- **[待验证假设]**：必须通过受控消融才能判断。

核心结论如下。

1. **当前模型并不是“FastViT + Graphormer”数据流，而是 ViTPose++-B 最后一层单尺度特征 + 三阶段 joint decoder。** **[代码事实]** 数据流为 `[B,3,224,224] -> [B,3,256,192] -> [B,768,16,12] -> [B,256,16,12]`；21 个 joint token 的布局始终是 `[B,21,256]`。每个阶段先做 joint self-attention，再让每个 joint 对全部 192 个图像 patch 做 dense cross-attention，随后用同一个回归头输出 `[B,21,6]`，其中前三维为 root-relative XYZ。

2. **现有日志明确显示泛化差距，但不能把它归因于 learnable joint token。** **[日志事实]** 最优验证 MPJPE 出现在 epoch 70：训练 epoch MPJPE 为 **8.996 mm**，验证 MPJPE 为 **15.680 mm**。epoch 120 时训练继续降至 **6.924 mm**，验证回升至 **16.258 mm**，最终差距为 **9.333 mm**。epoch 70 之后的 50 个 epoch 中，训练 MPJPE 全部低于 epoch 70，但没有一个验证 epoch 优于 epoch 70。**[推断]** 持续性过拟合可认为从约 epoch 70 开始；训练/验证差距更早在 epoch 10～20 已开始扩大。

3. **learnable token 的参数量本身很小，风险主要来自其表示作用，而不是直接的参数记忆容量。** **[代码事实]** `joint_tokens` 只有 5,376 个参数，连同 `joint_token_pos` 共 10,752 个，仅约占 6.95M 非-backbone decoder 参数的 **0.155%**。因此“这 10k 参数记住了训练集”不是当前最有力解释。**[待验证假设]** 更合理的风险是：两套图像无关的关节向量同时提供稳定的关节身份、拓扑索引和平均姿态起点，使 decoder 在图像证据不足时仍可依赖姿态先验；是否真的如此，应先做冻结、去 positional embedding 和图像置零/打乱测试。

4. **当前 dense cross-attention 的主要问题是重复的全局自由度，而不是注意力矩阵本身特别大。** **[代码事实]** 每个 cross-refinement layer 约 1.115M 参数，三层约 3.346M；其中 gated cross-attention 投影每层 327,936 参数，其余大部分是 1024 维 SwiGLU FFN。单样本单层注意力分数仅 `8×21×192=32,256` 个，但三阶段都允许每个关节重新连接全部 patch。**[推断]** 用局部采样替换 stage2/3 的价值主要是增加 2D 空间归纳偏置和降低重复全局自由度；参数量下降只是次要收益。

5. **Z 确实更难，但并非因为 Z 的数据数值跨度明显更大。** **[日志事实]** epoch 70 的验证 MAE 为 X=7.104 mm、Y=6.601 mm、Z=9.592 mm；Z 比 X 高 2.488 mm（35.0%），比 Y 高 2.991 mm（45.3%）。**[数据事实]** evaluation root-relative 标准差约为 X=63.319 mm、Y=49.869 mm、Z=62.201 mm，Z 与 X 的尺度相近。**[推断]** Z 的额外误差更符合单目深度歧义、尺度/旋转误差和局部手指深度结构共同造成，而不能仅用坐标量纲解释；也不能据此直接提高 Z loss 权重。

6. **首选是方案 D 的渐进式混合结构，而不是一次性把三层全改成多假设射线 token。** 推荐保留一次 stage1 全局 cross-attention，新增轻量 2D reference head 和全局图像条件化 token，stage2/3 改为 2D 引导的单尺度或三尺度 learnable-offset sampling，并继续用 Graphormer 传播关节结构。射线方向编码只作为后续可选增强；多深度候选会破坏当前固定 21-token Graphormer 偏置和 `N<=21` 假设，不应优先。

7. **最先做的低成本验证**是：

   - 不重训，分别把图像特征置零和在 batch 内打乱，测量预测对图像的依赖；
   - 分开冻结 `joint_tokens`、冻结两套 joint embedding，以及移除 `joint_token_pos`；
   - 只保留 stage1 dense cross-attention，比较三阶段全局交互是否必要。 

这些实验比直接实现复杂射线 token 更能回答“过拟合到底来自哪里”。

---

## 2. 当前模型的数据流与张量形状

### 2.1 从数据到 backbone

**[代码事实]** 当前配置的 dataset crop 为 `224×224`；dataset 返回：

- `img`: `[B,3,224,224]`，RGB，范围 `[0,1]`；
- `gt_pose`: `[B,21,3]`，FreiHAND 相机坐标，当前数据单位为米；
- `gt_pose_2d`: `[B,21,2]`，已经变换到 crop 坐标；
- `cam_k`: `[B,3,3]`，已经由 crop/resize 变换更新；
- `crop_center`: `[B,2]`、`crop_scale`: `[B]`、`crop_bbox`: `[B,4]`。

训练增强会对 crop 做中心扰动、尺度变化和相机 Z 轴旋转，并同步更新 3D pose、2D 点和 crop 内参。非增强路径使用：

\[
u_{crop}=s(u_{orig}-x_0),\qquad v_{crop}=s(v_{orig}-y_0),\qquad
K_{crop}=A_{crop}K_{orig}.
\]

`ViTPosePlusBBackbone.forward()` 又调用 checkpoint 对应的预处理，把方形 crop 非等比缩放为 `[B,3,256,192]` 并做 ImageNet normalization。ViTPose++-B 配置为 patch size 16、embed dim 768、12 层，因此最后特征为：

\[
F_{vit}\in\mathbb{R}^{B\times768\times16\times12}.
\]

这 192 个位置是最后一层 patch embedding，不包含 neck、heatmap head 或 2D keypoint head。项目中的 `verify_vitpose_integration.py` 也对 `(1,768,16,12)` 和最终 `(1,21,3)` 写有运行时断言。

### 2.2 decoder 的实际张量流

| 节点 | 当前实际形状 | 来源/操作 | 说明 |
|---|---:|---|---|
| dataset image | `[B,3,224,224]` | `FreiHANDExperimentDataset` | 方形手部 crop |
| ViTPose 输入 | `[B,3,256,192]` | backbone `preprocess()` | crop 被缩放为预训练分辨率 |
| `feature_map` | `[B,768,16,12]` | ViTPose++-B | 单尺度最后层特征 |
| `global_feature_map` | `[B,256,16,12]` | `Conv2d(768,256,1)` | decoder memory；名称“global”不代表已池化 |
| `pos_embed_map` | `[B,256,16,12]` | `PositionEmbeddingSine` | 图像 memory 的固定 2D sine encoding |
| flattened memory | `[B,192,256]` | `permute+flatten` | cross-attention 的 K/V 序列 |
| `joint_tokens` 参数 | `[1,21,256]` | learnable parameter | 每幅图共享同一初始化 |
| `curr_tokens` | `[B,21,256]` | batch expand | self/cross-attention 内容 |
| `joint_token_pos` 参数 | `[1,21,256]` | learnable parameter | 每幅图共享的 query position/identity |
| `query_pos` | `[B,21,256]` | batch expand | self-attention 的 Q/K 和 cross-attention 的 Q 使用 |
| stage raw output | `[B,21,6]` | shared `Pose3DRegressionHead` | `XYZ + logvar_xyz` |
| stage 3D output | `[B,21,3]` | 前 3 维后减 wrist | joint 0 被强制变为 0 |

主 forward 可以概括为：

```python
# x: [B, 3, 224, 224]
feature_map = vitmodel(x)                       # [B, 768, 16, 12]
memory_map = backbone_projection(feature_map)  # [B, 256, 16, 12]
memory_pos = pos_embed_layer(memory_map)        # [B, 256, 16, 12]

q = joint_tokens.expand(B, 21, 256)            # [B, 21, 256]
q_pos = joint_token_pos.expand(B, 21, 256)      # [B, 21, 256]

for stage in range(3):
    q = layers_sa[stage](q, q_pos)              # [B, 21, 256]
    q = layers_ca[stage](q, memory_map, q_pos, memory_pos)
    raw = shared_pose_head(q)                   # [B, 21, 6]
    xyz = raw[..., :3] - raw[:, 0:1, :3]        # [B, 21, 3]
```

### 2.3 self-attention、cross-attention 与关节身份的调用位置

**[代码事实]** 共 3 个 refinement stage：

| stage | joint 间交互 | 图像交互 | 预测头 |
|---|---|---|---|
| stage1 | `HandGraphormerLayer`：最短路径偏置 + 同指偏置 | `PoseRefinementLayer`：21 query 对 192 memory dense CA | 共享 `pose_3d_head_PR` |
| stage2 | `HandGraphormerLayer`：同上 | 另一套独立参数的 dense CA | 同一个共享 head |
| stage3 | 普通 `PoseSelfAttentionLayer` | 第三套独立参数的 dense CA | 同一个共享 head |

前两个 Graphormer 的图结构 bias 以 21 关节固定骨架构造，初始为零；第三层不使用图距离 bias。三阶段都把同一个 `query_pos` 加到 self-attention 的 Q/K，并在 cross-attention 中加入 query；memory sine position 只加到 K，V 是未加位置编码的 memory 内容。

因此当前 joint token 与图像的交互具有以下性质：

- joint content 与 joint position 都是图像无关的 learnable 参数；
- stage1 接收图像后，后续 token 内容已经图像条件化；
- stage2/3 仍重新对全部 192 patch 做全局检索；
- 没有显式 2D reference、局部 crop feature、offset 或可见性置信度；
- stage 间传递 token，但不显式传递 2D/3D reference；每个 stage 回归完整坐标而不是代码层面的坐标残差。

### 2.4 当前监督的实际含义

**[代码事实]** `FreiHANDPoseLightningModule` 使用归一化权重：

\[
(w_1,w_2,w_3)=\frac{(0.1,0.3,1.0)}{1.4}
=(0.0714,0.2143,0.7143).
\]

3D 主损失为每个 stage 的 root-relative MPJPE 加权和：

\[
L_{deep3D}=\sum_{s=1}^{3}w_s\frac{1}{N}\sum_j
\lVert \hat p_j^{(s)}-p_j^*\rVert_2.
\]

2D 辅助损失不是独立 2D head。代码先做：

```python
pred_camera_coordinates = pred_root_relative + gt_wrist
pred_2d = project(pred_camera_coordinates, cam_k)
```

再与 crop 内 `gt_pose_2d` 计算 normalized Smooth L1，并以 `0.02` 加入总损失。其含义是利用 **GT absolute wrist** 仅在损失计算中恢复相机坐标。它可以约束 root-relative 3D 的投影一致性，但不能在推理时产生 2D reference，也不能直接作为射线 token 的 2D 来源。

回归头的后三维 `logvar` 当前被返回但没有进入 FreiHAND wrapper 的训练/验证损失；`_compute_gnll_loss()` 也没有在 wrapper 中调用。因此现有 uncertainty 输出实际上没有被监督。

### 2.5 参数与计算量定位

在 `C=256, N=21, S=16×12=192` 下：

| 部分 | 参数量 |
|---|---:|
| `joint_tokens` | 5,376 |
| `joint_token_pos` | 5,376 |
| 两套 joint embedding 合计 | 10,752 |
| 单个 gated cross-attention 投影 | 327,936 |
| 单个完整 `PoseRefinementLayer`（含 FFN/norm） | 1,115,392 |
| 三个 cross-refinement layer | 3,346,176 |
| 非-backbone decoder 合计 | 6,950,072 |

单层 dense CA 的注意力 score 数量为 `8×21×192=32,256`/样本。包含 Q/K/V、gate、output projection、两次 attention matmul 和 FFN 后，约为 47.9M MAC/样本/层，其中约 16.5M 来自 SwiGLU FFN。**[推断]** 在 ViTPose++-B 面前这并非压倒性的计算瓶颈；重设计的首要理由应是空间归纳偏置和泛化，而不是声称当前 192-token attention “不可承受”。

---

## 3. Version 5 日志的定量分析

### 3.1 统计口径

**[日志事实]** CSV 有 1,473 行、30 列，epoch 范围为 0～120，共 121 个验证 epoch：

- `train_mpjpe_3d` 等有 1,231 个非空值，是按日志间隔记录的 step 级值；
- `train_mpjpe_3d_epoch`、三个 `train_stage*_mpjpe_3d_epoch` 各有 121 个非空值；
- 所有 `val_*` 目标指标各有 121 个非空值。

因此本文用 `train_mpjpe_3d_epoch` 与 epoch 级验证值比较，避免把某个训练 batch 的 step 值当作整轮均值。米到毫米的换算为 `×1000`。

### 3.2 训练过程与最优 epoch

| epoch | train MPJPE (mm) | val MPJPE (mm) | gap=val-train (mm) | val stage1/2/3 (mm) |
|---:|---:|---:|---:|---:|
| 0 | 117.903 | 95.996 | -21.907 | 117.210 / 101.142 / 95.996 |
| 10 | 19.657 | 20.097 | 0.439 | 22.224 / 20.185 / 20.097 |
| 20 | 14.774 | 18.289 | 3.515 | 20.015 / 18.444 / 18.289 |
| 30 | 12.760 | 17.186 | 4.426 | 18.405 / 17.338 / 17.186 |
| 40 | 11.510 | 17.186 | 5.676 | 18.500 / 17.508 / 17.186 |
| 50 | 10.402 | 16.850 | 6.448 | 18.055 / 17.064 / 16.850 |
| 60 | 9.577 | 16.871 | 7.293 | 18.044 / 17.152 / 16.871 |
| **70** | **8.996** | **15.680** | **6.684** | **16.993 / 16.051 / 15.680** |
| 80 | 8.371 | 16.436 | 8.065 | 17.598 / 16.678 / 16.436 |
| 100 | 7.569 | 15.960 | 8.391 | 17.151 / 16.264 / 15.960 |
| 120 | 6.924 | 16.258 | 9.333 | 17.048 / 16.454 / 16.258 |

早期 val 小于 train 并不代表验证更容易：训练指标在启用数据增强、dropout/stochastic depth 的训练模式下计算，而验证使用 eval 模式和不同 split，两者不是完全同分布的无噪声对照。

### 3.3 最优点与训练终点

| 指标 | epoch 70（最佳 val MPJPE） | epoch 120（终点） | 解释 |
|---|---:|---:|---|
| train MPJPE | 8.996 mm | 6.924 mm | 继续改善 23.0% |
| val MPJPE | **15.680 mm** | 16.258 mm | 恶化 0.578 mm / 3.68% |
| val-train gap | 6.684 mm | **9.333 mm** | 终点 val 为 train 的 2.348 倍 |
| train PA-MPJPE | 5.692 mm | 4.633 mm | 训练拟合继续增强 |
| val PA-MPJPE | 6.552 mm | 6.233 mm | 对齐后指标晚期仍略改善 |
| val N-MPJPE | 11.695 mm | 11.503 mm | scale 对齐后晚期仍略改善 |
| val scale ratio | 0.912 | 0.898 | 预测 RMS 尺度约低 8.8%～10.2% |
| val bone length error | 3.803 mm | 4.065 mm | 晚期骨长误差变差 |
| val 2D aux loss | 0.001744 | 0.001716 | 数值略降 |
| weighted 2D / 3D ratio | 0.002123 | 0.002045 | 2D 加权项约占 3D 项 0.2% |

val PA-MPJPE 的全局最小值为 6.194 mm（epoch 112），val N-MPJPE 最小值为 11.381 mm（epoch 115）；两者与 raw MPJPE 的最佳 epoch 不一致。**[推断]** 晚期可能仍在改善尺度/旋转对齐后的姿态形状，同时 raw camera-axis pose、尺度或局部骨长泛化没有同步改善。仅凭这些聚合指标不能把差异定位到某个手指或全局旋转。

### 3.4 X/Y/Z 误差

| 时点 | X MAE | Y MAE | Z MAE | Z-X | Z 相对 X | Z-Y | Z 相对 Y |
|---|---:|---:|---:|---:|---:|---:|---:|
| epoch 70 | 7.104 mm | 6.601 mm | **9.592 mm** | +2.488 mm | +35.0% | +2.991 mm | +45.3% |
| epoch 120 | 7.358 mm | 6.816 mm | **9.976 mm** | +2.618 mm | +35.6% | +3.160 mm | +46.4% |

各轴自己的日志最小值发生在不同 epoch：X=6.805 mm（epoch 55）、Y=6.334 mm（epoch 53）、Z=9.340 mm（epoch 100）。这说明优化各轴存在折中，不能用某个轴的单独最优拼成一个不存在的模型结果。

**[数据事实]** 按当前 grouped background sampling 的 32,560 个训练 pose group 和 3,960 个 evaluation 样本统计 root-relative 坐标：

| split | std X/Y/Z (mm) | mean abs X/Y/Z (mm) | wrist absolute Z mean ± std |
|---|---:|---:|---:|
| training unique pose | 67.327 / 49.073 / 66.232 | 56.372 / 41.262 / 55.593 | 0.685 ± 0.120 m |
| evaluation | 63.319 / 49.869 / 62.201 | 51.802 / 47.590 / 50.631 | 0.712 ± 0.116 m |

epoch 70 的轴 MAE 除以 evaluation 轴标准差，约为 X=11.2%、Y=13.2%、Z=15.4%。因此：

- Z 与 X 的 GT root-relative 数值尺度相近，Z MAE 较高不能简单归结为 Z 数据跨度更大；
- Y 的数据跨度较小，直接按 inverse variance 加权反而会优先加大 Y，而不是 Z；
- evaluation 的 absolute wrist depth 均值比 training 高约 27 mm，但主 3D 输出是 root-relative，absolute depth shift 不会直接出现在 MPJPE 中；它仍可能通过成像尺度和外观形成分布差异；
- 当前日志无法区分 Z 误差来自 wrist 全局深度、相对深度、全局尺度还是局部手指结构。主输出已经减 wrist，所以 raw MPJPE 不含 absolute wrist translation；但尺度、相对深度和相机轴方向误差仍在。

### 3.5 三个 stage 是否形成有效 refinement

最佳点与终点的训练/验证 stage 对照为：

| 时点 | train stage1/2/3 (mm) | val stage1/2/3 (mm) |
|---|---:|---:|
| epoch 70 | 12.303 / 9.464 / 8.996 | 16.993 / 16.051 / 15.680 |
| epoch 120 | 10.224 / 7.397 / 6.924 | 17.048 / 16.454 / 16.258 |

**[日志事实]** epoch 70：

- stage1 -> stage2：16.993 -> 16.051 mm，改善 0.941 mm / 5.54%；
- stage2 -> stage3：16.051 -> 15.680 mm，改善 0.371 mm / 2.31%；
- stage1 -> stage3：总改善 1.312 mm / 7.72%。

epoch 120：

- stage1 -> stage2 改善 0.594 mm / 3.48%；
- stage2 -> stage3 改善 0.196 mm / 1.19%；
- stage1 -> stage3 总改善 0.790 mm / 4.63%。

121 个 epoch 中，stage2 在 **121/121** 个 epoch 优于 stage1；stage3 在 **116/121** 个 epoch 优于 stage2，5 个例外都发生在早期 epoch 2、5、7、9、14。全程平均 stage1->2、stage2->3 改善分别约 1.220 mm、0.271 mm。

训练侧的 refinement 幅度明显更大：epoch 70 的 train stage1->3 改善 3.307 mm / 26.9%，epoch 120 改善 3.299 mm / 32.3%。**[推断]** 后两阶段确实学到了 refinement，但训练收益远大于验证收益也提示 refinement 容量可能参与泛化差距；这仍需 stage1-only CA/局部采样消融验证，不能据此归因于某一层。

**结论：** refinement 是有效的，不能因为过拟合就删除后两阶段；但 stage3 的边际收益明显小于 stage2，且晚期边际收益缩小，因此 stage2/3 是把重复 dense CA 换成有空间约束的局部采样的合理位置。

### 3.6 过拟合开始阶段

应区分两种表述：

- **泛化差距开始扩大：约 epoch 10～20。** gap 从 epoch 10 的 0.439 mm 增至 epoch 20 的 3.515 mm，之后总体继续扩大。训练/验证 split 和训练模式差异意味着 gap 不能单独证明模型开始过拟合，但它是早期信号。
- **持续性验证过拟合：约 epoch 70。** epoch 70 是 raw val MPJPE 的全局最佳点；其后 50 个 epoch 训练全部更低，但验证没有一次刷新最佳，最终训练再改善 23.0% 而验证恶化 3.68%。配置的 early-stop patience 为 50，训练在 epoch 120 结束也与此一致。

这是单个 seed 的相关性证据，不能说明 learnable token 是因果来源。

---

## 4. Learnable joint token 是否导致过拟合

### 4.1 可以确认的事实

1. `joint_tokens` 和 `joint_token_pos` 对所有图像相同，各自为 `[1,21,256]`。
2. 两者都按 joint index 区分，能够承载 wrist、各指骨节等关节身份。
3. `joint_tokens` 进入 value/content 路径；`joint_token_pos` 反复进入 Q/K，二者存在表达关节身份的功能重叠。
4. 这两套参数合计只有 10,752 个，占非-backbone decoder 约 0.155%。
5. decoder 仍会在每个 stage 从图像 memory 接收内容，不能把当前模型描述为“只靠 token 预测”。

### 4.2 三种不同风险必须分开

**A. 参数记忆能力。** **[推断]** 10.8k 参数相对 backbone 和约 6.95M decoder 很小，单纯参数量造成训练集记忆的可能性低于 backbone/多层 decoder。但小参数也可作为稳定索引，不等于完全无影响。

**B. 关节身份与平均姿态先验。** **[待验证假设]** 每个 joint 的固定内容和 position 可以在没有图像证据时经过 Graphormer + shared head 形成固定的平均手型；三阶段结构偏置进一步强化这一先验。它可能帮助遮挡样本，也可能使模型在训练集姿态分布上走捷径。

**C. 图像—关节交互自由度。** **[待验证假设]** 每阶段独立 dense CA 允许 21 个 joint 在 192 个 patch 中重复选择，stage2/3 没有显式 2D anchor 限制。它可能过拟合背景、crop 模式或训练增强伪特征；也可能只是必要的全局搜索。需要 stage1-only CA 和局部采样消融判断。

### 4.3 其他同样重要的解释

| 候选来源 | 当前证据 | 不能直接得出的结论 |
|---|---|---|
| backbone 容量/微调 | ViTPose++-B 为 12×768；冻结 5 epoch 后以 `1e-5` 微调 | 仅凭日志不能说 backbone 是主因 |
| refinement 容量 | 三套 SA + 三套 CA；非-backbone decoder 约 6.95M | stage2/3 验证上仍有真实收益，不能直接删掉 |
| dense CA 自由度 | 每层每 joint 看全部 192 patch | 192 patch 并不大，不能只按 FLOPs 判定“冗余” |
| 深监督 | 三 stage 都有 3D 和投影 2D 监督 | 深监督常提升可训练性，也可能加强中间层拟合；需消融 |
| loss 权重 | 2D 加权贡献仅约 0.2%；logvar 未监督 | 不能说 Z 高仅因为没有单独 Z loss |
| 训练时长 | best=70，训练到 120 | 后 50 epoch 明确扩大泛化差距 |
| 数据与分布 | 每轮 32,560 pose group；val 是 3,960 evaluation 样本；root depth/轴分布略不同 | gap 同时包含过拟合和 train/evaluation distribution shift |
| 增强策略 | train 有 crop、旋转、颜色、退化和遮挡，val 无 | 早期 train/val 数值不完全可直接比较 |

### 4.4 当前最可靠的判断

**[推断]** 完全 learnable joint token “可能让模型更多依赖姿态先验”，但现有日志不支持“它已经造成过拟合”的因果结论。由于参数量极小，优先验证的不是简单删 token，而是：

1. 模型在图像置零/打乱后还剩多少性能；
2. 固定 content、固定 position、移除 position 分别有什么影响；
3. stage2/3 是否需要重复全局 CA。

---

## 5. 方案 A：全局特征条件化 Joint Token

### 5.1 推荐构造

保留一套轻量 joint identity，但把初始内容改为图像条件化：

\[
g=\operatorname{GAP}(F)\in\mathbb{R}^{B\times256},
\]

\[
\Delta q_j=\operatorname{MLP}([e_j^{id},g]),\qquad
\alpha_j=\sigma(\operatorname{Gate}([e_j^{id},g])),
\]

\[
q_j^0=e_j^{id}+\alpha_j\odot\Delta q_j.
\]

实际布局保持 batch-first：

```python
# memory_map: [B, 256, 16, 12]
global_context = memory_map.mean(dim=(-2, -1))       # [B, 256]
joint_id = joint_id_embedding.expand(B, 21, 256)     # [B, 21, 256]
g = global_context[:, None, :].expand(B, 21, 256)    # [B, 21, 256]

condition = torch.cat([joint_id, g], dim=-1)         # [B, 21, 512]
delta = condition_mlp(condition)                     # [B, 21, 256]
gate = torch.sigmoid(condition_gate(condition))      # [B, 21, 1] 或 [B,21,256]
joint_tokens = joint_id + gate * delta               # [B, 21, 256]
```

为了避免 `joint_tokens` 和 `joint_token_pos` 重复表达身份，第一版建议：

- 保留一套 `joint_id_embedding` 作为内容 seed；
- `query_pos` 使用同一 identity 的规范化/投影版本，或仅依赖 Graphormer 固定拓扑 bias；
- 不再保留两套完全独立的 256 维 joint 参数。

### 5.2 FiLM 和低秩备选

**FiLM：**

\[
q_j^0=(1+\tanh\gamma(g))\odot e_j^{id}+\beta(g).
\]

它最稳定、参数少，但所有 joint 共享同一 `γ/β`，joint-specific 条件化主要来自 `e_id`。

**低秩 joint-specific 调制：** 先把 `e_id` 和 `g` 投影到 `r=16/32` 的低秩空间，再做双线性交互。它比 512->256 的直接 MLP 更省参数，并限制 conditioner 自己学习完整姿态模板的能力。

**零初始化门控：** 把 `delta` 最后一层或 gate bias 初始化为接近 0，使模型初始等价于 identity baseline，再逐步学会使用图像条件。这对已有 pretrained backbone 更稳。

### 5.3 参数量、优缺点与风险

- 当前两套 embedding：10,752 参数。
- 256->64->512 的共享 FiLM MLP约 49.7k 参数；保留一套 identity 后净增加约 44k。
- 512->64->256 的 delta MLP约 49.5k；直接 scalar gate 约 0.5k，若复用 64 维 bottleneck 生成 256 维 gate 则再增加约 16.6k，仍远小于一个 1.115M cross layer。

优点：

- 初始化从第一层开始依赖当前图像，而不是等到第一次 CA 后才条件化；
- 仍保留关节身份和 Graphormer 拓扑；
- 不需要 2D 标签或相机内参；
- 与当前 `[B,21,256]` 接口完全兼容。

局限与风险：

- global average pooling 缺少局部定位，不能替代 2D reference；
- conditioner 增加的容量也可能学习训练集的全局外观/姿态相关性，并不天然防过拟合；
- 对小手指、遮挡关节，单一全局向量可能把所有 joint 调制得过于相似。

建议用 bottleneck、dropout、zero-init gate，并监控图像打乱后的性能下降是否变大，以确认图像依赖确实增强。

### 5.4 未来实现位置

- `PoseLightningModule.__init__()`：把 `joint_tokens`/`joint_token_pos` 替换为 `joint_id_embedding`、`GlobalTokenConditioner`。
- `PoseLightningModule.forward()`：在初始化 `curr_tokens` 前由 `global_feature_map` 生成 `global_context`。
- `HandGraphormerLayer` 和 stage head 第一版不需修改。

---

## 6. 方案 B：2D 引导的局部/多尺度 Deformable Sampling

### 6.1 2D reference 从哪里来

当前 3D stage output 是 root-relative，推理时没有 absolute wrist depth，不能可靠地直接投影成 2D。当前 2D aux 之所以能投影，是训练时加了 GT wrist；这条路径不能用于推理。因此建议新增独立轻量 2D head：

```python
# memory_map: [B, 256, 16, 12]
heatmaps = heatmap_head(memory_map)             # 例如 [B, 21, 64, 48]
reference_points = soft_argmax_2d(heatmaps)     # [B, 21, 2], 归一化到 [0,1]
confidence = heatmap_confidence(heatmaps)       # [B, 21, 1]
```

FreiHAND 当前 batch 已有 crop 内 `gt_pose_2d`，所以可直接监督：

\[
u_n=u/(W_{crop}-1),\qquad v_n=v/(H_{crop}-1).
\]

使用归一化 reference 能自动适配 `[16,12]` 及不同 pyramid level；不要在 sampler 内混用 224 crop、192×256 backbone 输入和 feature index 三套坐标。

### 6.2 当前单尺度 backbone 的轻量金字塔

当前只返回最后层 `[B,768,16,12]`，不应在第一版强行改 ViTPose 内部 block 输出。可在投影后的 `[B,256,16,12]` 上构造伪金字塔：

| level | 建议形状 | 构造方式 | 作用与限制 |
|---|---:|---|---|
| P2 | `[B,128,32,24]` | bilinear upsample + depthwise 3×3 | 更细采样网格，但不能恢复已丢失的浅层纹理 |
| P3 | `[B,128,16,12]` | 1×1 projection | 保留最后层语义 |
| P4 | `[B,128,8,6]` | stride-2 depthwise/pointwise conv | 扩大全局感受野 |

第一轮也可只用 P3 做单尺度 learnable-offset sampling；验证有效后再加 P2/P4。真正的多层语义金字塔需要 ViTPose adapter 暴露中间 block feature，侵入性更高，不应把对最后层简单 resize 宣称为“拥有浅层纹理的 FPN”。

### 6.3 三类采样必须明确区分

1. **固定坐标双线性采样：** `grid_sample(F, reference_points)`；坐标固定、无 offset、无多点权重。这只是 sampling primitive。
2. **learnable-offset 局部采样：** 每个 joint 预测 K 个 offset 和权重，在一个或少数 feature map 上聚合。
3. **多尺度 deformable attention：** 每个 head、level、point 都有 reference-relative offset 与归一化 attention weight，并对多 level value 做聚合。只有第 3 类才是完整的多尺度 deformable attention；其底层可以使用 `grid_sample` 实现。

推荐布局（`h=8, L=3, K=4`）：

```python
# q: [B, 21, 256], refs: [B, 21, 2]
offsets = offset_head(norm(q))
offsets = offsets.view(B, 21, 8, 3, 4, 2)       # [B,N,h,L,K,2]

weights = weight_head(norm(q))
weights = weights.view(B, 21, 8, 3, 4)          # [B,N,h,L,K]
weights = weights.flatten(-2).softmax(-1).view_as(weights)

sampled = multi_scale_bilinear_sample(
    pyramid_features,                              # L 个 [B,C_l,H_l,W_l]
    reference_points,                             # [B,21,2]
    offsets,                                      # level 尺度归一化后的 offset
)                                                 # [B,21,h,L,K,d_head]

local_features = (sampled * weights[..., None]).sum(dim=(3, 4))
local_features = local_features.flatten(2)         # [B,21,256]
q = q + output_proj(local_features)                # [B,21,256]
```

reference 还可逐 stage 更新：

```python
delta_uv = reference_update(q)                     # [B,21,2]
reference_points = sigmoid(inverse_sigmoid(reference_points) + delta_uv)
```

### 6.4 替换全部还是部分 cross-attention

**推荐：stage1 保留 dense CA，stage2/3 替换为局部 sampler。** 原因：

- 初始 2D reference 可能偏离真实关节；stage1 全局检索可为粗姿态和 token 内容提供兜底；
- 日志证明 stage2/3 refinement 有效，不应删除，只改变它们访问图像的方式；
- stage3 当前边际收益约 0.2～0.4 mm，适合使用更强空间约束；
- 全部替换会让训练早期过度依赖尚未收敛的 2D head，容易出现采样落在背景后无法恢复。

可在训练早期对 offset 半径设置较大上限，随后按 confidence 收缩；但这属于待验证设计，不能假定一定提升。

### 6.5 参数、计算和显存

标准 `C=256,h=8,L=3,K=4` deformable attention 的 value/offset/weight/output projection 约 0.206M 参数，低于当前 gated MHA 的 0.328M；若保留同样的 1024 维 FFN，完整 layer 只降低约 0.12M，而不是数量级下降。用 `C_local=128` 和更小 FFN 才会有明显参数收益。

attention 聚合位置从每 head 的 192 个 patch 降为 `L×K=12`，score/weight 数量从 32,256 降到 `8×21×12=2,016`，约 16 倍稀疏；但多尺度 feature activation 会增加显存。上述三层 128-channel pyramid 每样本约 129k feature elements，而当前 256×16×12 为 49k，因此总显存未必下降。第一轮单尺度 sampler 更适合先验证机制。

### 6.6 未来实现位置

- `PoseLightningModule.__init__()`：新增 `ReferencePointHead`、`LightFeaturePyramid`、stage2/3 `DeformableJointSampler`。
- `PoseLightningModule.forward()`：生成 reference/pyramid；stage1 走 `layers_ca[0]`，stage2/3 走 sampler。
- `FreiHANDPoseLightningModule.training_step()/validation_step()`：新增独立 `L_ref`，记录 2D reference error/PCK。
- 当前 `PoseRefinementLayer` 可保留为 stage1，不必原地改造成两种职责混合的类。

---

## 7. 方案 C：2D 点到射线 Token

### 7.1 当前项目是否具备射线条件

**[代码事实]** FreiHAND batch 已提供每样本 `cam_k [B,3,3]`，且 dataset 在 crop/resize 后更新内参。因此对于当前数据，最稳妥的做法不是先恢复原图坐标，而是直接在 crop 坐标中用 `K_crop`：

\[
\tilde r_j=K_{crop}^{-1}[u_j,v_j,1]^T.
\]

若把 ray direction 归一化：

\[
r_j=\tilde r_j/\lVert\tilde r_j\rVert_2,
\]

则 `d_j r_j` 中的 `d_j` 是沿射线的欧氏距离。若不归一化，`\tilde r_j` 的第三维为 1，则 `z_j\tilde r_j` 中的 `z_j` 是相机 Z 深度。两种参数化不能混用。

### 7.2 crop、backbone 和原图坐标

2D head 若输出 `[0,1]` 归一化坐标，可先变换回 224 crop：

\[
u_{crop}=u_n(W_{crop}-1),\quad v_{crop}=v_n(H_{crop}-1).
\]

再与 dataset 返回的 `K_crop` 一起构造射线。不要直接把 `[16,12]` feature index 代入 `K_crop`。

若确实要恢复非增强原图坐标：

\[
x_0=c_x-\frac{s_{crop}}{2},\quad y_0=c_y-\frac{s_{crop}}{2},
\]

\[
u_{orig}=u_{crop}/s_{resize}+x_0,\quad
v_{orig}=v_{crop}/s_{resize}+y_0.
\]

当前 dataset 返回 `crop_center` 和 `crop_scale`，可重建普通 crop 变换；但训练旋转路径的完整 `image_h`/angle 没有返回，不能从 batch 精确恢复增强前原图点。射线训练无需这一步，因为旋转后的 pose、图像和 `K_crop` 已在同一增强相机坐标约定中。如果未来必须输出原图坐标，应让 dataset 额外返回完整 3×3 homography 及原始 K。

ViTPose 内部还把 224×224 拉伸到 192×256。若 2D head 直接定义在这个坐标系，应同步用：

\[
K_{vit}=\operatorname{diag}(192/224,256/224,1)K_{crop}.
\]

更推荐让 head 输出归一化坐标，最终统一回 224 crop 再构造 ray。

### 7.3 2D 点来源与 token 融合

2D 点应来自方案 B 的独立 2D head，而不是由 root-relative 3D stage output 加 GT wrist 后投影。融合方式：

```python
# all batch-first
uv_norm = reference_head(memory_map)                # [B,21,2]
uv_crop = denormalize_to_crop(uv_norm)              # [B,21,2]
uv1 = torch.cat([uv_crop, ones], dim=-1)            # [B,21,3]
ray = solve(cam_k, uv1)                              # [B,21,3]
ray = F.normalize(ray, dim=-1)

local = deformable_sampler(pyramid, uv_norm, q)      # [B,21,256]
token = (
    joint_id                                          # [B,21,256]
    + global_proj(global_context)[:, None, :]         # [B,1,256]
    + uv_encoder(uv_norm)                             # [B,21,256]
    + ray_encoder(ray)                                # [B,21,256]
    + local_proj(local)                               # [B,21,256]
)
```

`uv`、ray 和 local feature 应作为可分离分支保留，最好用 gate/LN 后融合，不要只把所有项直接相加后丢失 reference。

### 7.4 root-relative 3D 下的几何约束

**关键限制：仅有 2D 点不能唯一恢复深度。** ray token 只把每个 joint 的可行位置从整个 3D 空间约束到一条射线，仍需图像、骨长、手型和数据先验选择射线上的位置。

对于绝对相机坐标，若使用单位 ray：

\[
X_j=d_jr_j.
\]

root-relative 输出应为：

\[
p_j=X_j-X_0=d_jr_j-d_0r_0.
\]

因此简单写成 `p_j=Δd_j r_j` 一般不成立，因为 wrist 与 joint 的 ray 方向不同。可选两条路线：

**软几何编码（优先）：** 把 `ray_encoder(ray)` 当作 token 特征，仍直接回归 root-relative XYZ。它不需要预测 absolute root depth，兼容当前输出，但几何只是一种提示而非硬约束。

**硬射线重建（后续）：** 预测 wrist distance/depth `d0` 与每 joint 的 `Δd_j`：

\[
d_j=d_0+\Delta d_j,\qquad
\hat p_j=(d_0+\Delta d_j)r_j-d_0r_0.
\]

当前数据有 absolute wrist GT，可增加 root-depth auxiliary supervision；推理时必须由模型自己预测 `d0`，不能复用训练损失中的 GT wrist。若部署时没有可靠 K，还需预测焦距/尺度或退化为 normalized ray-like encoding。

### 7.5 缺少内参时的近似

若未来数据没有 K，可使用：

\[
\bar r=[2u_n-1,2v_n-1,1],\qquad r=\bar r/\lVert\bar r\rVert_2,
\]

并将 crop scale/aspect ratio、可学习 focal token 或 weak-perspective scale 一起输入。这只是归一化观察方向，不是物理相机射线；报告和实验命名应明确为 `normalized ray-like encoding`。

### 7.6 多深度候选为何不应优先

若每个 joint 保留 D 个候选，token 会从 `[B,21,256]` 变为 `[B,21,D,256]` 或 `[B,21D,256]`。当前 `HandGraphormerLayer` 的 spatial distance/same-finger bias 都是 `21×21`，并显式断言 `N<=21`。多候选需要：

- joint 内候选 mask/bias；
- joint 间拓扑 bias 扩展；
- candidate reduction/selection head；
- 显存和计算约按 D 增长；
- 防止多个候选塌缩到相同深度的监督。

**[推断]** 单射线软编码与当前项目兼容性高；多深度候选虽理论上更能表示歧义，但在当前代码中属于结构重写，不应作为第一优先级。

### 7.7 未来实现位置

- `PoseLightningModule.forward()` 需要新增 `cam_k` 参数；当前 wrapper 的 `self(imgs, imgs)` 应改为显式传 K。
- `FreiHANDPoseLightningModule.training_step()/validation_step()/test_step()` 负责传 `cam_k` 和新增 ray/root-depth loss。
- `PoseLightningModule.__init__()` 新增 `RayEncoder` 和可选 `RootDepthHead`。
- 多候选版本还需修改 `HandGraphormerLayer` 的 graph bias 和序列长度假设。
- 若必须恢复原图，未来修改 `dataset.py::__getitem__()` 返回完整 homography/original K；仅在 crop 空间工作时不需要。

---

## 8. 方案 D：推荐的混合结构

### 8.1 推荐结构

推荐的第一版混合结构为：

1. ViTPose++-B 输出 `[B,768,16,12]`，投影为 `[B,256,16,12]`；
2. 轻量 2D head 预测 `reference_points [B,21,2]` 和 confidence；
3. global pooled image context 生成 image-conditioned joint token；
4. stage1 使用 Graphormer + 一次 dense global CA，获得粗 3D；
5. stage2/3 使用 Graphormer/轻量 self-attention + reference-guided local sampler；
6. 保留当前 root-relative 3D 输出和 deep supervision；
7. ray direction 先作为可选 soft encoding，不在第一版强制预测多深度候选。

```python
# Backbone
vit_feature = vitmodel(images)                       # [B,768,16,12]
memory = backbone_projection(vit_feature)            # [B,256,16,12]
memory_pos = pos_embed_layer(memory)                  # [B,256,16,12]

# Explicit 2D evidence
reference_points, confidence = reference_head(memory)
# reference_points: [B,21,2], confidence: [B,21,1]

# Image-conditioned initialization
global_context = memory.mean((-2, -1))                # [B,256]
q = token_conditioner(joint_id, global_context)       # [B,21,256]

# Stage 1: global search
q = graphormer_stage1(q, joint_identity_pos)          # [B,21,256]
q = global_cross_attention(q, memory, memory_pos)      # [B,21,256]
pred_stage1 = root_center(pose_head(q)[..., :3])       # [B,21,3]

# Stage 2/3: spatially constrained refinement
pyramid = light_pyramid(memory)                       # list of [B,C_l,H_l,W_l]
all_preds = [pred_stage1]
for stage in (2, 3):
    q = graph_or_self_stage[stage](q, joint_identity_pos)
    local = deformable_sampler[stage](
        pyramid, reference_points, q, confidence
    )                                                 # [B,21,256]
    q = q + local_fusion[stage](local)                # [B,21,256]
    reference_points = update_reference(q, reference_points)
    pred = root_center(pose_head(q)[..., :3])          # [B,21,3]
    all_preds.append(pred)

# Optional, not core v1
q = q + ray_encoder(reference_points, cam_k)           # [B,21,256]
```

第一版应继续共享当前 pose head，保持参数和监督可比。若局部 refinement 明确有效，再测试 stage2/3 residual head：`p_s=p_{s-1}+Δp_s`；不要把“改交互方式”和“改预测参数化”放在同一首轮实验里。

### 8.2 为什么比每层 full dense CA 更适合当前项目

- stage1 保留全局搜索，降低错误 2D reference 导致不可恢复的风险；
- stage2/3 只需围绕已经较好的 joint hypothesis 找局部证据，dense 全图自由度的必要性更低；
- Graphormer 已提供固定手骨架关系，局部视觉采样与结构传播职责更清晰；
- 当前 stage2/3 验证增益真实存在，混合结构保留 refinement 而不是粗暴删层；
- 2D annotation 和 crop-adjusted K 已存在，新增 reference/ray 不需要改变数据源；
- 参数量预计与当前 decoder 同量级或略低，attention 聚合计算明显减少；
- 相比多深度 ray candidates，不破坏 21-token graph bias 和现有 checkpoint 结构的大部分接口。

### 8.3 训练稳定性措施

- 先训练/蒸馏 2D reference head，或前 5～10 epoch 用 GT 2D 加噪声做 scheduled teacher forcing；验证/测试始终只能用预测 reference；
- offset head 最后一层零初始化，使初始采样围绕 reference；
- stage2/3 local fusion 使用 residual gate，初始 gate 接近 0；
- 对 reference 做 `[0,1]` clamp/sigmoid，并记录 out-of-bound ratio；
- 低 confidence 时扩大半径或混入少量 pooled global feature，而不是直接创建 D 个深度候选；
- 第一轮只做单尺度 P3，确认因果后再加 P2/P4。

scheduled teacher forcing 必须在报告中单列训练/推理 gap，不能用验证时 GT reference 产生虚高结果。

---

## 9. Z 方向监督和损失设计

### 9.1 当前 `L_3D` 已包含 Z

当前 MPJPE：

\[
L_{3D}=\frac{1}{N}\sum_j
\sqrt{e_{x,j}^2+e_{y,j}^2+e_{z,j}^2}
\]

已经对 Z 产生梯度。额外 `L_z` 是重复监督，但重复并不一定无效：它会改变梯度方向，使 Z 在总目标中权重更高。风险是模型用牺牲 XY、骨长或整体尺度来换取 Z MAE 下降。

定义 Smooth L1：

\[
\rho_\beta(e)=
\begin{cases}
\frac{e^2}{2\beta},& |e|<\beta\\
|e|-\frac{\beta}{2},& \text{otherwise}.
\end{cases}
\]

### 9.2 七类候选监督

| 方法 | 建议公式 | 初始权重 | 评价 |
|---|---|---:|---|
| 1. 直接 `L_z` | `mean(abs(z-z*))` | `λ_z=0.05` | 最简单；与 MPJPE 重复，适合单独小步消融 |
| 2. Z Smooth L1/Huber | `mean(ρ_0.005m(z-z*))` | `λ_z=0.05` | 比 L1 平滑、比 L2 抗 outlier；首选的轻量 Z 项 |
| 3. 相对深度 | wrist：`ρ((z_j-z_0)-(z*_j-z*_0))`；或 bone edge depth difference | `λ_relz=0.05` | 当前输出已 wrist-relative，wrist 形式几乎重复；edge 形式更强调局部层次 |
| 4. 骨长/方向 | `mean | ||b||-||b*|| |`；`mean(1-cos(b,b*))` | `λ_len=0.05`；方向项用约 `0.001 m` 等效系数 | 同时约束 XYZ，防止只优化 Z；应与 Z 项配套观察 |
| 5. 沿射线深度/距离 | `ρ(d0-d0*) + mean ρ(Δd-Δd*)` | `λ_ray=0.02` | 只在方案 C 的 hard ray/root-depth head 中使用；不能无 K 硬套 |
| 6. 异方差 NLL | `mean(exp(-s)ρ(e)+0.5s)` | 作为替代目标，或 `λ_unc=0.05` 辅助 | 当前 head 已有 logvar 但未监督；可能学会给困难 Z 更大方差，而非降低 Z |
| 7. XYZ 自适应/归一化 | `Σ_a w_a ρ(e_a)`，`w` 按 train σ、梯度或不确定性归一并限幅 | 平均权重=1，clip `[0.5,2]` | 数据 σ 会优先提高 Y，不支持盲目提高 Z；动态方法要防任务权重漂移 |

### 9.3 建议的初始总损失

第一轮不要同时打开所有项。建议在方案 D 的基础上从：

\[
L_{total}=
\sum_{s=1}^{3}w_sL_{MPJPE}^{(s)}
+0.02\sum_{s=1}^{3}w_sL_{reproj2D}^{(s)}
+0.10L_{ref2D}
+0.05L_{z\_Huber}^{(3)}
+0.05L_{bone}^{(3)}.
\]

其中：

- `L_ref2D` 是新 2D head 的 crop-normalized coordinate Smooth L1；其尺度与 3D 米单位不同，`0.10` 只是起点，应把加权贡献或 gradient norm 控制在总目标约 5%～15%，而不是固定迷信该系数；
- `L_z_Huber` 和 `L_bone` 先只作用最终 stage，避免把新增目标通过三层重复放大；
- 首轮 loss 消融应分别测试 `+L_z` 和 `+L_bone`，再测试二者组合，才能知道提升来自哪里；
- 若启用异方差，建议在坐标先稳定 10～20 epoch 后开启、clamp `logvar`，并把 NLL 作为一个独立实验，不和 adaptive XYZ 同时启用。

### 9.4 如何判断 Z 改进是否真实

必须同时监控：

- `val_mpjpe_3d`、`val_mae_x/y/z_3d`；
- stage1/2/3 MPJPE；
- `val_n_mpjpe_3d`、`val_pa_mpjpe_3d`；
- `abs(val_scale_ratio-1)`；
- bone length error；
- 新 2D reference pixel/normalized error、PCK、confidence calibration；
- 若用 hard ray：root depth MAE、relative ray-depth MAE。

判定建议：Z MAE 在至少 3 个 seed 上均值下降，同时 raw MPJPE 下降；X/Y 不得出现超过实验噪声的恶化，bone/scale 指标不恶化。可先用 0.2 mm 作为 X/Y 工程 guardrail，但最终应以 baseline 三 seed 标准差确定阈值。若 Z 下降但 PA-MPJPE、bone error 或 XY 明显变差，应视为坐标轴权衡而非整体进步。

---

## 10. 方案对比表

| 方案 | 图像依赖 | 关节身份先验 | 需独立 2D 监督 | 需 K | 参数变化 | 计算/显存 | 深度建模 | 稳定性 | 过拟合风险 | 实现复杂度 | 当前兼容性 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 当前 baseline | CA 后较强；初始化无图像 | 两套 256D embedding + graph | 否；仅 3D 投影 aux | 训练 2D aux 需要 | 基准；decoder 6.95M | 三次 dense CA；显存可控 | 隐式回归 | 已验证可训练 | 中：固定先验 + 重复全局自由度 | 低 | 最高 |
| A global-conditioned | 从 stage0 就依赖图像 | 保留一套 identity | 否 | 否 | 约 +0.04～0.08M | 几乎不变 | 间接、全局 | 高，zero-init 可平滑迁移 | 中低到中；conditioner 也会过拟合 | 低 | 很高 |
| B1 单尺度 offset sampling | 强、局部 | 保留 | 是 | 否 | 约持平，取决于 2D head/FFN | CA 聚合明显降；feature 显存近似 | 间接 | 中高 | 较低自由度，但依赖 2D 精度 | 中 | 高 |
| B2 三尺度 deformable | 强、局部+多尺度 | 保留 | 是 | 否 | 约持平或小增 | sample 聚合约 16×稀疏；pyramid 显存可能增加 | 间接 | 中；需 reference warmup | 中低 | 中高 | 高，单尺度验证后再做 |
| C1 soft ray encoding | 强，显式观察方向 | 保留 | 是 | FreiHAND 有 K；部署需确认 | 小增 `<0.1M` 量级 | 小增 | 中高，但不唯一确定深度 | 中高 | 中；可能依赖错误 2D/K | 中 | 高 |
| C2 hard ray + root depth | 强 | 保留 | 是 | 是 | 小到中增 | 中增 | 高，显式预测沿 ray 深度 | 中 | 中；root depth/scale 易不稳 | 高 | 中 |
| C3 多深度候选 | 强 | 需扩展 joint/candidate identity | 是 | 最好有 | 随 D 增长 | token/graph 计算和显存约随 D 增长 | 最高表达力但仍有歧义 | 较低 | 中高，候选可塌缩/记忆 | 很高 | 低；破坏 21-token 假设 |
| **D 混合结构** | **全局初始化 + 局部 refinement** | **保留单一 identity + graph** | **是** | core 不需；ray 可选 | **约持平或小增** | **仅一次 dense CA；后两层稀疏，显存取决于 pyramid** | **中高；可加 soft ray** | **高于全替换** | **中低，仍需消融确认** | **中高** | **最高的准确率/风险折中** |

---

## 11. 消融实验矩阵

### 11.1 公平比较规则

所有可用于结论的实验必须：

- 固定 FreiHAND train/evaluation 划分、grouped background sampling、augmentation；
- 固定 batch size、backbone checkpoint、freeze 5 epoch、backbone/task LR、scheduler、weight decay；
- 使用至少 3 个 seed，例如 42、123、2026；
- 使用相同 epoch budget，建议关闭 early stopping 后固定跑到 epoch 120；
- 同时报告每 seed 最佳值、固定 epoch 70、固定 epoch 120，以及三 seed 均值±标准差；
- 报告参数量、训练/推理吞吐和峰值显存；
- 不允许用一个实验训练 200 epoch、另一个 80 epoch，再只比较各自单次最优；
- 模块开发可先做 1-seed/10-20 epoch smoke test，但 smoke test 不用于因果结论。

### 11.2 必做矩阵

| # | 实验 | 验证的具体假设 | 控制变量/做法 | 重点指标 | 可能结果及解释 |
|---:|---|---|---|---|---|
| 1 | 冻结 learnable token | token 内容是否通过训练集拟合造成 gap | 1a 只冻 `joint_tokens`；1b 同时冻 content+pos；其余全同 | train/val gap、best/fixed MPJPE、图像打乱退化 | val 改善且 train 变差：固定先验可能过拟合；两者都变差：token 是有用 identity/优化 seed；无变化：不是主要瓶颈 |
| 2 | 移除 `joint_token_pos` | 两套 joint embedding 是否功能重复、position 是否承载过强先验 | pos 置零；保留 Graphormer bias 和 content | MPJPE、stage refinement、收敛速度 | val 提升：重复身份可能不利；stage1 明显变差：pos 对 query 区分必要 |
| 3 | shared token + joint identity | 分离共享姿态 seed 与关节身份是否优于每 joint 独立 content | `q_shared [1,1,C] + E_id [1,21,C]`；尽量匹配参数/初始化 | gap、图像依赖、PA/N-MPJPE | gap 缩小且 val 不降：独立 joint content 可能携带平均姿态；若性能掉，joint-specific seed 有价值 |
| 4 | global-conditioned token | 图像条件化是否减少固定先验依赖 | 只改初始化，不改 CA/stage/loss；zero-init gate | 正常/打乱图像差、best/fixed MPJPE | 正常提升且打乱退化更大：图像依赖增强；train/val 同时改善但 gap不变：主要是表达力提升，不证明抗过拟合 |
| 5 | 图像特征置零/打乱 | 模型究竟多依赖图像还是固定姿态先验 | 先对同一 checkpoint 做 inference-only；A: memory置零，B: batch内 shuffle；token 不变 | MPJPE、输出方差、与数据均值姿态距离 | 性能仍异常好/输出方差很小：强姿态先验；shuffle 大幅恶化：图像与样本匹配重要；zero 是 OOD，需结合 shuffle 解释 |
| 6 | 仅 stage1 global CA | 重复 dense CA 是否必要 | stage2/3 保留 SA/Graphormer，但 CA 设 identity；参数不匹配需同时报告 | stage1/2/3、FLOPs、gap | val 基本不变：后两次全局 CA 冗余；stage2/3 损失：仍需要图像更新，但不说明必须 dense |
| 7 | stage2/3 局部采样 | 显式 2D anchor 是否比重复 global CA 泛化好 | stage1 相同；先单尺度 K=4/8；同一 2D head | 3D、2D ref、遮挡切片、显存 | val 提升且 train 不显著提升：归纳偏置有效；两者下降：reference/sampler 尚不稳或全局上下文仍必要 |
| 8 | 轻量 Z loss | Z 是否可小权重改善而不伤 XY | 只加 final-stage `λ_z=0.05` Huber；其余不变 | XYZ MAE、MPJPE、bone、scale、PA | Z降且 MPJPE降、XY稳定：保留；Z降但 XY/bone变差：只是权衡，降权或放弃 |
| 9 | 单尺度 vs 多尺度 deformable | 收益来自 offset/locality 还是多尺度 | 相同 2D head、K、C、训练预算；B1=P3，B2=P2/P3/P4 | 3D/2D、small/occluded joint、显存/吞吐 | B1已覆盖大部分收益：无需复杂 pyramid；B2稳定额外提升：多尺度有效 |

### 11.3 建议补充的诊断切片

- 按 2D confidence/heatmap entropy 分组；
- 按遮挡增强是否触发分组（需在 dataset/日志中保存标记）；
- 指尖 vs MCP/wrist 关节分组；
- 按 wrist depth、crop scale、hand size 分桶；
- 按每根手指的 local Z/bone-depth error 分组。

这些切片可以区分“全局尺度错误”“相对深度错误”和“局部手指结构错误”，比只看 aggregate `val_mae_z_3d` 更有诊断力。

---

## 12. 分阶段实施路线

### Phase 1：低成本定位过拟合来源

1. 对现有 checkpoint 做图像 memory 置零与 batch shuffle，无需重训；
2. 运行 freeze content、freeze content+pos、remove pos 三个小改动消融；
3. 运行 stage1-only dense CA；
4. 固定至少 3 seeds 和 120 epoch；建立 baseline mean±std；
5. 增加每 joint/每轴误差、预测输出方差和 image-shuffle sensitivity 日志。

**进入下一阶段的条件：** 确认固定 token/position 确实降低图像依赖，或确认重复 dense CA 对验证收益有限。若证据不支持，也仍可做 A 作为表达优化，但不应宣称它“修复 token 过拟合”。

### Phase 2：Global-conditioned token

1. 保留一套 joint identity；
2. 加 bottleneck conditioner + zero-init gate；
3. 其他 stage、head 和 loss 全部不变；
4. 对比正常/打乱图像差、最佳和固定 epoch 结果。

这是实现成本最低、与当前接口最兼容的结构改动。

### Phase 3：2D reference + 局部采样

1. 新增独立 2D head，先验证 crop-normalized 2D error；
2. stage1 保留 dense CA；
3. stage2/3 先用单尺度 P3 learnable-offset sampler；
4. local fusion/offset zero-init，必要时短期 GT+noise teacher forcing；
5. 加 `L_ref2D`，但不要同时改变 3D regression 参数化。

### Phase 4：多尺度与射线

1. 单尺度有效后再加 P2/P4 伪金字塔；
2. 先加 soft ray direction encoding；
3. 只有在 per-joint Z 诊断显示沿 ray depth 是主要瓶颈时，再测试 root-depth + relative-depth head；
4. 多深度 candidates 放在最后，并单独重设计 Graphormer candidate bias/mask。

---

## 13. 最终推荐

### 首选方案

**方案 D：global-conditioned identity token + 独立 2D reference + stage1 一次 global CA + stage2/3 单尺度局部 sampling + Graphormer。**

先使用单尺度 P3 和 soft 2D reference，不立即引入完整三尺度、多深度候选或 hard ray reconstruction。该组合同时满足：

- 保留关节身份与手骨架先验；
- 从初始化开始引入当前图像信息；
- 保留日志已经证明有效的三阶段 refinement；
- 把后两阶段交互从无约束全局检索改为显式局部证据；
- 不破坏 `[B,21,256]` 和当前 Graphormer `21×21` bias；
- 为后续 ray encoding 留出清晰接口。

### 次选方案

**方案 A 单独落地。** 如果开发预算有限，先实现 global-conditioned token，并合并/简化两套 joint embedding。它不能解决局部定位和 Z 几何，但风险低、兼容性最高，适合在 Phase 1 后快速验证。

### 最低成本验证方案

1. 现有 checkpoint 的 image zero/shuffle dependency audit；
2. freeze token / freeze token+pos / remove pos；
3. stage1-only dense CA。

这三类实验能以最少代码回答固定先验和重复 global interaction 是否真是问题。

### 不建议优先采用

**多深度 ray token。** 原因不是理论无效，而是：

- 2D 点不能消除单目深度歧义；
- 当前 3D 是 root-relative，硬射线重建还需 root depth/scale；
- 当前 Graphormer 固定为 21 tokens，多候选会要求重写 bias、mask 和 reduction；
- 在尚未验证 2D head、局部采样和图像依赖前，复杂候选机制会混入过多变量，难以解释收益来源。

完整三尺度 deformable attention 也不应早于单尺度 offset sampling；否则无法判断收益来自 locality、learnable offset 还是多尺度 feature。

---

## 14. 风险、假设和待确认信息

1. **单 seed 限制。** `version_5` 只能证明该次训练在 epoch 70 后出现泛化分离，不能证明结构因果；所有推荐都需三 seed 验证。
2. **train/val 分布不是同源随机切分。** 当前 train 使用 FreiHAND training pose group，val 使用 evaluation 文件；absolute wrist depth、Y 轴分布和外观可能不同，gap 同时包含分布差异。
3. **训练指标与验证指标模式不同。** 训练开启增强、dropout 和 stochastic depth，因此早期 gap 的符号不可简单解释。
4. **2D aux 使用 GT wrist。** 它是训练损失技巧，不是可用于 inference reference/ray 的 2D 模块；未来必须新增独立 2D head 或同时预测 root depth。
5. **K 的部署可用性待确认。** FreiHAND 当前有 crop-adjusted K；若真实部署没有可靠内参，物理 ray 方案需降级为 normalized ray-like encoding 或增加相机参数预测。
6. **伪金字塔不等于真实多层特征。** 对最后层 map 上采样不会恢复浅层细节；若多尺度收益不足，再评估暴露 ViTPose 中间 block，而不是夸大 resize pyramid 的语义。
7. **坐标单位和 Huber beta。** 当前 FreiHAND XYZ 为米，文中 `β_z=0.005 m` 即 5 mm；若未来数据预处理改变单位，loss 超参必须同步换算。
8. **PA/N-MPJPE 的解释有限。** PA、scale-aligned 和 raw 指标的差距说明 scale/rotation 相关误差存在，但无法从聚合值单独定位到 Z 或某根手指。
9. **参数估算边界。** 当前模块参数量为按现有类实际维度计算的确定值；候选方案参数量是指定 bottleneck/head 配置下的量级估算，最终应由实现后 `sum(p.numel())` 和 profiler 复核。
10. **参考文档的取舍。** `Joint Token 设计.docx` 提出的 image condition、deformable sampling、ray、多候选和 bone/finger token 均有理论价值；本文没有直接照搬。基于当前单尺度 `[16,12]` memory、GT-wrist 2D 投影、固定 21-token graph bias 和日志中 stage3 的边际收益，本文优先选择“一次全局 + 两次局部”的渐进结构，并把多候选 ray 延后。

### 未来需要修改的类/函数汇总（本轮未修改）

| 目标 | 建议修改点 |
|---|---|
| A image-conditioned token | `PoseLightningModule.__init__()`、`forward()` |
| B 2D head/pyramid/sampler | `PoseLightningModule.__init__()`、`forward()`；新增职责单一的模块类 |
| B 的 2D 监督 | `FreiHANDPoseLightningModule.training_step()`、`validation_step()`、`test_step()` |
| C ray/K 传递 | `PoseLightningModule.forward(..., cam_k)`；wrapper 三个 step 显式传 K |
| C hard ray/root depth | 新 `RayEncoder`/`RootDepthHead` 和对应 loss/metric |
| C 多候选 | `HandGraphormerLayer` 的 `N<=21`、graph bias、candidate mask/reduction |
| 原图坐标恢复（仅若需要） | `dataset.py::__getitem__()` 返回完整 homography/original K |
| Z/bone/heteroscedastic loss | wrapper 的 loss 计算与日志；复用当前未启用的 logvar 输出前先做独立实验 |

### 证据来源

- `experiments_graphormer_freihand_light_fastvit/pl_system_v6_graphormer.py`
- `experiments_graphormer_freihand_light_fastvit/lightning_module.py`
- `experiments_graphormer_freihand_light_fastvit/Joint Token 设计.docx`
- `experiments_graphormer_freihand_light_fastvit/outputs/lightning_logs/version_5/metrics.csv`
- 辅助核对：`dataset.py`、`configs/freihand_graphormer.yaml`、`pretrain_vitpose_pkl_and_call/vitpose_plus_base_backbone_config.py`、`pretrain_vitpose_pkl_and_call/load_vitpose_plus_backbone.py`、`models/GatedMultiHeadAttention.py`、`models/Pose3DRegressionHead.py`、`utils/PositionEmbeddingSine.py`。
