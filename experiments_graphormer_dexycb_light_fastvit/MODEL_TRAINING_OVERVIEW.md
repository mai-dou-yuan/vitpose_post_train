# DexYCB 手部姿态与网格模型说明

本文依据 `pl_system_v6_graphormer.py`、`lightning_module.py`，并结合当前训练入口与配置，概述实际执行的模型结构、损失和训练流程。以下默认设置指 `configs/dexycb_graphormer.yaml`；若运行时覆盖配置，应以实际参数为准。

## 1. 任务与输入输出

模型从 DexYCB 的手部 RGB 裁剪图像估计：

- 21 个以腕关节（joint 0）为原点的三维关节；
- 778 个同样以腕关节为原点的 MANO 网格顶点；
- 一个直接回归的初始二维关节位置，供第一阶段局部注意力使用。

训练 batch 至少包含 `img`、`gt_pose`（`[B,21,3]`）、`gt_pose_2d`（`[B,21,2]`）、`gt_vertices`（`[B,778,3]`）、相机内参 `cam_k` 和相机坐标系腕点 `origin_3d`。无效关节/顶点由显式 mask（若提供）与数值有限性共同判定。

## 2. 整体结构

整体数据流可写为：

```text
RGB crop
  -> ViTPose++-Base backbone
  -> 1x1 Conv 投影为 256 维空间特征（decoder memory）
  -> 初始 2D 坐标头 -----------------------> Stage 1 局部参考点
  -> 21 个可学习 joint tokens
       -> [Self-Attention -> Cross-Attention -> 回归头] x 3
       -> MeshRegressor -> 778 vertices -> MeshToJoints -> 21 joints
```

### 2.1 图像骨干与初始二维头

骨干实际为可训练的 **ViTPose++-Base** 适配器，而不是 FastViT。输入 RGB crop 的取值范围为 `[0,1]`；适配器内部按预训练模型规范 resize 至 `256x192` 并做 ImageNet normalization。骨干末端空间特征经 `1x1 Conv` 投影到默认的 256 通道，并加入二维正弦位置编码。

初始二维头不生成逐关节 heatmap。它先以 `1x1 Conv + depthwise 3x3 Conv` 压缩特征，再自适应池化至 `4x3`，最后通过 MLP 和 sigmoid 直接输出 `[B,21,2]` 的 crop 归一化坐标；坐标分别按原输入 crop 的宽、高还原为像素位置。

### 2.2 三阶段关节 token 细化

解码器以 21 个可学习 joint token 及其可学习位置向量为初始状态。默认三个 stage 均依次执行一次 token 自注意力和一次图像交叉注意力，主干维度为 256、8 个 attention heads，FFN 为 1024 维 SwiGLU，并采用 RMSNorm、残差连接和 dropout。

- Stage 1、2 的自注意力为 `HandGraphormerLayer`：在全局多头注意力中加入手骨架最短路径距离偏置与同手指偏置，并以可学习门控调制注意力输出。
- Stage 3 使用普通的门控多头自注意力，不加入图结构偏置。
- 交叉注意力按 `local, local, full` 循环。Stage 1 以初始二维头预测为中心，从特征图为每个关节双线性采样默认 `5x5` 邻域；Stage 2 将上一 stage 的腕相对三维预测加上真实相机坐标腕点，再通过 `cam_k` 投影得到局部参考点；Stage 3 对整张特征图做全局交叉注意力。

每个 stage 共用一个坐标头，输出每关节 3D 均值与 3 个 log-variance；三维均值减去其腕点后成为腕相对预测。每个 stage 也将 token 输入共享的 `MeshRegressor` 得到 778 个顶点，再由固定/封装的 `MeshToJoints` 映射为 21 个关节，并统一减去网格关节的腕点。

需要特别区分两条输出支路：逐 stage 的 token 坐标预测接受 2D/3D 深监督，并为后续局部注意力提供参考；模型对外的最终 `pose3d` 则是**最后 stage 的网格经 `MeshToJoints` 得到的关节**。网格分支只对最后 stage 顶点施加显式顶点损失。

### 2.3 MeshRegressor

`MeshRegressor` 是从 21 个关节 token 逐级生成稠密网格的 simpleHand 风格回归头，输入、输出分别为 `[B,21,256]` 和 `[B,778,3]`。默认结构为：

```text
21x256 joint tokens
  -> AttentionBlock -> token 线性上采样 21 -> 84
  -> 通道投影 256 -> 128 -> AttentionBlock -> 上采样 84 -> 336
  -> 通道投影 128 -> 64  -> AttentionBlock -> 上采样 336 -> 778
  -> 多尺度跨阶段注意力 -> Linear(64, 3) -> 778 个三维顶点
```

各尺度加入独立可学习位置编码。每个 `AttentionBlock` 默认包含 4-head self-attention、4 倍扩张的 GELU-MLP、LayerNorm、残差连接和 stochastic depth；“上采样”是沿 token/节点维度进行的可学习线性映射，而非图像插值。得到 778 个 64 维顶点 token 后，以它们为 query，对投影到 64 维并拼接的 21、84、336 三个历史尺度做 4-head cross-attention，再经残差 FFN 融合，最后逐 token 线性回归 xyz。

随后 `MeshToJoints` 使用 MANO joint regressor 从顶点计算基础关节，并补充指尖顶点、重排为 SimpleHand 的 21 关节顺序。该映射不含可训练参数；其 MANO 回归矩阵注册为模型 buffer。

## 3. 损失函数

设有效关节集合为 \(\mathcal V_J\)，有效顶点集合为 \(\mathcal V_M\)，stage 权重为

\[
\alpha_s=\frac{w_s}{\sum_t w_t}.
\]

默认 `stage_supervision_weights=[0.1,0.3,1.0]`，归一化后为 `[1/14, 3/14, 10/14]`，越后的 stage 权重越大。

1. **初始二维损失**：将 GT 像素坐标按 `(W-1,H-1)` 归一化，仅保留落在 `[0,1]` 内的有效点，对预测与 GT 计算逐坐标 L1，再对坐标及有效关节取平均：\(L_{init2D}\)。

2. **逐阶段三维关节损失**：GT 和预测均以腕点为原点。每个 stage 计算有效关节的欧氏距离均值（MPJPE），再做加权和：

   \[
   L_{J3D}=\sum_s\alpha_s\,\frac{1}{|\mathcal V_J|}
   \sum_{j\in\mathcal V_J}\|\hat{J}_{s,j}-J_j\|_2.
   \]

3. **逐阶段二维重投影损失**：训练时将各 stage 的腕相对预测加上 **GT 腕点**，用相机内参投影到 crop 像素坐标；预测与 GT 坐标均除以 `max(H,W)`，计算 `beta=1` 的 Smooth L1，先对 xy 平均，再对有效关节和 stage 加权：\(L_{J2D}\)。

4. **最终顶点损失**：最后 stage 的腕相对顶点与 `gt_vertices - gt_wrist` 之间计算逐坐标 L1，对 xyz 和有效顶点平均：\(L_V\)。

总损失为

\[
L=\lambda_{J3D}L_{J3D}+\lambda_{J2D}L_{J2D}
 +\lambda_{init2D}L_{init2D}+\lambda_VL_V.
\]

当前 YAML 配置取 `λ_J3D=1.0`、`λ_J2D=0.02`、`λ_init2D=0.1`、`λ_V=4.0`。类构造函数中 `λ_V` 的兜底默认值为 10.0，但训练入口会使用 YAML 中的 4.0。

代码中保留了基于 log-variance 的 `_compute_gnll_loss`，但当前 `DexYCBPoseLightningModule._compute_losses` **没有调用它**；因此 log-variance 目前不参与总损失，实际三维监督是上述 MPJPE。

## 4. 训练流程

1. 构建 DexYCB train 与 test 数据集。训练集启用裁剪抖动、尺度/旋转、颜色、退化和遮挡增强；test 不启用训练增强。默认 crop 为 `224x224`。
2. 前向传播：骨干提取特征，初始二维头产生 Stage 1 参考点，三个 stage 依次细化 token；训练时默认对 self-attention、cross-attention 和网格回归器使用 gradient checkpointing。
3. 计算四类损失并反向传播。优化器为 AdamW，weight decay 为 0.04；任务模块基础学习率为 `1e-4`，ViTPose 骨干为 `1e-5`。
4. 前 5 个 epoch 冻结骨干（包括保持其 BN 统计不变）；同时所有参数组的学习率在 5 个 epoch 内由基础值的 0.1% 线性升至基础值。之后解冻骨干，并以 cosine annealing 衰减至 `1e-6`。
5. 默认训练 260 epochs、batch size 64、单设备、FP32，梯度范数裁剪为 1.0。按 `test_mpjpe_3d` 越小越好保存 top-3 和 last checkpoint，并以 240 epochs patience 早停。

训练入口有一个需留意的评估约定：`trainer.fit` 的 validation dataloader 实际传入 **test split**，而 `validation_step` 也统一记录为 `test_*`。因此每个 epoch 的模型选择指标 `test_mpjpe_3d` 来自 test split，而不是配置中的 val split。

## 5. 评估指标

最终指标基于网格分支输出：

- `MPJPE`：腕相对 21 关节的平均欧氏误差；
- `PA-MPJPE`：逐样本相似 Procrustes 对齐后的关节误差；
- `MPVPE` / `PA-MPVPE`：顶点误差及 Procrustes 对齐后的顶点误差；
- 测试阶段另记录保持根点约束的刚体对齐 MPJPE，以及 N-MPJPE、xyz 轴 MAE、尺度比和骨长误差等诊断量。

上述误差沿用数据标注的三维坐标单位；实现中没有额外乘以 1000，解释数值时应先确认 DexYCB 数据管线采用米还是毫米。
