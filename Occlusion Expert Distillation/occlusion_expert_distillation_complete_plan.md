# Occlusion Expert Distillation 完整方案

## 0. 方案目标

本方案用于提升手部 3D pose estimation 中 **self-occluded joints** 的预测精度。

核心思想是：

```text
先从 Base model 初始化一个 Occlusion Expert，
让 Expert 专门强化遮挡关节预测能力；

然后冻结 Expert，
再从 Base model 初始化 Student，
通过 Gaussian KL + joint token distillation，
把 Expert 的遮挡知识蒸馏给 Student；

最终只部署 Student。
```

整体流程：

```text
Base checkpoint
    ↓
初始化 Occlusion Expert
    ↓
Expert 针对 self-occluded joints 做遮挡特化训练
    ↓
冻结 Expert
    ↓
Base checkpoint 初始化 Student
    ↓
Student 使用 GNLL + stable diagonal Gaussian KL + joint token KD 蒸馏
    ↓
最终只部署 Student
```

---

## 1. 数据字段与遮挡定义

当前 `dataset.py` 已经返回以下字段：

```python
ret_data = {
    "img": img_tensor,
    "hand_back": hand_back_tensor,
    "origin_3d": torch.from_numpy(hand_central_3d).float(),
    "gt_pose": torch.from_numpy(pose_gt_3d).float(),
    "cam_k": torch.from_numpy(cam_k_new).float(),
    "dist_coeffs": dist_coeffs,
    "dataset_idx": idx,
    "gt_pose_2d": torch.from_numpy(gt_pose_2d).float(),
    "visibility_label": torch.from_numpy(visibility_label).long(),
    "in_view_mask": torch.from_numpy(in_view_mask).bool(),
    "visible_joint_ratio": torch.from_numpy(visible_ratio).float(),
}
```

因此训练时不需要额外从 dataset 返回：

```python
batch["occ_mask"]
```

遮挡 mask 在训练过程中动态构造即可。

---

## 2. Self-occlusion mask 构造

本方案只针对 **self-occlusion**，即：

```text
joint 仍在图像内，但由于手指、手掌或其他手部结构遮挡而不可见。
```

推荐定义：

```python
def build_self_occ_mask(batch):
    """
    构造 self-occlusion mask.

    Args:
        batch:
            visibility_label: [B, 21]
            in_view_mask: [B, 21]

    Returns:
        occ_mask: [B, 21], bool
    """
    visibility_label = batch["visibility_label"]
    in_view_mask = batch["in_view_mask"]

    # 假设 visibility_label == 0 表示 visible
    # visibility_label != 0 表示不可见 / 被遮挡
    occ_mask = (visibility_label != 0) & in_view_mask

    return occ_mask
```

不建议在第一阶段把 out-of-view joints 混入 self-occlusion mask。

原因：

```text
self-occlusion:
    joint 仍在图像内；
    可以通过图像上下文和手部结构进行推理。

out-of-view:
    joint 已经离开图像边界；
    视觉信息缺失更严重；
    和 self-occlusion 属于不同难度类型。
```

如果后续需要单独研究 out-of-view，可以额外定义：

```python
out_view_mask = ~in_view_mask
```

---

## 3. 模型 forward 输出修改

为了支持 Expert 蒸馏，模型 forward 需要返回以下内容：

```python
results["pose3d"]             # [B, 21, 3]
results["pose3d_logvar"]      # [B, 21, 3]
results["joint_token"]        # [B, 21, C]

results["all_stage_pose3d"]   # List[[B, 21, 3]]
results["all_stage_logvars"]  # List[[B, 21, 3]]
results["all_stage_tokens"]   # List[[B, 21, C]]
```

其中：

```text
pose3d:
    最后一层 refinement stage 的 3D 坐标均值。

pose3d_logvar:
    最后一层 refinement stage 的 diagonal log variance。

joint_token:
    最后一层 refinement stage 的 joint token。

all_stage_pose3d:
    每一个 refinement stage 的 3D 坐标均值。

all_stage_logvars:
    每一个 refinement stage 的 diagonal log variance。

all_stage_tokens:
    每一个 refinement stage 的 joint token。
```

### 3.1 forward 修改示意

在 refinement loop 前加入：

```python
all_stage_preds = []
all_stage_logvars = []
all_stage_tokens = []
```

在每个 refinement stage 内部保存 token 和预测结果：

```python
all_stage_tokens.append(curr_tokens)

raw_pred = self.pose_3d_head_PR(curr_tokens)
stage_pred_mu = raw_pred[..., :3]
stage_pred_logvar = raw_pred[..., 3:]

all_stage_preds.append(stage_pred_mu)
all_stage_logvars.append(stage_pred_logvar)
```

最后返回：

```python
results = {
    "pose3d": all_stage_preds[-1],
    "pose3d_logvar": all_stage_logvars[-1],
    "joint_token": curr_tokens,
    "all_stage_pose3d": all_stage_preds,
    "all_stage_logvars": all_stage_logvars,
    "all_stage_tokens": all_stage_tokens,
}
```

---

## 4. GNLL warmup 设置

Base model 从头训练时可以使用 GNLL warmup。

但是 Expert 和 Student 都是从已经训练好的 Base checkpoint 初始化，因此不需要重新 warmup。

推荐设置：

```text
Base 从头训练:
    gnll_warmup_epochs = 5

Expert 微调:
    gnll_warmup_epochs = 0

Student 蒸馏:
    gnll_warmup_epochs = 0
```

原因：

```text
1. Base checkpoint 中 logvar 分支已经训练过；
2. Expert 需要继续维护不确定性预测；
3. Student 的 Gaussian KL 蒸馏依赖 logvar；
4. 重新 warmup 会导致前几个 epoch 不训练 logvar。
```

### 4.1 PoseLightningModule 中加入参数

```python
class PoseLightningModule(pl.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        num_joints=21,
        local_model_dir=None,
        feature_dim=768,
        layers=[3, 6, -1],
        upsample_dim=512,
        num_refine_layers=3,
        use_gradient_checkpointing=True,
        gnll_warmup_epochs=5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.gnll_warmup_epochs = gnll_warmup_epochs
```

### 4.2 `_compute_gnll_loss` 修改

```python
def _compute_gnll_loss(
    self,
    pred_mu,
    pred_logvar,
    gt,
    warmup_epochs=None,
    beta=10.0,
):
    if warmup_epochs is None:
        warmup_epochs = self.gnll_warmup_epochs

    robust_dist = F.smooth_l1_loss(
        pred_mu,
        gt,
        reduction="none",
        beta=beta,
    )

    if self.current_epoch < warmup_epochs:
        return robust_dist.mean()

    pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
    precision = torch.exp(-pred_logvar)
    loss = precision * robust_dist + 0.5 * pred_logvar

    return loss.mean()
```

### 4.3 加载 Expert / Student 后关闭 GNLL warmup

```python
model.gnll_warmup_epochs = 0
model.hparams.gnll_warmup_epochs = 0
```

---

## 5. 通用 Masked GNLL

该函数用于 Expert 加权训练和 Student GT supervision。

```python
import torch
import torch.nn.functional as F


def compute_gnll_loss_masked(
    pred_mu,
    pred_logvar,
    gt,
    mask=None,
    current_epoch=0,
    warmup_epochs=0,
    beta=10.0,
    eps=1e-6,
):
    """
    Args:
        pred_mu: [B, 21, 3]
        pred_logvar: [B, 21, 3]
        gt: [B, 21, 3]
        mask: None or [B, 21]
    """
    robust_dist = F.smooth_l1_loss(
        pred_mu,
        gt,
        reduction="none",
        beta=beta,
    )  # [B, 21, 3]

    if current_epoch < warmup_epochs:
        loss = robust_dist
    else:
        pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
        precision = torch.exp(-pred_logvar)
        loss = precision * robust_dist + 0.5 * pred_logvar

    if mask is None:
        return loss.mean()

    mask_f = mask.float().unsqueeze(-1)  # [B, 21, 1]
    valid_count = mask_f.sum()

    if valid_count < 1:
        return pred_mu.sum() * 0.0

    return (loss * mask_f).sum() / (valid_count * pred_mu.shape[-1] + eps)
```

---

## 6. Stage 1：Base model

Base model 已经训练完成。

```text
Model:
pl_system_v6_graphormer_7_3.py

Checkpoint:
checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt
```

该 checkpoint 用于：

```text
1. 初始化 Occlusion Expert；
2. 初始化 Student。
```

---

## 7. Stage 2：训练 Occlusion Expert

### 7.1 初始化 Expert

```python
expert = PoseLightningModule.load_from_checkpoint(
    "checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt",
    strict=False,
)

expert.gnll_warmup_epochs = 0
expert.hparams.gnll_warmup_epochs = 0
```

Expert 不从头训练，而是从 Base checkpoint 微调。

---

## 8. Expert loss 设计

Expert 的目标是强化 self-occluded joints，同时避免完全遗忘 visible joints。

推荐使用互斥 mask 加权 GNLL：

```text
L_expert = w_occ * L_GNLL_occ + w_vis * L_GNLL_vis
```

其中：

```text
w_occ = 1.0
w_vis = 0.1
```

含义：

```text
遮挡 joint:
    使用较高权重，重点优化。

非遮挡 joint:
    使用较低权重，保留基本能力，防止灾难性遗忘。
```

---

## 9. Expert 加权 GNLL 函数

```python
def compute_expert_gnll_weighted(
    pred_mu,
    pred_logvar,
    gt,
    occ_mask,
    current_epoch=0,
    warmup_epochs=0,
    w_occ=1.0,
    w_vis=0.1,
    beta=10.0,
    eps=1e-6,
):
    """
    Expert 使用互斥 mask 加权的 GNLL.

    Args:
        pred_mu: [B, 21, 3]
        pred_logvar: [B, 21, 3]
        gt: [B, 21, 3]
        occ_mask: [B, 21]
    """
    robust_dist = F.smooth_l1_loss(
        pred_mu,
        gt,
        reduction="none",
        beta=beta,
    )

    if current_epoch < warmup_epochs:
        loss_matrix = robust_dist
    else:
        pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
        precision = torch.exp(-pred_logvar)
        loss_matrix = precision * robust_dist + 0.5 * pred_logvar

    occ_mask_f = occ_mask.float().unsqueeze(-1)
    vis_mask_f = 1.0 - occ_mask_f

    occ_count = occ_mask_f.sum()
    vis_count = vis_mask_f.sum()

    loss = pred_mu.sum() * 0.0

    if occ_count >= 1:
        loss_occ = (loss_matrix * occ_mask_f).sum() / (
            occ_count * pred_mu.shape[-1] + eps
        )
        loss = loss + w_occ * loss_occ

    if vis_count >= 1:
        loss_vis = (loss_matrix * vis_mask_f).sum() / (
            vis_count * pred_mu.shape[-1] + eps
        )
        loss = loss + w_vis * loss_vis

    return loss
```

---

## 10. Expert training_step

```python
def training_step_expert(self, batch, batch_idx):
    imgs = batch["img"]
    hand_back = batch["hand_back"]
    gt_pose = batch["gt_pose"]

    occ_mask = build_self_occ_mask(batch)

    results = self(imgs, hand_back)

    loss_expert = 0.0

    for pred_mu, pred_logvar in zip(
        results["all_stage_pose3d"],
        results["all_stage_logvars"],
    ):
        loss_expert = loss_expert + compute_expert_gnll_weighted(
            pred_mu=pred_mu,
            pred_logvar=pred_logvar,
            gt=gt_pose,
            occ_mask=occ_mask,
            current_epoch=self.current_epoch,
            warmup_epochs=0,
            w_occ=1.0,
            w_vis=0.1,
        )

    loss_expert = loss_expert / len(results["all_stage_pose3d"])

    with torch.no_grad():
        pred_pose_3d = results["pose3d"]
        joint_error = torch.norm(pred_pose_3d - gt_pose, dim=-1)

        occ_mask_f = occ_mask.float()
        vis_mask_f = 1.0 - occ_mask_f

        occ_count = occ_mask_f.sum()
        vis_count = vis_mask_f.sum()

        if occ_count >= 1:
            occ_mpjpe = (joint_error * occ_mask_f).sum() / (occ_count + 1e-6)
        else:
            occ_mpjpe = joint_error.sum() * 0.0

        if vis_count >= 1:
            vis_mpjpe = (joint_error * vis_mask_f).sum() / (vis_count + 1e-6)
        else:
            vis_mpjpe = joint_error.sum() * 0.0

        overall_mpjpe = joint_error.mean()

    self.log("train/loss_expert", loss_expert, prog_bar=True)
    self.log("train/mpjpe_3d", overall_mpjpe, prog_bar=True)
    self.log("train/occ_mpjpe_3d", occ_mpjpe, prog_bar=True)
    self.log("train/vis_mpjpe_3d", vis_mpjpe, prog_bar=True)

    return loss_expert
```

---

## 11. Expert 训练配置

```text
初始化:
Expert ← Base checkpoint

数据:
全部训练样本

遮挡定义:
occ_mask = (visibility_label != 0) & in_view_mask

GNLL warmup:
0

Loss:
L_expert = w_occ * L_GNLL_occ + w_vis * L_GNLL_vis

推荐:
w_occ = 1.0
w_vis = 0.1
lr = base lr × 0.1 ~ 0.3
epoch = 10 ~ 30
```

训练 Expert 后，需要保存 Expert checkpoint：

```text
checkpoints/occlusion_expert.ckpt
```

---

## 12. Stage 3：初始化 Student

Student 从 Base checkpoint 初始化，而不是从 Expert 初始化。

```python
student = PoseLightningModule.load_from_checkpoint(
    "checkpoints/pose-epoch=63-val_mpjpe_3d=15.9398.ckpt",
    strict=False,
)

student.gnll_warmup_epochs = 0
student.hparams.gnll_warmup_epochs = 0
```

原因：

```text
Expert 是遮挡偏置模型；
Student 是最终部署模型；
Student 需要保留 Base 的整体泛化能力，同时吸收 Expert 的遮挡知识。
```

---

## 13. Stage 4：冻结 Expert

加载训练好的 Expert：

```python
expert = PoseLightningModule.load_from_checkpoint(
    "checkpoints/occlusion_expert.ckpt",
    strict=False,
)

expert.eval()

for p in expert.parameters():
    p.requires_grad = False
```

Expert 前向必须使用：

```python
with torch.no_grad():
    expert_out = expert(imgs, hand_back)
```

---

## 14. Stable diagonal Gaussian KL

### 14.1 KL 方向

使用：

```text
KL(P_E || P_S)
```

也就是让 Student 的分布拟合 Expert 的分布。

其中：

```text
P_E = N(mu_E, diag(exp(logvar_E)))
P_S = N(mu_S, diag(exp(logvar_S)))
```

对于每个 joint 的 3D diagonal Gaussian：

```text
KL(P_E || P_S)
=
0.5 * sum_d [
    logvar_S,d - logvar_E,d
    + exp(logvar_E,d - logvar_S,d)
    + (mu_E,d - mu_S,d)^2 * exp(-logvar_S,d)
    - 1
]
```

### 14.2 稳定 KL 实现

```python
def diagonal_gaussian_kl_occ_stable(
    mu_e,
    logvar_e,
    mu_s,
    logvar_s,
    occ_mask,
    eps=1e-6,
):
    """
    Stable diagonal Gaussian KL on self-occluded joints.

    Args:
        mu_e: [B, 21, 3]
        logvar_e: [B, 21, 3]
        mu_s: [B, 21, 3]
        logvar_s: [B, 21, 3]
        occ_mask: [B, 21]
    """
    mu_e = mu_e.detach()
    logvar_e = logvar_e.detach()

    logvar_e = torch.clamp(logvar_e, min=-5.0, max=5.0)
    logvar_s_safe = torch.clamp(logvar_s, min=-5.0, max=5.0)

    term1 = logvar_s_safe - logvar_e
    term2 = torch.exp(
        torch.clamp(logvar_e - logvar_s_safe, min=-20.0, max=20.0)
    )
    term3 = (mu_e - mu_s).pow(2) * torch.exp(
        torch.clamp(-logvar_s_safe, min=-20.0, max=20.0)
    )

    kl_per_dim = 0.5 * (term1 + term2 + term3 - 1.0)
    kl_per_joint = kl_per_dim.sum(dim=-1)

    occ_mask_f = occ_mask.float()
    valid_count = occ_mask_f.sum()

    if valid_count < 1:
        return mu_s.sum() * 0.0

    return (kl_per_joint * occ_mask_f).sum() / (valid_count + eps)
```

---

## 15. Joint token distillation

### 15.1 Token KD 目标

Token KD 用于让 Student 的 joint token 在 self-occluded joints 上靠近 Expert 的 joint token。

第一版使用最后一层 joint token：

```python
student_out["joint_token"]
expert_out["joint_token"]
```

### 15.2 Token KD 实现

```python
def token_distill_occ_safe(
    token_s,
    token_e,
    occ_mask,
    eps=1e-6,
):
    """
    Last-stage joint token KD on self-occluded joints.

    Args:
        token_s: [B, 21, C]
        token_e: [B, 21, C]
        occ_mask: [B, 21]
    """
    token_e = token_e.detach()

    token_s = F.layer_norm(token_s, token_s.shape[-1:])
    token_e = F.layer_norm(token_e, token_e.shape[-1:])

    loss_per_joint = (token_s - token_e).pow(2).mean(dim=-1)

    occ_mask_f = occ_mask.float()
    valid_count = occ_mask_f.sum()

    if valid_count < 1:
        return token_s.sum() * 0.0

    return (loss_per_joint * occ_mask_f).sum() / (valid_count + eps)
```

---

## 16. Student loss

Student 总 loss：

```text
L_student =
    L_GNLL_all_stage
  + lambda_KL * L_KL_occ_all_stage
  + lambda_token * L_token_occ_last_stage
```

含义：

```text
L_GNLL_all_stage:
    Student 对所有 joints 使用 GT supervision。

L_KL_occ_all_stage:
    Student 在所有 refinement stage 上拟合 Expert 的遮挡关节高斯分布。

L_token_occ_last_stage:
    Student 在最后一层 joint token 上拟合 Expert 的遮挡关节 token 表征。
```

推荐权重：

```text
正式权重:
lambda_KL = 0.5
lambda_token = 0.2

前 5 epoch KD warmup:
lambda_KL = 0.1
lambda_token = 0.05
```

注意：

```text
KD loss 权重可以 warmup；
GNLL warmup 仍然保持 0。
```

---

## 17. Student distillation training_step

```python
def training_step_distill(self, batch, batch_idx):
    imgs = batch["img"]
    hand_back = batch["hand_back"]
    gt_pose = batch["gt_pose"]

    occ_mask = build_self_occ_mask(batch)

    student_out = self.student(imgs, hand_back)

    with torch.no_grad():
        expert_out = self.expert(imgs, hand_back)

    # 1. Student 原始 GT supervision: all-stage GNLL
    loss_unc = 0.0

    for pred_mu, pred_logvar in zip(
        student_out["all_stage_pose3d"],
        student_out["all_stage_logvars"],
    ):
        loss_unc = loss_unc + compute_gnll_loss_masked(
            pred_mu=pred_mu,
            pred_logvar=pred_logvar,
            gt=gt_pose,
            mask=None,
            current_epoch=self.current_epoch,
            warmup_epochs=0,
        )

    loss_unc = loss_unc / len(student_out["all_stage_pose3d"])

    # 2. All-stage stable Gaussian KL on self-occluded joints
    loss_kl = 0.0

    for mu_e, logvar_e, mu_s, logvar_s in zip(
        expert_out["all_stage_pose3d"],
        expert_out["all_stage_logvars"],
        student_out["all_stage_pose3d"],
        student_out["all_stage_logvars"],
    ):
        loss_kl = loss_kl + diagonal_gaussian_kl_occ_stable(
            mu_e=mu_e,
            logvar_e=logvar_e,
            mu_s=mu_s,
            logvar_s=logvar_s,
            occ_mask=occ_mask,
        )

    loss_kl = loss_kl / len(student_out["all_stage_pose3d"])

    # 3. Last-stage joint token KD
    loss_token = token_distill_occ_safe(
        token_s=student_out["joint_token"],
        token_e=expert_out["joint_token"],
        occ_mask=occ_mask,
    )

    lambda_kl = 0.5
    lambda_token = 0.2

    if self.current_epoch < 5:
        lambda_kl = 0.1
        lambda_token = 0.05

    loss = loss_unc + lambda_kl * loss_kl + lambda_token * loss_token

    with torch.no_grad():
        pred_pose_3d = student_out["pose3d"]
        joint_error = torch.norm(pred_pose_3d - gt_pose, dim=-1)

        occ_mask_f = occ_mask.float()
        vis_mask_f = 1.0 - occ_mask_f

        occ_count = occ_mask_f.sum()
        vis_count = vis_mask_f.sum()

        if occ_count >= 1:
            occ_mpjpe = (joint_error * occ_mask_f).sum() / (occ_count + 1e-6)
        else:
            occ_mpjpe = joint_error.sum() * 0.0

        if vis_count >= 1:
            vis_mpjpe = (joint_error * vis_mask_f).sum() / (vis_count + 1e-6)
        else:
            vis_mpjpe = joint_error.sum() * 0.0

        overall_mpjpe = joint_error.mean()

    self.log("train/loss", loss, prog_bar=True)
    self.log("train/loss_unc", loss_unc, prog_bar=True)
    self.log("train/loss_kl_occ", loss_kl, prog_bar=True)
    self.log("train/loss_token_occ", loss_token, prog_bar=True)
    self.log("train/mpjpe_3d", overall_mpjpe, prog_bar=True)
    self.log("train/occ_mpjpe_3d", occ_mpjpe, prog_bar=True)
    self.log("train/vis_mpjpe_3d", vis_mpjpe, prog_bar=True)

    return loss
```

---

## 18. 验证指标

建议基于 `visibility_label` 和 `in_view_mask` 分开统计。

```python
visibility_label = batch["visibility_label"]
in_view_mask = batch["in_view_mask"]

self_occ_mask = (visibility_label != 0) & in_view_mask
out_view_mask = ~in_view_mask
visible_mask = visibility_label == 0
```

推荐汇报：

```text
Overall MPJPE
Visible MPJPE
Self-occluded MPJPE
Out-of-view MPJPE
Fingertip MPJPE
Self-occluded Fingertip MPJPE
PCK@20
PCK@30
```

核心观察指标：

```text
1. Self-occluded MPJPE 是否下降；
2. Overall MPJPE 是否不明显变差；
3. Visible MPJPE 是否没有明显恶化；
4. Fingertip MPJPE 是否改善。
```

---

## 19. 推荐实验顺序

### 19.1 实验 A：只训练 Expert

目的：

```text
验证 Expert 是否真的提升 self-occluded joints。
```

设置：

```text
Expert ← Base checkpoint
gnll_warmup_epochs = 0
w_occ = 1.0
w_vis = 0.1
epoch = 10 ~ 30
```

观察：

```text
Self-occluded MPJPE
Visible MPJPE
Overall MPJPE
```

如果 Expert 的 self-occluded MPJPE 没有改善，不建议继续做 Student 蒸馏，需要先调整 Expert 训练权重或学习率。

---

### 19.2 实验 B：Student 只加 KL，不加 token KD

Student loss：

```text
L_student =
    L_GNLL_all_stage
  + lambda_KL * L_KL_occ_all_stage
```

配置：

```text
lambda_KL = 0.5

前 5 epoch:
lambda_KL = 0.1
```

目的：

```text
确认 Gaussian KL 蒸馏是否有效。
```

---

### 19.3 实验 C：Student 加 KL + token KD

Student loss：

```text
L_student =
    L_GNLL_all_stage
  + lambda_KL * L_KL_occ_all_stage
  + lambda_token * L_token_occ_last_stage
```

配置：

```text
lambda_KL = 0.5
lambda_token = 0.2

前 5 epoch:
lambda_KL = 0.1
lambda_token = 0.05
```

目的：

```text
验证最后一层 joint token KD 是否进一步提升遮挡关节。
```

---

### 19.4 实验 D：加入 all-stage token KD

在实验 C 稳定提升后，可以尝试：

```text
L_token_occ_all_stage
```

即对每个 refinement stage 的 joint token 都做 token KD。

该实验作为增强版，不作为第一版默认配置。

---

## 20. 推荐训练配置汇总

### 20.1 Expert

```text
初始化:
Base checkpoint

训练数据:
全部训练样本

遮挡定义:
occ_mask = (visibility_label != 0) & in_view_mask

GNLL warmup:
0

Loss:
L_expert = w_occ * L_GNLL_occ + w_vis * L_GNLL_vis

推荐参数:
w_occ = 1.0
w_vis = 0.1
lr = base lr × 0.1 ~ 0.3
epoch = 10 ~ 30
```

---

### 20.2 Student

```text
初始化:
Base checkpoint

Teacher:
Frozen Occlusion Expert

训练数据:
全部训练样本

遮挡定义:
occ_mask = (visibility_label != 0) & in_view_mask

GNLL warmup:
0

Loss:
L_student =
    L_GNLL_all_stage
  + lambda_KL * L_KL_occ_all_stage
  + lambda_token * L_token_occ_last_stage

推荐参数:
lambda_KL = 0.5
lambda_token = 0.2

前 5 epoch:
lambda_KL = 0.1
lambda_token = 0.05
```

---

## 21. 最终执行清单

```text
1. 修改 forward，返回 joint_token 和 all_stage_tokens。
2. 增加 build_self_occ_mask(batch)。
3. 增加 compute_gnll_loss_masked。
4. 增加 compute_expert_gnll_weighted。
5. 增加 diagonal_gaussian_kl_occ_stable。
6. 增加 token_distill_occ_safe。
7. Expert 从 Base checkpoint 初始化。
8. Expert 设置 gnll_warmup_epochs = 0。
9. Expert 使用互斥 mask 加权 GNLL 训练。
10. 保存 occlusion_expert.ckpt。
11. Student 从 Base checkpoint 初始化。
12. Student 设置 gnll_warmup_epochs = 0。
13. 加载并冻结 Expert。
14. Student 使用 all-stage GNLL + all-stage KL + last-stage token KD 蒸馏。
15. 验证 Self-occluded MPJPE、Visible MPJPE、Overall MPJPE 和 Fingertip MPJPE。
```

---

## 22. 最终部署

训练完成后，只部署 Student：

```text
部署模型:
Student checkpoint

不部署:
Occlusion Expert

推理阶段:
不需要 visibility_label
不需要 in_view_mask
不需要 occ_mask
不需要 Expert
不需要 KD loss
```

推理时只保留正常 forward：

```python
results = student(imgs, hand_back)
pred_pose_3d = results["pose3d"]
```

最终部署路径：

```text
image pair
    ↓
Student
    ↓
pose3d
```
