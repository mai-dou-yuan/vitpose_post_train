# Multi-Scale Cross-Stage Attention 方案

## 1. 目标

在现有 `MeshRegressor` 主干不变的前提下，让最终的 778 个 vertex token 通过 Cross-Attention 动态读取 21 / 84 / 336 三个尺度的历史特征：

```text
[B,21,256] → [B,84,256] → [B,336,128] → [B,778,64] → [B,778,3]
```

只在 `upsample_3` 与 `pred_final` 之间增加融合模块，不改变 token 上采样路径和最终输出形状。

## 2. 历史特征的实际取值位置

当前实现位于 `mesh_regressor.py`。历史特征必须取自下面三个位置，避免把 encoder 输出通道写错：

```python
# Stage 1 输出
x21 = self.encoder_1(self.proj_1(joint_tokens + self.pos_emb_1))
# x21: [B, 21, 256]

# upsample_1 输出；进入 proj_2 前保存
x84 = self.upsample_1(x21)
# x84: [B, 84, 256]
x84_encoded = self.encoder_2(self.proj_2(x84 + self.pos_emb_2))
# x84_encoded: [B, 84, 128]

# upsample_2 输出；进入 proj_3 前保存
x336 = self.upsample_2(x84_encoded)
# x336: [B, 336, 128]
x336_encoded = self.encoder_3(self.proj_3(x336 + self.pos_emb_3))
# x336_encoded: [B, 336, 64]

x778 = self.upsample_3(x336_encoded)
# x778: [B, 778, 64]
```

其中 `x84`、`x336` 是对应尺度的上采样结果，而不是 `encoder_2`、`encoder_3` 的输出。

## 3. 模块设计

将三个历史尺度投影到 `d_attn=64`，加入可学习 Stage Embedding 后拼接：

```python
self.proj21 = nn.Linear(256, 64)
self.proj84 = nn.Linear(256, 64)
self.proj336 = nn.Linear(128, 64)

self.stage_embed21 = nn.Parameter(torch.zeros(1, 1, 64))
self.stage_embed84 = nn.Parameter(torch.zeros(1, 1, 64))
self.stage_embed336 = nn.Parameter(torch.zeros(1, 1, 64))
```

```text
21×256  → 21×64  ─┐
84×256  → 84×64  ─┼→ Concat → history: [B,441,64] → K,V
336×128 → 336×64 ─┘

x778: [B,778,64] → Q
```

Cross-Attention 使用 Pre-Norm、4 个 head 和标准残差；不增加 `gamma` 或 gate。随后接 `64 → 256 → 64` 的 Pre-Norm FFN，并再次残差相加。

在 `layers.py` 中新增模块：

```python
class MultiScaleCrossStageAttention(nn.Module):
    def __init__(self, d_attn=64, num_heads=4, ffn_ratio=4):
        super().__init__()
        if d_attn != 64:
            raise ValueError("d_attn must be 64 to match x778")

        self.proj21 = nn.Linear(256, d_attn)
        self.proj84 = nn.Linear(256, d_attn)
        self.proj336 = nn.Linear(128, d_attn)

        self.stage_embed21 = nn.Parameter(torch.zeros(1, 1, d_attn))
        self.stage_embed84 = nn.Parameter(torch.zeros(1, 1, d_attn))
        self.stage_embed336 = nn.Parameter(torch.zeros(1, 1, d_attn))

        self.norm_q = nn.LayerNorm(d_attn)
        self.norm_kv = nn.LayerNorm(d_attn)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_attn,
            num_heads=num_heads,
            batch_first=True,
        )

        hidden_dim = d_attn * ffn_ratio
        self.norm_ffn = nn.LayerNorm(d_attn)
        self.ffn = nn.Sequential(
            nn.Linear(d_attn, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_attn),
        )

    def forward(self, x21, x84, x336, x778):
        h21 = self.proj21(x21) + self.stage_embed21
        h84 = self.proj84(x84) + self.stage_embed84
        h336 = self.proj336(x336) + self.stage_embed336
        history = torch.cat([h21, h84, h336], dim=1)

        kv = self.norm_kv(history)
        cross_feat, _ = self.cross_attn(
            query=self.norm_q(x778),
            key=kv,
            value=kv,
            need_weights=False,
        )
        x778 = x778 + cross_feat
        return x778 + self.ffn(self.norm_ffn(x778))
```

## 4. 接入 `MeshRegressor`

在 `mesh_regressor.py` 中导入该模块，并在 `MeshRegressor.__init__` 中注册：

```python
self.cross_stage_attn = MultiScaleCrossStageAttention(
    d_attn=64,
    num_heads=4,
    ffn_ratio=4,
)
```

`forward` 按第 2 节保留 `x21`、`x84`、`x336`，并在输出层前融合：

```python
x778 = self.upsample_3(x336_encoded)
x778 = self.cross_stage_attn(
    x21=x21,
    x84=x84,
    x336=x336,
    x778=x778,
)
vertices = self.pred_final(x778)
```

最终数据流：

```text
21  ─┐
84  ─┼→ 投影 + Stage Embedding → 441-token History → K,V
336 ─┘
778 ───────────────────────────────────────────────→ Q
                              ↓
                    Cross-Attention + Residual
                              ↓
                         FFN + Residual
                              ↓
                         Linear 64→3
```

## 5. 权重加载兼容性

现有 `load_from_mesh_head` 和 `load_from_checkpoint` 会校验目标模型的全部参数。加入 `cross_stage_attn.*` 后，旧 simpleHand `MeshHead` 或旧 checkpoint 不包含这些 key，因此需要同步调整兼容加载逻辑：

- 原主干已有参数仍要求 key 存在且 shape 一致；
- 允许旧权重缺少且仅缺少 `cross_stage_attn.*`，这些参数保留新模块初始化值；
- 若 checkpoint 已包含 `cross_stage_attn.*`，则正常加载；
- 不要放宽其它参数的缺失或 shape mismatch 校验。

这样既能加载旧模型作为初始化，也能完整恢复新结构 checkpoint。

## 6. 验收

更新 `test_mesh_regressor.py`，至少覆盖：

1. 前向输出仍为 `[B,778,3]`，反向传播后新旧模块参数均有梯度；
2. 新结构 checkpoint 保存后可等价恢复；
3. 删除 `cross_stage_attn.*` key 模拟旧 checkpoint 时可以加载，且其它缺失或 shape 不匹配仍报错。

运行：

```bash
python -m pytest -q mesh_regressor_package/test_mesh_regressor.py
```

核心思路不变：由 778 个 vertex token 作为 Query，通过 Attention 动态选择 21 / 84 / 336 三个尺度的历史信息。
