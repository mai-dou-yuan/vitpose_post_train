# pl_system.py
import math
import os
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
# 鲈鱼
print("鲁钰")
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, bias=False):
        super().__init__()
        # 对应图中的 Gate 路径 和 Up 路径
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=bias)
        
        # 对应图中的 Down 路径
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)
        self.act = nn.SiLU() 

    def forward(self, x):
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))


class HandMultiHeadGATLayer(nn.Module):
    """
    结合了 Transformer 架构容量 (Multi-Head, FFN) 与 GAT 物理先验的满血版图注意力层。
    完全等价替换原有的 PoseSelfAttentionLayer。
    【更新】已引入 G1 同源输出门控机制，完美适配异构注意力的串行分组堆叠。
    【优化】使用加法分配律重构多头注意力计算，消除高维张量拼接，大幅降低显存开销。
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.1, alpha=0.2, num_joints=21):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
            
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        # ==========================================
        # 1. 骨架拓扑定义 (布尔型 Mask 优化)
        # ==========================================
        edges = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12),
            (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20)
        ]
        
        # 初始化布尔类型的 Mask，极大提升运行效率和安全性
        adj_mask = torch.zeros(num_joints, num_joints, dtype=torch.bool)
        for i, j in edges:
            adj_mask[i, j] = True
            adj_mask[j, i] = True
        for i in range(num_joints):
            adj_mask[i, i] = True # 自环
            
        self.register_buffer('adj_mask', adj_mask)

        # ==========================================
        # 2. Multi-Head GAT 分支 
        # ==========================================
        # 假设 RMSNorm 已在外部定义
        self.norm_attn_in = RMSNorm(d_model) # 建议后续替换为 ZeroCenteredRMSNorm
        
        # 特征变换矩阵
        self.W = nn.Linear(d_model, d_model, bias=False)
        
        # 【优化】将注意力权重拆分为 source (a_l) 和 target (a_r)
        self.a_l = nn.Parameter(torch.empty(1, 1, n_head, self.head_dim))
        self.a_r = nn.Parameter(torch.empty(1, 1, n_head, self.head_dim))
        nn.init.xavier_uniform_(self.a_l)
        nn.init.xavier_uniform_(self.a_r)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attn_dropout = nn.Dropout(dropout)

        # ===== 输出门控投影 (G1 Position 同源设计) =====
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)
        
        # 多头拼接后的输出映射 (Wo)
        self.linear_out = nn.Linear(d_model, d_model)

        # ==========================================
        # 3. FFN / MLP 分支 (复用 SwiGLU 保持容量对齐)
        # ==========================================
        self.norm_mlp_in = RMSNorm(d_model) # 建议后续替换为 ZeroCenteredRMSNorm
        # SwiGLU 已在外部定义
        self.mlp = SwiGLU(d_model, dim_feedforward) 
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None):
        """
        x:   [B, N, C]
        pos: [B, N, C]
        """
        B, N, C = x.size()
        
        # ==========================================
        # Block 1: Multi-Head Graph Attention
        # ==========================================
        residual = x
        
        # Norm 与 Positional Encoding 融合
        h_in = self.norm_attn_in(x)
        if pos is not None:
            h_in = h_in + pos
            
        gate_input = h_in 
        
        # 1. 线性映射并拆分多头 -> [B, N, n_head, head_dim]
        h = self.W(h_in).view(B, N, self.n_head, self.head_dim)
        
        # 2. 【优化】计算节点作为 source 和 target 的注意力特征得分
        # 形状变化: [B, N, n_head, head_dim] * [1, 1, n_head, head_dim] -> sum -> [B, N, n_head]
        e_l = (h * self.a_l).sum(dim=-1) 
        e_r = (h * self.a_r).sum(dim=-1) 
        
        # 3. 【优化】利用广播机制计算稠密注意力得分矩阵 e
        # [B, N, 1, n_head] + [B, 1, N, n_head] -> [B, N, N, n_head]
        e = e_l.unsqueeze(2) + e_r.unsqueeze(1) 
        
        # 缩放机制
        e = e / math.sqrt(self.head_dim)
        e = self.leakyrelu(e)
        
        # 将 head 维度提前 -> [B, n_head, N, N]
        e = e.permute(0, 3, 1, 2)
        
        # 4. Mask 操作 (FP16/BF16 混合精度安全的 Mask)
        # 【优化】使用 float('-inf') 替代 -1e4，确保 Softmax 后的值为纯 0
        e = e.masked_fill(~self.adj_mask, float('-inf'))
        
        # 5. Softmax & Dropout
        attention = F.softmax(e, dim=-1)
        attention = self.attn_dropout(attention)
        
        # 6. 聚合邻居节点特征
        h_v = h.permute(0, 2, 1, 3) # -> [B, n_head, N, head_dim]
        h_prime = torch.matmul(attention, h_v) # -> [B, n_head, N, head_dim]
        
        # 7. 拼接多头特征 -> [B, N, C]
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        
        # ===== 计算并应用 G1 位置门控 =====
        gate_score = self.gate_proj(gate_input)
        gate = torch.sigmoid(gate_score) 
        h_gated = h_prime * gate
        
        # 8. 输出映射
        out_attn = self.linear_out(h_gated)
        x = residual + self.attn_dropout(out_attn)

        # ==========================================
        # Block 2: FFN (SwiGLU) 分支
        # ==========================================
        residual = x
        
        mlp_in = self.norm_mlp_in(x)
        mlp_out = self.mlp(mlp_in)
        
        x = residual + self.mlp_dropout(mlp_out)
        
        return x

