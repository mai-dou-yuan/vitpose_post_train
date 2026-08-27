# pl_system.py
import math
import os
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.HandBackResidualNet import HandBackResidualNet
from models.model_PoseEstimationModel import PoseEstimationModel
from models.model_vit import ViTFeatureExtractor
from models.model_PoseEstimationModel import ViTPoseFusionBlock
from models.UpsampleHead4x import UpsampleHead4x
from models.GatedMultiHeadAttention import GatedMultiHeadAttention, MultiHeadAttention
from models.ChannelWiseGatedFusion import ChannelWiseGatedFusion
from utils.PositionEmbeddingSine import PositionEmbeddingSine
from models.Pose3DRegressionHead import Pose3DRegressionHead
from utils.project_3d_to_2d_batch_with_distortion import project_3d_to_2d_batch_with_distortion
from models.LightweightBackFeatureExtractor import LightweightBackFeatureExtractor
from torch.utils.checkpoint import checkpoint


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

class PoseRefinementLayer(nn.Module):
    """
    Cross Attention Layer: 
    Query = Joint Tokens
    Key/Value = Global Feature Map (Memory)
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.2):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
        
        self.norm_attn_in = RMSNorm(d_model)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)

        # self.cross_attention = GatedMultiHeadAttention(
        #     d_model=d_model, 
        #     n_head=n_head, 
        #     dropout=dropout
        # )
        self.cross_attention = MultiHeadAttention(
            d_model=d_model, 
            n_head=n_head, 
            dropout=dropout
        )
        
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, query_pos=None, memory_pos=None):
        """
        tgt:        [B, N, C]       - Joint Tokens
        memory:     [B, C, H, W]    - Global Feature Map
        query_pos:  [B, N, C]       - Joint Token Positional Encoding
        memory_pos: [B, C, H, W]    - Feature Map Positional Encoding
        """
        # ====== Block 1: Attention (Cross-Attention) ======
        tgt_norm = self.norm_attn_in(tgt)
        
        # Query Path
        q_content = self.norm_q(tgt_norm) 
        query = q_content + (query_pos if query_pos is not None else 0)
        
        # Key / Value Path
        if memory.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            memory_flatten = memory.permute(0, 2, 3, 1).flatten(1, 2)
            if memory_pos is not None:
                m_pos_flatten = memory_pos.permute(0, 2, 3, 1).flatten(1, 2)
            else:
                m_pos_flatten = 0
        else:
            memory_flatten = memory
            m_pos_flatten = memory_pos if memory_pos is not None else 0

        k_content = self.norm_k(memory_flatten) 
        key = k_content + m_pos_flatten
        value = memory_flatten
        
        attn_out = self.cross_attention(
            query=query,
            key=key,
            value=value,
            is_causal=False
        )
        
        tgt = tgt + self.dropout(attn_out)
        
        # ====== Block 2: MLP (SwiGLU) ======
        mlp_in = self.norm_mlp_in(tgt)
        mlp_out = self.mlp(mlp_in)
        tgt = tgt + self.dropout(mlp_out)
        
        return tgt

class PoseSelfAttentionLayer(nn.Module):
    """
    Self Attention Layer:
    Joint Tokens attend to themselves
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.2):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)

        self.norm_attn_in = RMSNorm(d_model)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)

        # self.self_attention = GatedMultiHeadAttention(
        #     d_model=d_model, 
        #     n_head=n_head, 
        #     dropout=dropout
        # )
        self.self_attention = MultiHeadAttention(
            d_model=d_model, 
            n_head=n_head, 
            dropout=dropout
        )
        
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None):
        """
        x:   [B, N, C] - Joint Tokens
        pos: [B, N, C] - Joint Token Positional Encoding
        """
        # Block 1: Self-Attention ---
        residual = x
        x = self.norm_attn_in(x)
        
        q = self.norm_q(x) + (pos if pos is not None else 0)
        k = self.norm_k(x) + (pos if pos is not None else 0)
        v = x
        
        attn_out = self.self_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False
        )
        x = residual + self.dropout(attn_out)

        # Block 2: MLP (SwiGLU) ---
        residual = x
        x = self.norm_mlp_in(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)
        
        return x

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

'''
class HandGraphormerLayer(nn.Module):
    """
    结合了 Graphormer 空间距离偏置、同指运动学偏置与门控机制的增强版全局注意力层。
    针对混合精度训练和早期训练稳定性进行了深度优化，并使用 PyTorch 原生 SDPA 加速。
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.1, num_joints=21):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
            
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        # ==========================================
        # 1. 结构偏置 A：最短路径距离 (Shortest Path Distance)
        # ==========================================
        edges = [
            (0,1), (1,2), (2,3), (3,4),       # Thumb
            (0,5), (5,6), (6,7), (7,8),       # Index
            (0,9), (9,10), (10,11), (11,12),  # Middle
            (0,13), (13,14), (14,15), (15,16),# Ring
            (0,17), (17,18), (18,19), (19,20) # Pinky
        ]
        
        spd = torch.full((num_joints, num_joints), float('inf'))
        for i in range(num_joints): spd[i, i] = 0
        for i, j in edges:
            spd[i, j] = 1
            spd[j, i] = 1
            
        for k in range(num_joints):
            for i in range(num_joints):
                for j in range(num_joints):
                    if spd[i, k] + spd[k, j] < spd[i, j]:
                        spd[i, j] = spd[i, k] + spd[k, j]
                        
        max_dist = int(spd.max().item()) 
        self.register_buffer('spatial_distance', spd.long())
        
        # 【优化】0 初始化，保证训练初期行为等价于标准 Self-Attention
        self.spatial_bias_table = nn.Embedding(max_dist + 1, n_head)
        nn.init.zeros_(self.spatial_bias_table.weight)

        # ==========================================
        # 2. 结构偏置 B：同指偏置 (Same-Finger Bias)
        # ==========================================
        fingers = [
            [1, 2, 3, 4],         # Thumb
            [5, 6, 7, 8],         # Index
            [9, 10, 11, 12],      # Middle
            [13, 14, 15, 16],     # Ring
            [17, 18, 19, 20]      # Pinky
        ]
        
        same_finger = torch.zeros((num_joints, num_joints), dtype=torch.long)
        for finger in fingers:
            for i in finger:
                for j in finger:
                    same_finger[i, j] = 1
        # 手腕 (0) 与所有手指不被视为"同指"
        for i in range(num_joints): same_finger[i, i] = 1 
        
        self.register_buffer('same_finger_index', same_finger)
        self.same_finger_bias = nn.Embedding(2, n_head)
        nn.init.zeros_(self.same_finger_bias.weight)

        # ==========================================
        # 3. Attention 分支
        # ==========================================
        self.norm_attn_in = RMSNorm(d_model) # 需确保外部已定义
        self.pos_scale = nn.Parameter(torch.tensor(1.0)) # 控制位置编码的强度
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        # 语义分离的 Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # ===== 输出门控投影 =====
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        # 初始 bias 从 2.0 降至 0.0，初始 Gate ≈ 0.5，更平稳
        nn.init.constant_(self.gate_proj.bias, 0.0) 
        
        self.linear_out = nn.Linear(d_model, d_model)

        # ==========================================
        # 4. FFN / MLP 分支
        # ==========================================
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward) # 需确保外部已定义
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None, attn_mask=None):
        """
        x:   [B, N, C]
        pos: [B, N, C]
        attn_mask: [B, N, N] 布尔型，True 代表被遮罩 (可选)
        """
        B, N, C = x.size()
        
        # 安全断言，预防未来添加 Token 时维度崩溃
        assert N <= self.spatial_distance.size(0), f"Sequence length N={N} exceeds maximum joint bias map size."
        
        residual = x
        
        # ==========================================
        # Block 1: Graphormer Global Attention
        # ==========================================
        h_in = self.norm_attn_in(x)
        if pos is not None:
            # 独立缩放，防止 pos 方差压倒内容特征
            h_in = h_in + self.pos_scale * pos
            
        gate_input = h_in 
        
        # 生成 Q, K, V
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        # shape 转换: [B, N, n_head, head_dim] -> [B, n_head, N, head_dim]
        q, k, v = map(lambda t: t.view(B, N, self.n_head, self.head_dim).transpose(1, 2), qkv) 
        
        # ----- 计算图结构偏置 -----
        # 获取偏置 (支持 N < 21 的切片操作)
        bias_spd = self.spatial_bias_table(self.spatial_distance[:N, :N]) 
        bias_finger = self.same_finger_bias(self.same_finger_index[:N, :N])
        
        # 合并偏置并调整形状: [N, N, n_head] -> [1, n_head, N, N]
        bias = (bias_spd + bias_finger).permute(2, 0, 1).unsqueeze(0)
        # bias = (bias_spd).permute(2, 0, 1).unsqueeze(0)
        
        # ----- 处理 Mask -----
        if attn_mask is not None:
            # attn_mask 形状: [B, N, N] -> [B, 1, N, N]
            # 你的约定是 True 代表被遮罩，因此将被遮罩位置替换为 -inf
            bias = bias.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        # ----- PyTorch SDPA 核心调用 -----
        # 使用 F.scaled_dot_product_attention 替代手动计算的 scores, softmax 和 matmul
        out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=bias, # 传入计算好的加性结构偏置 (内部会自动做加上 bias 的操作)
            dropout_p=self.attn_dropout.p if self.training else 0.0, # 动态控制 dropout
        )
        
        # 恢复维度: [B, n_head, N, head_dim] -> [B, N, n_head, head_dim] -> [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        # ----- 门控机制 -----
        gate = torch.sigmoid(self.gate_proj(gate_input)) 
        out_gated = out * gate
        
        out_attn = self.linear_out(out_gated)
        x = residual + self.proj_dropout(out_attn)

        # ==========================================
        # Block 2: FFN (SwiGLU)
        # ==========================================
        residual = x
        
        mlp_in = self.norm_mlp_in(x)
        mlp_out = self.mlp(mlp_in)
        
        x = residual + self.mlp_dropout(mlp_out)
        
        return x

'''
class HandGraphormerLayer(nn.Module):
    """
    去掉 gate 后的普通注意力版 HandGraphormerLayer
    保留：
    1. Graphormer 空间距离偏置
    2. Same-finger bias
    3. RMSNorm + SwiGLU FFN
    4. PyTorch SDPA
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.1, num_joints=21):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
            
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        # ==========================================
        # 1. 结构偏置 A：最短路径距离 (Shortest Path Distance)
        # ==========================================
        edges = [
            (0,1), (1,2), (2,3), (3,4),
            (0,5), (5,6), (6,7), (7,8),
            (0,9), (9,10), (10,11), (11,12),
            (0,13), (13,14), (14,15), (15,16),
            (0,17), (17,18), (18,19), (19,20)
        ]
        
        spd = torch.full((num_joints, num_joints), float('inf'))
        for i in range(num_joints):
            spd[i, i] = 0
        for i, j in edges:
            spd[i, j] = 1
            spd[j, i] = 1
            
        for k in range(num_joints):
            for i in range(num_joints):
                for j in range(num_joints):
                    if spd[i, k] + spd[k, j] < spd[i, j]:
                        spd[i, j] = spd[i, k] + spd[k, j]
                        
        max_dist = int(spd.max().item()) 
        self.register_buffer('spatial_distance', spd.long())
        
        self.spatial_bias_table = nn.Embedding(max_dist + 1, n_head)
        nn.init.zeros_(self.spatial_bias_table.weight)

        # ==========================================
        # 2. 结构偏置 B：同指偏置 (Same-Finger Bias)
        # ==========================================
        fingers = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20]
        ]
        
        same_finger = torch.zeros((num_joints, num_joints), dtype=torch.long)
        for finger in fingers:
            for i in finger:
                for j in finger:
                    same_finger[i, j] = 1
        for i in range(num_joints):
            same_finger[i, i] = 1 
        
        self.register_buffer('same_finger_index', same_finger)
        self.same_finger_bias = nn.Embedding(2, n_head)
        nn.init.zeros_(self.same_finger_bias.weight)

        # ==========================================
        # 3. Attention 分支
        # ==========================================
        self.norm_attn_in = RMSNorm(d_model)
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # 去掉 gate_proj
        self.linear_out = nn.Linear(d_model, d_model)

        # ==========================================
        # 4. FFN / MLP 分支
        # ==========================================
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None, attn_mask=None):
        """
        x:   [B, N, C]
        pos: [B, N, C]
        attn_mask: [B, N, N] 布尔型，True 代表被遮罩 (可选)
        """
        B, N, C = x.size()
        
        assert N <= self.spatial_distance.size(0), (
            f"Sequence length N={N} exceeds maximum joint bias map size."
        )
        
        residual = x
        
        # ==========================================
        # Block 1: Graphormer Global Attention
        # ==========================================
        h_in = self.norm_attn_in(x)
        if pos is not None:
            h_in = h_in + self.pos_scale * pos
            
        # 生成 Q, K, V
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.n_head, self.head_dim).transpose(1, 2),
            qkv
        )
        
        # ----- 计算图结构偏置 -----
        bias_spd = self.spatial_bias_table(self.spatial_distance[:N, :N]) 
        bias_finger = self.same_finger_bias(self.same_finger_index[:N, :N])
        
        bias = (bias_spd + bias_finger).permute(2, 0, 1).unsqueeze(0)
        
        # ----- 处理 Mask -----
        if attn_mask is not None:
            bias = bias.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        # ----- PyTorch SDPA -----
        out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=bias,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        
        # [B, n_head, N, head_dim] -> [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        # 普通 attention：直接输出投影
        out_attn = self.linear_out(out)
        x = residual + self.proj_dropout(out_attn)

        # ==========================================
        # Block 2: FFN (SwiGLU)
        # ==========================================
        residual = x
        
        mlp_in = self.norm_mlp_in(x)
        mlp_out = self.mlp(mlp_in)
        
        x = residual + self.mlp_dropout(mlp_out)
        
        return x
class PoseLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, num_joints=21, local_model_dir=None, feature_dim=768, layers=[3, 6, -1],
                  upsample_dim=512, num_refine_layers=3, use_gradient_checkpointing=True): # 默认层数建议设为3
        super().__init__()
        self.save_hyperparameters() 
        self.layers = layers
        self.upsample_dim = upsample_dim
        self.num_joints = num_joints
        self.num_refine_layers = num_refine_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 1. Backbone
        self.vitmodel = ViTFeatureExtractor(
            model_name_or_path=local_model_dir, 
            layers_to_extract=[3, 6, -1],
            freeze_backbone=True
        )
        
        # 2. Upsampling & Fusion
        self.upsample_heads = nn.ModuleList([
            UpsampleHead4x(
                in_channels=feature_dim, 
                out_channels=self.upsample_dim, 
                hidden_dim=256
            )
            for _ in range(len(layers))
        ])

        self.fuse_block = ViTPoseFusionBlock(
            in_channels_list=[self.upsample_dim, self.upsample_dim, self.upsample_dim], 
            hidden_dim=256,
            out_dim=self.upsample_dim      
        )
        
        # 3. Positional Embedding for Feature Map (Memory)
        self.pos_embed_layer = PositionEmbeddingSine(num_pos_feats=upsample_dim // 2, normalize=True)
        
        # 4. Learnable Joint Tokens & Queries
        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        self.joint_token_pos = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        
        # 初始化策略
        nn.init.normal_(self.joint_tokens, std=0.02)
        nn.init.normal_(self.joint_token_pos, std=0.02)

        # 5. Transformer Layers
        self.layers_sa = nn.ModuleList()
        for i in range(self.num_refine_layers):
            if i < 2:
                self.layers_sa.append(
                    HandGraphormerLayer(
                        d_model=upsample_dim,
                        n_head=8, 
                        dim_feedforward=1024,
                        dropout=0.1,
                    )
                )
            else: # 最后一层使用全局 Self-Attention
                self.layers_sa.append(
                    PoseSelfAttentionLayer(
                        d_model=upsample_dim,
                        n_head=8, 
                        dim_feedforward=1024,
                        dropout=0.1,
                    )
                )

        self.layers_ca = nn.ModuleList([
            PoseRefinementLayer(
                d_model=upsample_dim,
                n_head=8, 
                dim_feedforward=1024,
                dropout=0.1,
            )
            for _ in range(self.num_refine_layers)
        ])
        

        # 6. Regression Head (Shared or Independent, here shared for simplicity)
        self.pose_3d_head_PR = Pose3DRegressionHead(
            in_channels=upsample_dim, 
            mid_channels=128, 
            out_channels=6, # 3 coord + 3 logvar
            dropout=0.1,
        )
        
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 默认冻结 Backbone
        for param in self.vitmodel.parameters():
            param.requires_grad = False
    
    # def _compute_gnll_loss(self, pred_mu, pred_logvar, gt):
    #     """
    #     计算 Gaussian Negative Log Likelihood Loss
    #     """
    #     mse_term = (pred_mu - gt) ** 2
    #     precision = torch.exp(-pred_logvar)
    #     loss = 0.5 * (precision * mse_term + pred_logvar)
    #     return loss.mean()
    
    def _compute_gnll_loss(self, pred_mu, pred_logvar, gt, warmup_epochs=5, beta=10.0):
        """
        组合拳版 GNLL Loss: 延迟不确定性学习 + Smooth L1 鲁棒回归
        """
        # 1. 计算鲁棒的回归距离 (注意 reduction='none' 确保后续能逐元素乘以权重)
        # beta 是 L1 和 L2 的平滑过渡阈值
        robust_dist = F.smooth_l1_loss(pred_mu, gt, reduction='none', beta=beta)
        
        # 2. 阶段控制：利用 Lightning 内置的 self.current_epoch
        if self.current_epoch < warmup_epochs:
            # 【先学走】退化为纯 Smooth L1，完全不给 pred_logvar 传导梯度
            # 让模型专心致志地拟合 3D 坐标
            return robust_dist.mean()
        else:
            # 【再学跑】放开 logvar 权重，进行动态不确定性学习
            # 引入 clamp 防止 logvar 极端暴走 (可选，但推荐作为最后一道防线)
            pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
            
            precision = torch.exp(-pred_logvar)
            loss = precision * robust_dist + 0.5 * pred_logvar
            
            return loss.mean()
    
    # def _compute_gnll_loss(self, pred_mu, pred_logvar, gt, warmup_epochs=5, beta=10.0):
    #     return F.smooth_l1_loss(pred_mu, gt, reduction='mean', beta=beta)
    
    def forward(self, x, hand_back):
        results = {}
        B = x.shape[0]
        
        # 1. 特征提取 (ViT)
        features_dict = self.vitmodel(x) 
        
        # 2. 特征重构与上采样
        extracted_features = []
        for layer_id in self.layers:
            feat = features_dict[layer_id]
            patch_tokens = feat[:, 1:, :] # 去除 CLS
            patch_tokens = patch_tokens.transpose(1, 2)
            B, C, N = patch_tokens.shape
            H = W = int(math.sqrt(N))
            feature_map = patch_tokens.view(B, C, H, W)
            extracted_features.append(feature_map)
        
        upsampled_features = []
        for i, feat in enumerate(extracted_features):
            up_feat = self.upsample_heads[i](feat)
            upsampled_features.append(up_feat)
        

        # .Part A: 手掌处理)
        global_feature_map = self.fuse_block(upsampled_features) 

        # Memory Pos Embed
        pos_embed_map = self.pos_embed_layer(global_feature_map)
        
        # Query Tokens (Learnable) - 扩展到 Batch 维度
        curr_tokens = self.joint_tokens.expand(B, -1, -1)   # [B, 21, 256]
        query_pos   = self.joint_token_pos.expand(B, -1, -1) # [B,hand_back=None 21, 256]

        # 5. Transformer Loop: Self-Attn -> Cross-Attn -> Predict
        all_stage_preds = []
        all_stage_logvars = []
        all_stage_gates = [] # 用于记录 gate 值，分析手背利用率
        
        for i in range(self.num_refine_layers):
            if self.training and self.use_gradient_checkpointing:
                curr_tokens = checkpoint(
                    self.layers_sa[i],
                    curr_tokens,   
                    query_pos,     
                    use_reentrant=False
                )

                curr_tokens = checkpoint(
                    self.layers_ca[i],
                    curr_tokens,         
                    global_feature_map,  
                    query_pos,           
                    pos_embed_map,       
                    use_reentrant=False
                )
            else:
                curr_tokens = self.layers_sa[i](x=curr_tokens, pos=query_pos)
                curr_tokens = self.layers_ca[i](
                    tgt=curr_tokens,
                    memory=global_feature_map,
                    query_pos=query_pos,
                    memory_pos=pos_embed_map
                )

            raw_pred = self.pose_3d_head_PR(curr_tokens)
            stage_pred_mu = raw_pred[..., :3]
            stage_pred_logvar = raw_pred[..., 3:]

            all_stage_preds.append(stage_pred_mu)
            all_stage_logvars.append(stage_pred_logvar)
                

        # 取最后一个 Stage 作为最终结果
        pred_pose_3d = all_stage_preds[-1]
        pred_logvar_3d = all_stage_logvars[-1]
        
        
        results["pose3d"] = pred_pose_3d 
        results["pose3d_logvar"] = pred_logvar_3d
        results["all_stage_pose3d"] = all_stage_preds 
        results["all_stage_logvars"] = all_stage_logvars
        
        return results

    def _compute_mpjpe_3d(self, pred, gt):
        dist = torch.norm(pred - gt, dim=-1)
        return dist.mean()
    
    def training_step(self, batch, batch_idx):
        imgs = batch['img'] 
        gt_pose = batch['gt_pose'] 
        gt_pose_2d = batch['gt_pose_2d'] # 来自 dataset，已经包含了畸变
        cam_k = batch['cam_k']
        dist_coeffs = batch['dist_coeffs'] 
        hand_back = batch["hand_back"]
        results = self(imgs, hand_back)
        pred_pose_3d = results['pose3d']
        
        # 计算 Loss (GNLL Loss + Deep Supervision)
        all_preds = results['all_stage_pose3d']
        all_logvars = results['all_stage_logvars']
        
        
        loss_3d_pose = 0
        for pred_mu, pred_logvar in zip(all_preds, all_logvars):
            loss_3d_pose += self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)
        
        
        self.log('train_loss', loss_3d_pose, prog_bar=True)
        
        with torch.no_grad():
            mpjpe_3d = self._compute_mpjpe_3d(results['pose3d'], gt_pose)
            self.log('train_mpjpe_3d', mpjpe_3d, prog_bar=True)

        return loss_3d_pose

    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        gt_pose = batch['gt_pose']
        hand_back = batch["hand_back"]
        results = self(imgs, hand_back)
        # self._save_batch_visuals(batch, results, batch_idx, mode='val')
        pred_mu = results['pose3d']
        pred_logvar = results['pose3d_logvar']
        
        val_loss = self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)  
        val_mpjpe_3d = self._compute_mpjpe_3d(pred_mu, gt_pose)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mpjpe_3d', val_mpjpe_3d, prog_bar=True)

    def test_step(self, batch, batch_idx):
        imgs = batch['img']
        gt_pose = batch['gt_pose']
        hand_back = batch["hand_back"]
        results = self(imgs, hand_back)
        self._save_batch_visuals(batch, results, batch_idx, mode='test')
        pred_mu = results['pose3d']
        pred_logvar = results['pose3d_logvar']
        
        test_loss = self._compute_gnll_loss(pred_mu, pred_logvar, gt_pose)  
        test_mpjpe_3d = self._compute_mpjpe_3d(pred_mu, gt_pose)
        
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_mpjpe_3d', test_mpjpe_3d, prog_bar=True)
    
        # 用于保存可视化数据
    def _save_batch_visuals(self, batch, results, batch_idx, mode='val'):
        """
        保存前10个batch的数据用于可视化分析
        """
        if batch_idx >= 10:
            return

        # 创建保存目录
        save_dir = "vis_result"
        os.makedirs(save_dir, exist_ok=True)

        # 创建字典
        # 注意：将 Tensor 转为 numpy 
        data_to_save = {
            "img": batch['img'].detach().cpu().numpy(),              # [B, 3, H, W]
            "gt_pose_3d": batch['gt_pose'].detach().cpu().numpy(),   # [B, 21, 3]
            "gt_pose_2d": batch['gt_pose_2d'].detach().cpu().numpy(),# [B, 21, 2]
            
            # "pred_pose_2d": results['pose2d'].detach().cpu().numpy(), # [B, 21, 2]
        }
        
        # 如果有 3D 预测结果，保存
        if 'pose3d' in results:
            data_to_save["pred_pose_3d"] = results['pose3d'].detach().cpu().numpy() # [B, 21, 3]

        # 生成文件名: vis_result/val_epoch_0_batch_1.pkl
        filename = f"{mode}_epoch_{self.current_epoch}_batch_{batch_idx}.pkl"
        save_path = os.path.join(save_dir, filename)

        # 保存为 pickle 文件
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
    
    def configure_optimizers(self):
        # 训练所有参数 (除了被冻结的 backbone)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.hparams.lr, 
            weight_decay=0.04
        )
        
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
        warmup_epochs = 5
        
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }