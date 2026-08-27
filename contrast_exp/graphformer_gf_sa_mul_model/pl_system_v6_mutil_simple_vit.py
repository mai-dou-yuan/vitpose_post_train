# pl_system.py
import math
import os
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.HandMultiHeadGATLayer import HandMultiHeadGATLayer
from models.HandBackResidualNet import HandBackResidualNet
from models.PalmModulatesBackBlock import PalmModulatesBackBlock
from models.model_PoseEstimationModel import LightweightBackFusion, PoseEstimationModel
from models.model_vit import ViTFeatureExtractor
from models.model_PoseEstimationModel import ViTPoseFusionBlock
from models.UpsampleHead4x import UpsampleHead4x
from models.GatedMultiHeadAttention import GatedMultiHeadAttention
from models.ChannelWiseGatedFusion import ChannelWiseGatedFusion
from utils.PositionEmbeddingSine import HierarchicalPositionEmbedding, PositionEmbeddingSine
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

        self.cross_attention = GatedMultiHeadAttention(
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

        self.self_attention = GatedMultiHeadAttention(
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

class GatedBackRefinementBlock(nn.Module):
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)

        # ====== Part 1: Attention 相关 ======
        self.norm_input = RMSNorm(d_model) 
        self.norm_mem = RMSNorm(d_model)
        
        self.back_attention = GatedMultiHeadAttention(
            d_model=d_model, 
            n_head=n_head, 
            dropout=dropout
        )
        
        
        # ====== Part 3: 门控网络 (Token-wise Gating) ======
        hidden_dim = d_model // 4
        self.gate_net = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, 1), # 注意：这里改为输出 1 个权重
            nn.Sigmoid() 
        )
        # 初始化，保持在极小值，让模型初期先依赖手掌
        nn.init.constant_(self.gate_net[-2].bias, -2.0)
        nn.init.zeros_(self.gate_net[-2].weight)


        # ====== Part 4: 主干 MLP ======
        # 作用：融合后的特征整合
        self.norm_main_mlp = RMSNorm(d_model)
        self.main_mlp = SwiGLU(d_model, dim_feedforward)

        self.dropout = nn.Dropout(dropout)

    def forward(self, curr_tokens, back_memory, query_pos=None, back_pos=None):
        """
        curr_tokens: [B, N, C]
        back_memory: [B, H*W, C]
        """
        # ==========================================
        # Step 1: Cross Attention (获取原始手背信息)
        # ==========================================
        residual_input = curr_tokens
        
        q = self.norm_input(curr_tokens) + (query_pos if query_pos is not None else 0)
        k = self.norm_mem(back_memory) + (back_pos if back_pos is not None else 0)
        v = back_memory
        
        t_back_raw = self.back_attention(query=q, key=k, value=v)


        # 根据当前 Token 决定融合多少处理后的手背特征
        gate = self.gate_net(curr_tokens) 
        # 训练时以一定概率 (比如 20%) 随机丢弃手背分支
        if self.training and torch.rand(1).item() < 0.2:
            gate = gate * 0.0
        # gate_input = torch.cat([residual_input, t_back_final], dim=-1)
        # gate = self.gate_net(gate_input)
        
        # Step 3: 直接融合
        curr_tokens = residual_input + gate * self.dropout(t_back_raw)
        
        # Step 4: Main MLP (整理特征)
        residual_main = curr_tokens
        mlp_in = self.norm_main_mlp(curr_tokens)
        mlp_out = self.main_mlp(mlp_in)
        curr_tokens = residual_main + self.dropout(mlp_out)
        
        return curr_tokens, gate


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


class PoseLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, num_joints=21, local_model_dir=None, feature_dim=768, layers=[3, 6, -1],
                  upsample_dim=512, num_refine_layers=3): # 默认层数建议设为3
        super().__init__()
        self.save_hyperparameters() 
        self.layers = layers
        self.upsample_dim = upsample_dim
        self.num_joints = num_joints
        self.num_refine_layers = num_refine_layers
        
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
        # self.pos_embed_layer = PositionEmbeddingSine(num_pos_feats=upsample_dim // 2, normalize=True)
        
        # 3. 层次化 Positional Embedding for Feature Map (Memory)
        self.hierarchical_pe = HierarchicalPositionEmbedding(upsample_dim=self.upsample_dim, num_views=2)

        self.back_proj = nn.Sequential(
            RMSNorm(feature_dim),                           
            nn.Linear(feature_dim, feature_dim),            
            nn.SiLU(),                                      
            nn.Dropout(0.1),                                
            nn.Linear(feature_dim, self.upsample_dim)       # 5. 最终降维到目标空间
        )

        # 两个视角的标识符：0代表手掌，1代表手背
        # 维度 [2, 1, upsample_dim] -> 广播到 [B, L, C]
        self.view_embed = nn.Parameter(torch.randn(2, 1, self.upsample_dim))
        nn.init.normal_(self.view_embed, std=0.02)
        
        # 4. Learnable Joint Tokens & Queries (替换原有的 2D Sample 逻辑)
        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        self.joint_token_pos = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        
        # 初始化策略
        nn.init.normal_(self.joint_tokens, std=0.02)
        nn.init.normal_(self.joint_token_pos, std=0.02)



        # 5. Transformer Layers
        self.layers_sa = nn.ModuleList()
        for i in range(self.num_refine_layers):
            # if i < self.num_refine_layers - 2: # 前面 N-1 层使用 Gformer
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
        
        
        # 2. 手背修正层列表 (与 num_refine_layers 对应)
        self.layers_back_refine = nn.ModuleList([
            GatedBackRefinementBlock(
                d_model=upsample_dim,
                n_head=8,
                dropout=0.2
            )
            for _ in range(num_refine_layers)
        ])


        # 6. Regression Head (Shared or Independent, here shared for simplicity)
        self.pose_3d_head_PR = Pose3DRegressionHead(
            in_channels=upsample_dim, 
            mid_channels=256, 
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
    def _compute_gnll_loss(self, pred_mu, pred_logvar, gt):
        # 防止过度自信导致 exp(-logvar) 爆炸
        pred_logvar = torch.clamp(pred_logvar, min=-4.0, max=4.0) 
        mse_term = (pred_mu - gt) ** 2
        precision = torch.exp(-pred_logvar)
        loss = 0.5 * (precision * mse_term + pred_logvar)
        return loss.mean()
        
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
        
        # Part A: 手掌处理
        global_feature_map = self.fuse_block(upsampled_features) 

        # memory_palm = memory_palm + self.view_embed[0]
        global_feature_map = global_feature_map + self.view_embed[0].view(1, self.upsample_dim, 1, 1)
        pos_embed_map = self.hierarchical_pe(global_feature_map, view_id=0)


        # PartB: 辅助视角 (Back) 特征提取与融合 (轻量化)
        # 复用冻结的 ViT 提取手背特征
        back_features_dict = self.vitmodel(hand_back)
        # 直接取 ViT 提取的最后一层特征
        last_layer = self.layers[-1]
        back_patch_tokens = back_features_dict[last_layer][:, 1:, :] # 去除 CLS Token, 形状 [B, N, feature_dim]
        
        # 维度对齐: feature_dim -> upsample_dim
        back_tokens = self.back_proj(back_patch_tokens) # [B, N, upsample_dim]
        
        B, N_back, C_back = back_tokens.shape
        H_back = W_back = int(math.sqrt(N_back))
        back_feature_map = back_tokens.transpose(1, 2).view(B, C_back, H_back, W_back) # [B, upsample_dim, H_back, W_back]
        
        # 准备手背的 Memory 和 Positional Embedding
        memory_back = back_tokens + self.view_embed[1] # [B, N, upsample_dim], 加入手背视角 ID
        
        pos_back_map = self.hierarchical_pe(back_feature_map, view_id=1)
        pos_back = pos_back_map.flatten(2).transpose(1, 2)


        
        # Query Tokens (Learnable) - 扩展到 Batch 维度
        curr_tokens = self.joint_tokens.expand(B, -1, -1)   # [B, 21, 256]
        query_pos   = self.joint_token_pos.expand(B, -1, -1) # [B,hand_back=None 21, 256]


        #  Transformer Loop: Self-Attn -> Cross-Attn -> Predict
        all_stage_preds = []
        all_stage_logvars = []
        all_stage_gates = [] # 用于监控门控值


        for i in range(self.num_refine_layers):
            if self.training:
                curr_tokens = checkpoint(self.layers_sa[i], curr_tokens, query_pos, use_reentrant=False)
                curr_tokens = checkpoint(self.layers_ca[i], curr_tokens, global_feature_map, query_pos, pos_embed_map, use_reentrant=False)
                
                # Step 3: Back Cross Attention + Gate (向手背图查询缺失特征) - 仅在最后一层执行
                # if i == self.num_refine_layers - 1:
                curr_tokens, gate_val = checkpoint(self.layers_back_refine[i], curr_tokens, memory_back, query_pos, pos_back, use_reentrant=False)
                # else:
                #     gate_val = None
            else:
                curr_tokens = self.layers_sa[i](x=curr_tokens, pos=query_pos)
                curr_tokens = self.layers_ca[i](tgt=curr_tokens, memory=global_feature_map, query_pos=query_pos, memory_pos=pos_embed_map)
                
                # 仅在最后一层执行手背融合
                # if i == self.num_refine_layers - 1:
                curr_tokens, gate_val = self.layers_back_refine[i](curr_tokens=curr_tokens, back_memory=memory_back, query_pos=query_pos, back_pos=pos_back)
                # else:
                #     gate_val = None

            # 记录门控值（非最后一层记录 None）
            if gate_val is not None:
                all_stage_gates.append(gate_val.detach())
            else:
                all_stage_gates.append(None) 

            # Regression
            raw_pred = self.pose_3d_head_PR(curr_tokens) 
            all_stage_preds.append(raw_pred[..., :3])
            all_stage_logvars.append(raw_pred[..., 3:])

        # 取最后一个 Stage 作为最终结果
        pred_pose_3d = all_stage_preds[-1]
        pred_logvar_3d = all_stage_logvars[-1]

        results["pose3d"] = all_stage_preds[-1]
        results["pose3d_logvar"] = all_stage_logvars[-1]
        results["all_stage_pose3d"] = all_stage_preds 
        results["all_stage_logvars"] = all_stage_logvars
        results["gate_values"] = all_stage_gates # 在验证集打印这个值，观察哪些关节激活了手背信息
        
        
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