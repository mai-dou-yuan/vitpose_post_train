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
from models.GatedMultiHeadAttention import GatedMultiHeadAttention
from models.ChannelWiseGatedFusion import ChannelWiseGatedFusion
from utils.PositionEmbeddingSine import PositionEmbeddingSine
from models.Pose3DRegressionHead import Pose3DRegressionHead
from utils.project_3d_to_2d_batch_with_distortion import project_3d_to_2d_batch_with_distortion
from models.LightweightBackFeatureExtractor import LightweightBackFeatureExtractor


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
        
        # ====== Part 2: 分支专属的 MLP (新增) ======
        # 作用：在融合前，先对手背特征进行非线性变换和对齐
        self.norm_branch_mlp = RMSNorm(d_model)
        self.branch_mlp = SwiGLU(d_model, dim_feedforward)
        
        # ====== Part 3: 门控网络 ======
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(), 
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )
        nn.init.constant_(self.gate_net[-2].bias, -2.0) 

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
        
        # ==========================================
        # Step 2: Branch MLP (你的改进点)
        # ==========================================
        # 逻辑： t_back = MLP(Norm(t_back_raw)) + t_back_raw
        # 注意：这里我们可以选择是否要残差连接。
        # 通常作为 Adapter，我们可以直接变换：out = MLP(Norm(raw)) + raw
        
        branch_residual = t_back_raw
        
        # 1. Norm
        t_back_norm = self.norm_branch_mlp(t_back_raw)
        # 2. MLP
        t_back_processed = self.branch_mlp(t_back_norm)
        # 3. Residual (可选，建议加上防止梯度消失)
        t_back_final = branch_residual + self.dropout(t_back_processed)

        # ==========================================
        # Step 3: Gated Fusion (门控融合)
        # ==========================================
        # 根据当前 Token 决定融合多少处理后的手背特征
        gate = self.gate_net(curr_tokens) 
        
        # 融合到主干
        curr_tokens = residual_input + gate * self.dropout(t_back_final)
        
        # ==========================================
        # Step 4: Main MLP (主干整合)
        # ==========================================
        residual_main = curr_tokens
        
        mlp_in = self.norm_main_mlp(curr_tokens)
        mlp_out = self.main_mlp(mlp_in)
        
        curr_tokens = residual_main + self.dropout(mlp_out)
        
        return curr_tokens, gate


class HandMultiHeadGATLayer(nn.Module):
    """
    结合了 Transformer 架构容量 (Multi-Head, FFN) 与 GAT 物理先验的满血版图注意力层。
    完全等价替换原有的 PoseSelfAttentionLayer。
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
        self.norm_attn_in = RMSNorm(d_model)
        
        # 特征变换矩阵
        self.W = nn.Linear(d_model, d_model, bias=False)
        
        # 多头注意力计算权重 a: 每个头独立拥有自己的权重向量 (形状: [n_head, 2*head_dim])
        self.a = nn.Parameter(torch.empty(n_head, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attn_dropout = nn.Dropout(dropout)
        
        # 多头拼接后的输出映射
        self.linear_out = nn.Linear(d_model, d_model)

        # ==========================================
        # 3. FFN / MLP 分支 (复用 SwiGLU 保持容量对齐)
        # ==========================================
        self.norm_mlp_in = RMSNorm(d_model)
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
            
        # 1. 线性映射并拆分多头 -> [B, N, n_head, head_dim]
        h = self.W(h_in).view(B, N, self.n_head, self.head_dim)
        
        # 2. 准备计算注意力分数
        # h_i: [B, N, N, n_head, head_dim]
        h_i = h.unsqueeze(2).expand(B, N, N, self.n_head, self.head_dim)
        h_j = h.unsqueeze(1).expand(B, N, N, self.n_head, self.head_dim)
        
        # 拼接特征 -> [B, N, N, n_head, 2*head_dim]
        h_cat = torch.cat([h_i, h_j], dim=-1) 
        
        # 3. 计算多头注意力得分 e
        # 广播 a 使得它能和 h_cat 相乘后在最后一个维度求和
        a_expanded = self.a.view(1, 1, 1, self.n_head, 2 * self.head_dim)
        e = (h_cat * a_expanded).sum(dim=-1) # -> [B, N, N, n_head]
        
        # 【新增】缩放机制：除以 sqrt(head_dim) 以防止 Softmax 梯度消失
        e = e / math.sqrt(self.head_dim)
        
        e = self.leakyrelu(e)
        
        # 将 head 维度提前，方便后续和 Mask 交互 -> [B, n_head, N, N]
        e = e.permute(0, 3, 1, 2)
        
        # 4. Mask 操作 (FP16/BF16 混合精度安全的 Mask)
        # adj_mask [N, N] 会自动广播到 [B, n_head, N, N]
        # 使用 -1e4 防止混合精度计算时出现 NaN 或 Inf
        e = e.masked_fill(~self.adj_mask, -1e4)
        
        # 5. Softmax & Dropout
        attention = F.softmax(e, dim=-1)
        attention = self.attn_dropout(attention)
        
        # 6. 聚合邻居节点特征
        # h_v: 调整维度以用于矩阵乘法 -> [B, n_head, N, head_dim]
        h_v = h.permute(0, 2, 1, 3)
        # [B, n_head, N, N] @ [B, n_head, N, head_dim] -> [B, n_head, N, head_dim]
        h_prime = torch.matmul(attention, h_v)
        
        # 7. 拼接多头并输出映射
        # [B, n_head, N, head_dim] -> [B, N, n_head, head_dim] -> [B, N, C]
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out_attn = self.linear_out(h_prime)
        
        # Attention 模块的残差连接
        x = residual + self.attn_dropout(out_attn)

        # ==========================================
        # Block 2: FFN (SwiGLU) 分支
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
        self.pos_embed_layer = PositionEmbeddingSine(num_pos_feats=upsample_dim // 2, normalize=True)
        

        # 两个视角的标识符：0代表手掌，1代表手背
        # 维度 [2, 1, upsample_dim] -> 广播到 [B, L, C]
        self.view_embed = nn.Parameter(torch.randn(2, 1, self.upsample_dim))
        nn.init.normal_(self.view_embed, std=0.02)
        # 4. Learnable Joint Tokens & Queries (替换原有的 2D Sample 逻辑)
        # [1, 21, 256]
        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        self.joint_token_pos = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        
        # 初始化策略
        nn.init.normal_(self.joint_tokens, std=0.02)
        nn.init.normal_(self.joint_token_pos, std=0.02)

        # 5. Transformer Layers
        # 结构：每一层包含一个 Self-Attn 和一个 Cross-Attn
        self.layers_sa = nn.ModuleList([
            PoseSelfAttentionLayer(
                d_model=upsample_dim,
                n_head=8, 
                dim_feedforward=1024,
                dropout=0.1,
            )
            for _ in range(num_refine_layers)
        ])
        
        self.layers_ca = nn.ModuleList([
            PoseRefinementLayer(
                d_model=upsample_dim,
                n_head=8, 
                dim_feedforward=1024,
                dropout=0.1,
            )
            for _ in range(num_refine_layers)
        ])
        
        # 2. 新增：手背修正层列表 (与 num_refine_layers 对应)
        self.layers_back_refine = nn.ModuleList([
            GatedBackRefinementBlock(
                d_model=upsample_dim,
                n_head=8,
                dropout=0.1
            )
            for _ in range(num_refine_layers)
        ])
        self.back_feature_extractor = LightweightBackFeatureExtractor(
            out_channels=self.upsample_dim, 
            pretrained=True
        )


        # 6. Regression Head (Shared or Independent, here shared for simplicity)
        self.pose_3d_head_PR = Pose3DRegressionHead(
            in_channels=upsample_dim, 
            mid_channels=128, 
            out_channels=6, # 3 coord + 3 logvar
            dropout=0.1,
        )
        
        # self.residual_branch = HandBackResidualNet(num_joints=num_joints, hidden_dim=256)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 默认冻结 Backbone
        for param in self.vitmodel.parameters():
            param.requires_grad = False
    
    def _compute_gnll_loss(self, pred_mu, pred_logvar, gt):
        """
        计算 Gaussian Negative Log Likelihood Loss
        """
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
        
        # 3. 全局特征融合 (Memory)
        # global_feature_map: [B, 256, 64, 64]
        # global_feature_map = self.fuse_block(upsampled_features) 
        # ... (Part A: 手掌处理) ...
        global_feature_map = self.fuse_block(upsampled_features) 
        # [B, C, H, W] -> [B, L, C]
        memory_palm = global_feature_map.flatten(2).transpose(1, 2)
        # === 改进点 1: 加上手掌的 View ID ===
        memory_palm = memory_palm + self.view_embed[0]
        # 4. 准备 Transformer 输入
        # Memory Pos Embed
        # pos_embed_map = self.pos_embed_layer(global_feature_map)
        pos_embed_map = self.pos_embed_layer(global_feature_map).flatten(2).transpose(1, 2)

        # ==========================================
        # Part B: 辅助视角 (Back) - 使用轻量级模块
        # ==========================================

        
        # 准备 Back Memory
        # [B, C, H, W] -> [B, L, C] (Flatten & Transpose)
        # memory_back = back_feature_map.flatten(2).transpose(1, 2)
        # ... (Part B: 手背处理) ...
        back_feature_map = self.back_feature_extractor(hand_back)
        # [B, C, H, W] -> [B, L, C]
        memory_back = back_feature_map.flatten(2).transpose(1, 2)
        # === 改进点 2: 加上手背的 View ID ===
        memory_back = memory_back + self.view_embed[1]
        
        # 生成 Positional Embedding
        # 注意：如果 ResNet 输出尺寸和 ViT 主分支尺寸完全一致 (e.g. 64x64)，
        # 可以直接复用 self.pos_embed_layer。如果不一致，需要新建一个 pos_embed_layer。
        # 这里的 FPN 设计为 stride 4，通常 ViT patch 16 + upsample 4x 也是 stride 4，大概率是一样的。
        pos_back = self.pos_embed_layer(back_feature_map).flatten(2).transpose(1, 2)


        
        # Query Tokens (Learnable) - 扩展到 Batch 维度
        curr_tokens = self.joint_tokens.expand(B, -1, -1)   # [B, 21, 256]
        query_pos   = self.joint_token_pos.expand(B, -1, -1) # [B,hand_back=None 21, 256]

        # 5. Transformer Loop: Self-Attn -> Cross-Attn -> Predict
        all_stage_preds = []
        all_stage_logvars = []
        all_stage_gates = [] # 用于记录 gate 值，分析手背利用率
        for i in range(self.num_refine_layers):
            # Step A: Self Attention (Token 之间交互)
            curr_tokens = self.layers_sa[i](
                x=curr_tokens,
                pos=query_pos
            )
            
            # Step B: Cross Attention (Token 查询全局特征)
            curr_tokens = self.layers_ca[i](
                tgt=curr_tokens,
                # memory=global_feature_map,
                memory=memory_palm,
                query_pos=query_pos,
                memory_pos=pos_embed_map
            )

            if i == self.num_refine_layers - 1:           
                # 3. [新增] Cross Attention to Back (手背视角 + 门控)
                # 逻辑：tokens = tokens + gate * Attn(back)
                curr_tokens, gate_val = self.layers_back_refine[i](
                    curr_tokens=curr_tokens,
                    back_memory=memory_back,
                    query_pos=query_pos,
                    back_pos=pos_back
                )
                all_stage_gates.append(gate_val) # 新添的
            # Step C: Regression (Deep Supervision)
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
        
        # # 2D Projection Loss
        # # 调用带畸变的投影函数
        # pred_pose_2d = project_3d_to_2d_batch_with_distortion(
        #     pred_pose_3d, 
        #     cam_k, 
        #     dist_coeffs
        # )
        
        # # 计算 Loss (用 Mask 过滤无效点)
        # valid_mask = (gt_pose_2d[..., 0] > -500) # 过滤 -1000 的点
        # valid_mask = valid_mask.unsqueeze(-1).expand_as(pred_pose_2d)
        
        # loss_2d = 0.0
        # if valid_mask.sum() > 0:
        #     loss_2d = torch.nn.functional.smooth_l1_loss(
        #         pred_pose_2d[valid_mask], 
        #         gt_pose_2d[valid_mask]
        #     )
        
        # 总 Loss 加权 
        # 2D Loss 的数值通常很大 (像素级)，3D Loss 很小 (米级)
        # 需要调节权重。假设 3D 是米(0.02左右)，2D 是像素(10-20左右)
        # 建议给 loss_2d 一个较小的权重，或者先 normalize 2D 坐标
        
        
        # w_2d = 0.01  # 经验值，建议从 0.01 或 0.1 开始调
        # total_loss = loss_3d_pose + w_2d * loss_2d
        
        # self.log('train_loss_3d', loss_3d_pose)
        # self.log('train_loss_2d', loss_2d)
        # self.log('train_loss', total_loss, prog_bar=True)
        
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
        warmup_epochs = 10
        
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