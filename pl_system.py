# pl_system.py
import math
import os
import pdb
import pickle
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.HandBackRegressionHead import HandBackRegressionHead
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
        # 定义特征图尺寸 (请根据实际情况修改)
        self.h_palm, self.w_palm = 96, 96  # ViT(16) -> Upsample(4x)
        self.h_back, self.w_back = 42, 42  # 假设手背网络输出是 

        # 可学习的位置编码 (Learnable Positional Embeddings)
        # Shape: [1, H*W, C]
        self.pos_embed_palm = nn.Parameter(torch.randn(1, self.h_palm * self.w_palm, self.upsample_dim))
        self.pos_embed_back = nn.Parameter(torch.randn(1, self.h_back * self.w_back, self.upsample_dim))

        # 初始化
        nn.init.trunc_normal_(self.pos_embed_palm, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_back, std=0.02)



        # 4. Learnable Joint Tokens & Queries (替换原有的 2D Sample 逻辑)
        # [1, 21, 256]
        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        self.joint_token_pos = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        
        # 初始化策略
        nn.init.normal_(self.joint_tokens, std=0.02)
        nn.init.normal_(self.joint_token_pos, std=0.02)

        # Transformer Layers
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
        
        
        self.back_regression_head = HandBackRegressionHead(
            in_channels=self.upsample_dim, # 512
            num_joints=num_joints,         # 21
            hidden_dim=256
        )


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
        
        # 特征提取 (ViT)
        features_dict = self.vitmodel(x) 
        
        # 特征重构与上采样
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
        
        # 全局特征融合 (Memory)
        # global_feature_map: [B, 256, 64, 64]
        global_feature_map = self.fuse_block(upsampled_features) # torch.Size([12, 512, 96, 96])
        # [B, C, H, W] -> [B, L, C]
        memory_palm = global_feature_map.flatten(2).transpose(1, 2)
        # Memory Pos Embed
        # pos_embed_map = self.pos_embed_layer(global_feature_map)
        pos_palm = self.pos_embed_palm.expand(B, -1, -1)

        
        # Query Tokens (Learnable) - 扩展到 Batch 维度
        curr_tokens = self.joint_tokens.expand(B, -1, -1)   # [B, 21, 256]
        query_pos   = self.joint_token_pos.expand(B, -1, -1) # [B,hand_back=None 21, 256]

        # 5. Transformer Loop: Self-Attn -> Cross-Attn -> Predict
        all_stage_preds = []
        all_stage_logvars = []

        for i in range(self.num_refine_layers):
            # Step A: Self Attention (Token 之间交互)
            curr_tokens = self.layers_sa[i](
                x=curr_tokens,
                pos=query_pos
            )
            
            # Step B: Cross Attention (Token 查询全局特征)
            curr_tokens = self.layers_ca[i](
                tgt=curr_tokens,
                memory=memory_palm,
                query_pos=query_pos,
                memory_pos=pos_palm,

            )

            # Step C: Regression (Deep Supervision)
            raw_pred = self.pose_3d_head_PR(curr_tokens) 
            
            stage_pred_mu = raw_pred[..., :3]
            stage_pred_logvar = raw_pred[..., 3:]
            
            all_stage_preds.append(stage_pred_mu)
            all_stage_logvars.append(stage_pred_logvar)

        # 获取 Transformer 最终预测的手掌 3D 姿态 (Palm-Only Result)
        palm_pose_3d = all_stage_preds[-1]
        palm_logvar_3d = all_stage_logvars[-1]
           
        # 手背残差分支 (Back Residual Branch)
        back_feature_map = self.back_feature_extractor(hand_back) # [B, 512, 11, 11]
        # 计算残差 (Delta)
        delta_pose = self.back_regression_head(back_feature_map) # [B, 21, 3]

        #Root-Relative 融合
        # 取出 Transformer 预测的 Root (通常是 idx 0，手腕)
        root_idx = 0 
        pred_root = palm_pose_3d[:, root_idx:root_idx+1, :] # [B, 1, 3]
        
        # 将预测转为相对坐标 (去掉平移影响)
        pred_relative = palm_pose_3d - pred_root
        
        # 将残差加在相对坐标上 (修正姿态，不修正位置)
        # 注意：这里假设 delta_pose 也是相对坐标修正量
        refined_relative = pred_relative + delta_pose
        
        # 把 Root 加回去 (恢复绝对坐标)
        final_pose_3d = refined_relative + pred_root

        # 保存结果
        results["pose3d_palm_only"] = palm_pose_3d
        results["pose3d"] = final_pose_3d 
        results["pose3d_logvar"] = palm_logvar_3d # Logvar 通常延用 Transformer 的
        
        # Deep Supervision 处理：
        # 之前的层也可以加残差，但最简单的是只修正最后一层
        results["all_stage_pose3d"] = all_stage_preds[:-1] + [final_pose_3d] # list中+表示合并
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
    
