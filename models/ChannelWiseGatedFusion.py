import torch
import torch.nn as nn

class ChannelWiseGatedFusion(nn.Module):
    def __init__(self, channel_dim=256, num_levels=3):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_levels = num_levels
        
        # 1. 压缩特征用于计算权重 (类似于 SE-Block 的 Squeeze 操作)
        # 输入维度: channel_dim * num_levels
        # 这一步是为了整合所有层的信息来做决策
        self.w_generator = nn.Sequential(
            nn.Linear(channel_dim * num_levels, channel_dim // 2), # 降维减少参数
            nn.ReLU(inplace=True),
            nn.Linear(channel_dim // 2, channel_dim * num_levels)  # 恢复到每个层每个通道都有一个权重
        )
        
        # 2. 最后的融合映射 (可选，但推荐)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(channel_dim),
            nn.ReLU(inplace=True),
            nn.Linear(channel_dim, channel_dim)
        )

    def forward(self, features_list):
        """
        features_list: List of [B, K, C]
        """
        batch_size, num_joints, channels = features_list[0].shape
        
        # [B, K, Levels, C] -> Stack 起来
        stack_feats = torch.stack(features_list, dim=2) 
        
        # [B, K, Levels*C] -> Concat 起来喂给全连接层
        cat_feats = torch.cat(features_list, dim=-1)
        
        # 生成权重 [B, K, Levels*C]
        raw_weights = self.w_generator(cat_feats)
        
        # Reshape 成 [B, K, Levels, C] 以便做 Softmax
        raw_weights = raw_weights.view(batch_size, num_joints, self.num_levels, channels)
        
        # === 关键点: Channel-wise Softmax ===
        # dim=2 表示在 "Levels" 维度上做归一化。
        # 意义：对于第 c 个通道，Layer1_w + Layer2_w + Layer3_w = 1
        # 这意味着模型在为每个通道“投票”选出最佳来源。
        weights = torch.softmax(raw_weights, dim=2)
        
        # 加权求和
        # weights: [B, K, Levels, C]
        # stack_feats: [B, K, Levels, C]
        # sum(dim=2) -> [B, K, C]
        fused_feat = (weights * stack_feats).sum(dim=2)
        
        return self.out_proj(fused_feat)