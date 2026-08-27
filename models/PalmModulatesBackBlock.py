import math
import os
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
class PalmModulatesBackBlock(nn.Module):
    """
    基于主从跨视角的通道与空间注意力调制模块 (Cross-CBAM Style)。
    利用结构更完整的手掌特征 (palm_map) 来指导手背特征 (back_map) 的提纯，
    依次进行通道重标定和空间掩码过滤。
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # ==========================================
        # 1. 跨通道注意力 (Cross-Channel Attention)
        # ==========================================
        # 作用：结合掌心和手背的全局上下文，推断“哪些通道对当前手背特征是有用的”
        # 例如抑制纯背景通道，增强纹理或边缘通道
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        # ==========================================
        # 2. 跨空间注意力 (Cross-Spatial Attention)
        # ==========================================
        # 作用：利用手掌的空间显著性特征，给手背生成一个精准的空间 Mask
        # 输入维度为 4：掌心(Max+Avg) + 手背(Max+Avg)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False),  # 7x7 大感受野
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, palm_map, back_map):
        """
        palm_map: [B, C, H, W] - 作为强指导信号 (Guidance)
        back_map: [B, C, H, W] - 被调制的特征图 (Target)
        """
        B, C, H, W = palm_map.size()

        # ------------------------------------------
        # Step 1: 跨通道调制 (Cross-Channel Modulation)
        # ------------------------------------------
        # 分别提取两者的全局通道上下文 (Global Average Pooling)
        palm_gap = F.adaptive_avg_pool2d(palm_map, 1).view(B, C) 
        back_gap = F.adaptive_avg_pool2d(back_map, 1).view(B, C) 

        # 联合推断手背特征的通道权重: [B, 2C] -> [B, C]
        channel_weights = self.channel_mlp(torch.cat([palm_gap, back_gap], dim=1))
        channel_weights = channel_weights.view(B, C, 1, 1)

        # 通道提纯：用权重乘回手背特征
        back_refined_channel = back_map * channel_weights

        # ------------------------------------------
        # Step 2: 跨空间调制 (Cross-Spatial Modulation)
        # ------------------------------------------
        # 提取空间维度的显著性先验 (通道维度的 Max 和 Avg)
        
        # 手掌的空间指导特征 (提供手掌的大致轮廓和位置)
        palm_max = torch.max(palm_map, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        palm_avg = torch.mean(palm_map, dim=1, keepdim=True)    # [B, 1, H, W]

        # 手背当前的空间特征 (使用经过通道提纯后的)
        back_max = torch.max(back_refined_channel, dim=1, keepdim=True)[0]
        back_avg = torch.mean(back_refined_channel, dim=1, keepdim=True)

        # 融合两者空间信息，生成最终的空间掩码
        spatial_input = torch.cat([palm_max, palm_avg, back_max, back_avg], dim=1) # [B, 4, H, W]
        spatial_mask = self.spatial_conv(spatial_input) # [B, 1, H, W]

        # 空间提纯：抑制空间背景噪声
        modulated_back = back_refined_channel * spatial_mask

        return modulated_back, spatial_mask, channel_weights
