import math
import torch
import torch.nn as nn

class PositionEmbeddingSine(nn.Module):
    """
    标准的 2D 正弦位置编码 (DETR Style)
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        # x: [Batch, Channel, H, W] (只用到 H 和 W)
        # mask: 可选，这里假设全图有效，不传 mask
        not_mask = torch.ones((x.shape[0], x.shape[2], x.shape[3]), device=x.device)
        
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # [B, H, W, C] -> [B, C, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
class HierarchicalPositionEmbedding(nn.Module):
    def __init__(self, upsample_dim, num_views=2):
        super().__init__()
        # 基础 2D PE 生成器
        self.base_pe = PositionEmbeddingSine(num_pos_feats=upsample_dim // 2, normalize=True)

        # 为不同视角分配专属的缩放(gamma)和平移(beta)
        # view=0 为手掌，view=1 为手背
        self.gamma = nn.Parameter(torch.ones(num_views, upsample_dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(num_views, upsample_dim, 1, 1))

    def forward(self, feature_map, view_id):
        """
        view_id: 0 (palm) 或 1 (back)
        """
        # 1. 获取基础 2D 编码
        base_pos = self.base_pe(feature_map) # [B, C, H, W]

        # 2. 注入视角层次信息 (广播机制)
        hierarchical_pos = base_pos * self.gamma[view_id] + self.beta[view_id]

        return hierarchical_pos