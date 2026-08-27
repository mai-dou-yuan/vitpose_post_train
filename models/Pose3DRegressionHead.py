import torch
import torch.nn as nn

class Pose3DRegressionHead(nn.Module):
    def __init__(self, in_channels=256, mid_channels=128, out_channels=3, dropout=0.2):
        """
        Args:
            in_channels: 输入特征维度 (256)
            mid_channels: 中间层维度 (通常是 in/2 或保持不变)
            out_channels: 输出维度 (3: x, y, z)
        """
        super().__init__()
        
        self.net = nn.Sequential(
            # 第一层：特征压缩与非线性变换
            nn.Linear(in_channels, mid_channels),
            nn.LayerNorm(mid_channels), # 归一化有助于回归任务稳定
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第二层：进一步提取
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            
            # 第三层：直接映射到 3D 坐标
            nn.Linear(mid_channels, out_channels)
        )
        
        # 权重初始化（非常重要！）
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 针对最后一层 Linear，通常希望初始输出接近 0 或分布中心
        # 这样梯度在初期不会爆炸
        last_layer = self.net[-1]
        nn.init.normal_(last_layer.weight, mean=0, std=0.01)
        nn.init.constant_(last_layer.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 21, 256]
        Returns:
            out: [B, 21, 3]
        """
        return self.net(x)