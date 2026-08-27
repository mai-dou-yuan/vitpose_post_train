import torch.nn as nn
class HandBackRegressionHead(nn.Module):
    def __init__(self, in_channels, num_joints=21, hidden_dim=256):
        super().__init__()
        # 1. 空间降维: [B, C, 11, 11] -> [B, C, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. MLP 回归 Delta 坐标
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim), # 增加稳定性
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_joints * 3) # 输出 21 * 3
        )

        # 3. 零初始化 (Zero Initialization) - 关键技巧
        # 确保初始状态下该残差为 0，不破坏主网络的预测
        nn.init.constant_(self.fc[-1].weight, 0)
        nn.init.constant_(self.fc[-1].bias, 0)

    def forward(self, x):
        # x: [B, 512, 11, 11]
        x = self.avg_pool(x)
        x = self.fc(x)
        # Reshape 为 [B, 21, 3]
        return x.view(-1, 21, 3)