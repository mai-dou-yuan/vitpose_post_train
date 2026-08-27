import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """标准残差块: 用于特征精修，不改变分辨率"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 维度对齐
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

# --- 2. 新增 DeconvBlock (专门处理上采样) ---
class DeconvBlock(nn.Module):
    """
    反卷积块: ConvTranspose -> BN -> ReLU
    用于将分辨率扩大 2 倍
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.deconv = nn.Sequential(
            # kernel=4, stride=2, padding=1 是标准的2倍上采样参数
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.deconv(x)

# --- 3. 优化后的 UpsampleHead4x ---
class UpsampleHead4x(nn.Module):
    """
    使用模块化的 DeconvBlock 和 ResBlock 搭建
    """
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super().__init__()
        
        # 第一次 2倍上采样 (Stage 1)
        self.stage1 = nn.Sequential(
            DeconvBlock(in_channels, hidden_dim), # 变大
            ResBlock(hidden_dim, hidden_dim)      # 精修
        )
        
        # 第二次 2倍上采样 (Stage 2)
        self.stage2 = nn.Sequential(
            DeconvBlock(hidden_dim, hidden_dim),  # 变大
            ResBlock(hidden_dim, hidden_dim)      # 精修
        )
        
        # 最终预测层 (1x1 Conv)
        self.final_layer = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.final_layer(x)
        return x

# --- 测试 ---
if __name__ == "__main__":
    model = UpsampleHead4x(in_channels=512, out_channels=17)
    x = torch.randn(1, 512, 16, 16)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    # Output: torch.Size([1, 17, 64, 64])