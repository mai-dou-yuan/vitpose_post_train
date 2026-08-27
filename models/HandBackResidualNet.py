
import torch.nn.functional as F
from torchvision.models import convnext_small
from torchvision.models.feature_extraction import create_feature_extractor
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """基本卷积块: Conv -> BN -> ReLU"""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class HandBackResidualNet(nn.Module):
    def __init__(self, num_joints=21, hidden_dim=256):
        super().__init__()
        
        # 1. Backbone: ConvNeXt-Small
        backbone = convnext_small(weights='DEFAULT')
        
        # 提取 c2 (1/4), c3 (1/8), c4 (1/16 或 1/32)
        # ConvNeXt small channels: [96, 192, 384, 768]
        return_nodes = {
            'features.3': 'c2', # 192 channels, stride 8
            'features.5': 'c3', # 384 channels, stride 16
            'features.7': 'c4', # 768 channels, stride 32
        }
        self.feature_extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        
        # 2. 侧向连接 (Lateral Connections) - 统一通道数
        self.lat_c4 = nn.Conv2d(768, hidden_dim, 1)
        self.lat_c3 = nn.Conv2d(384, hidden_dim, 1)
        self.lat_c2 = nn.Conv2d(192, hidden_dim, 1)

        # 3. FPN (Top-Down) 卷积层
        self.fpn_conv4 = ConvBlock(hidden_dim, hidden_dim)
        self.fpn_conv3 = ConvBlock(hidden_dim, hidden_dim)
        self.fpn_conv2 = ConvBlock(hidden_dim, hidden_dim)

        # 4. PAN (Bottom-Up) 卷积层 (下采样)
        self.downsample_conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.downsample_conv3 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        
        self.pan_conv3 = ConvBlock(hidden_dim, hidden_dim)
        self.pan_conv4 = ConvBlock(hidden_dim, hidden_dim)

        # 5. 回归头
        # 融合后的特征将被池化到一个固定的网格大小 (例如 4x4)，而不是 1x1
        # 这样保留了部分空间结构信息 (上、下、左、右)
        self.pool_size = 4 
        fusion_dim = hidden_dim * 3 # 拼接三层
        
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_dim * self.pool_size * self.pool_size, 1024), # 输入更大
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, num_joints * 3) 
        )
        
        # 初始化
        nn.init.uniform_(self.reg_head[-1].weight, -0.001, 0.001)
        nn.init.constant_(self.reg_head[-1].bias, 0)

    def forward(self, x):
        # x: [B, 3, 518, 518]
        feats = self.feature_extractor(x)
        c2, c3, c4 = feats['c2'], feats['c3'], feats['c4']
        
        # === 1. Lateral Projections ===
        p4 = self.lat_c4(c4) # [B, 256, H/32, W/32]
        p3 = self.lat_c3(c3) # [B, 256, H/16, W/16]
        p2 = self.lat_c2(c2) # [B, 256, H/8,  W/8]
        
        # === 2. FPN (Top-Down): Deep -> Shallow ===
        # P4 -> P3
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        # P3 -> P2
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        
        # 平滑
        f4 = self.fpn_conv4(p4)
        f3 = self.fpn_conv3(p3)
        f2 = self.fpn_conv2(p2)
        
        # === 3. PAN (Bottom-Up): Shallow -> Deep ===
        # N2 (就是 f2)
        n2 = f2
        
        # N2 -> N3 (Downsample + Add)
        n3 = self.pan_conv3(f3 + self.downsample_conv2(n2))
        
        # N3 -> N4 (Downsample + Add)
        n4 = self.pan_conv4(f4 + self.downsample_conv3(n3))
        
        # === 4. Fusion ===
        # 将所有特征统一到相同尺寸 (通常统一到中间尺寸 N3 或者最小尺寸 N4)
        # 这里为了计算效率，统一对齐到 N4 (H/32)
        target_h, target_w = n4.shape[-2:]
        
        n2_down = F.adaptive_avg_pool2d(n2, (target_h, target_w))
        n3_down = F.adaptive_avg_pool2d(n3, (target_h, target_w))
        
        # Channel Concatenation: [B, 768, H/32, W/32]
        fused_feat = torch.cat([n2_down, n3_down, n4], dim=1)
        
        # === 5. Regression Head ===
        # 关键修改：不要由 (H,W) 直接变为 (1,1)
        # 而是变为 (4,4) 保留粗略的空间位置
        pooled_feat = F.adaptive_avg_pool2d(fused_feat, (self.pool_size, self.pool_size))
        
        # Flatten: [B, 768, 4, 4] -> [B, 768*16]
        flat_feat = pooled_feat.flatten(1)
        
        residual = self.reg_head(flat_feat)
        residual = residual.view(x.shape[0], -1, 3)
        
        return residual
    
if __name__ == "__main__":

    # 确保这些导入与你的实际环境一致
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = HandBackResidualNet(num_joints=21, hidden_dim=256).to(device)

    # 创建一个模拟输入：batch_size=2, 3通道, 224x224（ConvNeXt 默认输入尺寸）
    dummy_input = torch.randn(2, 3, 518, 518).to(device)

    # 前向传播
    with torch.no_grad():
        output = model(dummy_input)

    # 打印输入输出形状
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # 应为 [2, 21, 3]

    # 检查输出是否接近零（因最后一层 bias 初始化为 0，weight 很小）
    print(f"Output mean (should be near 0): {output.mean().item():.6f}")
    print(f"Output std (should be small): {output.std().item():.6f}")

    # 可选：统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")