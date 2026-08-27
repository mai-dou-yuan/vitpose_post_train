import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ConvNormRelu(nn.Module):
    """辅助用的卷积-BN-ReLU块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class LightweightBackFeatureExtractor_withfpn(nn.Module):
    """
    专门用于手背的轻量级特征提取器
    结构: ResNet18 Backbone + Simple FPN Fusion
    修改: 仅融合到 Layer2，不再融合 Layer1
    输出: [B, out_channels, H/8, W/8] (Stride 8)
    """
    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        
        # 1. 加载 ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        back_model = resnet18(weights=weights)
        
        # 提取 ResNet 的层
        self.conv1 = back_model.conv1
        self.bn1 = back_model.bn1
        self.relu = back_model.relu
        self.maxpool = back_model.maxpool
        
        self.layer1 = back_model.layer1 # Stride 4,  64 ch
        self.layer2 = back_model.layer2 # Stride 8,  128 ch
        self.layer3 = back_model.layer3 # Stride 16, 256 ch
        self.layer4 = back_model.layer4 # Stride 32, 512 ch
        
        # 2. FPN Lateral Layers (1x1 Conv 降维/对齐)
        mid_channels = out_channels 
        
        self.lat_layer4 = nn.Conv2d(512, mid_channels, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(256, mid_channels, kernel_size=1)
        self.lat_layer2 = nn.Conv2d(128, mid_channels, kernel_size=1)
        # 删除了 lat_layer1，因为我们不需要融合到 stride 4
        
        # 3. Smooth Layers (3x3 Conv 消除上采样混叠)
        self.smooth_layer3 = ConvNormRelu(mid_channels, mid_channels, kernel_size=3)
        self.smooth_layer2 = ConvNormRelu(mid_channels, mid_channels, kernel_size=3) # 这将是最终特征前的处理
        # 删除了 smooth_layer1
        
        # 4. 最终对齐输出层
        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: [B, 3, H, W]
        Return: [B, out_channels, H/8, W/8]
        """
        # --- Bottom-up Pathway (ResNet) ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.maxpool(x) # Stride 4
        
        c1 = self.layer1(x)  # [B, 64,  H/4,  W/4]  (c1 仍需计算，因为它是 c2 的输入)
        c2 = self.layer2(c1) # [B, 128, H/8,  W/8]  <-- 我们的目标分辨率
        c3 = self.layer3(c2) # [B, 256, H/16, W/16]
        c4 = self.layer4(c3) # [B, 512, H/32, W/32]
        
        # --- Top-down Pathway (FPN) ---
        
        # P4
        p4 = self.lat_layer4(c4)
        
        # P3 = P4_up + C3_lat
        p3_up = F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lat_layer3(c3) + p3_up
        p3 = self.smooth_layer3(p3)
        
        # P2 = P3_up + C2_lat (Stride 8)
        p2_up = F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lat_layer2(c2) + p2_up
        p2 = self.smooth_layer2(p2)
        
        # 这里的 P2 已经是 [H/8, W/8]，我们不再继续上采样到 P1
        
        # --- Final Output ---
        out = self.final_conv(p2)
        
        return out
    

class LightweightBackFeatureExtractor(nn.Module):
    """
    轻量级特征提取器 (标准 ResNet 流程)
    结构: ResNet18 Backbone (全阶段) + Channel Projection
    输出: [B, out_channels, H/32, W/32] (Stride 32)
    """
    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        
        # 1. 加载 ResNet18 完整模型
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        back_model = resnet18(weights=weights)
        
        # 2. 提取所有 Backbone 层
        self.conv1 = back_model.conv1
        self.bn1 = back_model.bn1
        self.relu = back_model.relu
        self.maxpool = back_model.maxpool
        
        self.layer1 = back_model.layer1 # Stride 4,  64 ch
        self.layer2 = back_model.layer2 # Stride 8,  128 ch
        self.layer3 = back_model.layer3 # Stride 16, 256 ch
        self.layer4 = back_model.layer4 # Stride 32, 512 ch
        
        # 3. 最终投影层 (将 ResNet18 最后的 512 通道映射到指定的 out_channels)
        # 如果你希望直接输出 512 通道，也可以去掉这一层。
        self.project = nn.Sequential(
            nn.Conv2d(512, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        x: [B, 3, H, W]
        Return: [B, out_channels, H/32, W/32]
        """
        # --- Bottom-up Pathway ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # Stride 4
        
        x = self.layer1(x)  # Stride 4
        x = self.layer2(x)  # Stride 8
        x = self.layer3(x)  # Stride 16
        x = self.layer4(x)  # Stride 32
        
        # --- Final Output ---
        out = self.project(x)
        
        return out
if __name__ == "__main__":
    # 1. 模拟超参数
    batch_size = 2
    input_h, input_w = 224, 224
    target_channels = 256
    
    # 2. 实例化模型
    # 注意：首次运行 pretrained=True 会下载 ResNet18 权重，如果网络环境不佳可以设为 False
    model = LightweightBackFeatureExtractor(out_channels=target_channels, pretrained=False)
    model.eval() # 切换到评估模式
    
    # 3. 构造模拟输入 (B, C, H, W)
    dummy_input = torch.randn(batch_size, 3, input_h, input_w)
    
    # 4. 执行推理
    with torch.no_grad():
        output = model(dummy_input)
    
    # 5. 验证与输出结果
    print("="*30)
    print("模型测试成功！")
    print(f"输入尺寸: {dummy_input.shape}")  # [2, 3, 224, 224]
    print(f"输出尺寸: {output.shape}")       # [2, 256, 7, 7]
    print("="*30)
    
    # 计算尺寸缩放比例
    stride_h = input_h // output.shape[2]
    stride_w = input_w // output.shape[3]
    
    print(f"检测到的 Stride: {stride_h} (预期: 32)")
    print(f"检测到的输出通道: {output.shape[1]} (预期: {target_channels})")
    
    # 统计模型参数量 (百万级别)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型总参数量: {total_params:.2f} M")