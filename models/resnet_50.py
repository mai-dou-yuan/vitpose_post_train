import torch
import torch.nn as nn
import torchvision.models as models

# 如果需要空间特征图 (Feature Map)
class ResNet50Spatial(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base = models.resnet50(weights=weights)
        
        # 去掉最后两层 (AvgPool 和 FC)
        self.backbone = nn.Sequential(*list(base.children())[:-2])

    def forward(self, x):
        return self.backbone(x) # 输出 (B, 2048, H/32, W/32)
    
if __name__ == '__main__':
    # 实例化
    encoder = ResNet50Spatial(pretrained=True)
    
    # 模拟输入 (Batch=2, RGB, 224x224)
    x = torch.randn(2, 3, 224, 224)
    
    # 提取特征
    features = encoder(x)
    print(features.shape)  # 输出: torch.Size([2, 2048, 7, 7])