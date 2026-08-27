import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class FastViTFeatureExtractor(nn.Module):
    def __init__(self, freeze_backbone=False, feature_dim=768, checkpoint_path=None):
        super().__init__()
        
        # 1. 核心修改：利用 pretrained_cfg_overlay 替换本地路径
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"✅ Loading FastViT weights locally via overlay: {checkpoint_path}")
            self.backbone = timm.create_model(
                'fastvit_ma36.apple_in1k', 
                pretrained=True,  # 必须设为 True，触发内部正常的权重加载顺序
                pretrained_cfg_overlay={'file': checkpoint_path}, # 关键：将下载源重定向到本地文件
                features_only=True
            )
        else:
            print("⚠️ Local checkpoint not found, downloading from internet...")
            self.backbone = timm.create_model(
                'fastvit_ma36.apple_in1k', 
                pretrained=True, 
                features_only=True
            )

        # 2. 冻结参数 (根据需要)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. 投影层：将 FastViT 的特征维度对齐到 feature_dim (768)
        self.proj1 = nn.Conv2d(152, feature_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(304, feature_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(608, feature_dim, kernel_size=1)

    def forward(self, x):
        target_h, target_w = x.shape[2] // 16, x.shape[3] // 16
        
        features = self.backbone(x)
        f1 = features[1] 
        f2 = features[2] 
        f3 = features[3] 
        
        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        f3 = self.proj3(f3)
        
        f1 = F.adaptive_avg_pool2d(f1, (target_h, target_w)) 
        f2 = F.interpolate(f2, size=(target_h, target_w), mode='bilinear', align_corners=False) 
        f3 = F.interpolate(f3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return [f1, f2, f3]