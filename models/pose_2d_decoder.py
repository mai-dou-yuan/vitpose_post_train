import torch
import torch.nn as nn
import math

class ViTPose2D_Decoder(nn.Module):
    def __init__(self, 
                 in_channels=768,   # ViT Base 的输出维度
                 hidden_channels=256, 
                 out_channels=17,   # 关键点数量 (例如 COCO 是 17, 手部通常是 21)
                 patch_size=16,     # ViT 的 patch size
                 img_size=(224, 224)): # 原始输入图片的尺寸 (H, W)
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        
        # 计算 Feature Map 的形状 (H/16, W/16)
        # 如果输入是 256x192, 这里的 grid_size 就是 (16, 12)
        # 如果输入是 256x256, 这里的 grid_size 就是 (16, 16)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)

        # -----------------------------------------------------------
        # 对应你图片中的结构 (c): 
        # Deconv -> BN -> ReLU -> Deconv -> BN -> ReLU -> Predictor
        # -----------------------------------------------------------
        
        # 第一层 Deconv Block (上采样 2倍)
        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 第二层 Deconv Block (上采样 2倍)
        self.deconv_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 最终 Predictor (1x1 卷积得到 Heatmap)
        self.final_layer = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        """
        输入 x: 来自 ViT 的深层特征，形状 [B, N, C]
               例如 [16, 257, 768] (包含 1 个 cls_token)
        """
        B, N, C = x.shape
        
        # 1. 去掉 Class Token
        # ViT 输出通常 index 0 是 cls_token，后面 256 个是 patch tokens
        x = x[:, 1:]  # shape: [B, 256, 768]
        
        # 2. Reshape: Sequence -> Image
        # 需要将 flat 的序列变回 (B, C, H, W)
        h, w = self.grid_size
        
        # 检查维度是否匹配，防止 reshape 报错
        assert x.shape[1] == h * w, f"特征数量 {x.shape[1]} 与预设网格 {h}x{w} 不匹配"
        
        # 变换过程: [B, N, C] -> [B, H, W, C] -> [B, C, H, W]
        x = x.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()
        
        # 3. 通过 Decoder (对应图片流程)
        x = self.deconv_layer1(x) # 放大 2x
        x = self.deconv_layer2(x) # 放大 2x
        
        # 4. Predictor
        heatmap = self.final_layer(x) # 得到最终 Heatmap
        
        return heatmap

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 假设你的设置
    batch_size = 16
    seq_len = 257 # 1 cls + 16*16 patches
    dim = 768
    num_joints = 21 # 假设做手部 pose
    
    # 模拟你的输入数据
    # dict_keys([3, 6, -1]), torch.Size([16, 257, 768])
    deep_feature = torch.randn(batch_size, seq_len, dim)
    
    # 实例化模型 (假设输入图片是 256x256, patch=16)
    decoder = ViTPose2D_Decoder(
        in_channels=768, 
        hidden_channels=256, 
        out_channels=num_joints,
        patch_size=16,
        img_size=(256, 256) 
    )
    
    # 前向传播
    output = decoder(deep_feature)
    
    print(f"输入形状: {deep_feature.shape}") # [16, 257, 768]
    print(f"输出形状: {output.shape}")       # [16, 21, 64, 64]
    
    # 输出通常是原图的 1/4 大小 (256/4 = 64)
    # 之后通常会对 output 做 softmax 或 argmax 获取坐标