import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    """基础卷积块: Conv -> BN -> ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ViTPoseFusionBlock(nn.Module):
    """
    适配 ViT 的 FPN+PAN 特征融合模块
    假设输入的特征图分辨率相同 (例如 16x16)
    """
    def __init__(self, in_channels_list, hidden_dim=256, out_dim=512):
        """
        Args:
            in_channels_list: 输入特征通道列表，顺序应为 [浅层, 中层, 深层]
            hidden_dim: 中间处理的通道维度
            out_dim: 最终输出的通道维度
        """
        super().__init__()
        
        # 1. 侧向连接 (Lateral Connections) - 将输入映射到统一维度
        # 对应输入的 [shallow, mid, deep]
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, kernel_size=1) for c in in_channels_list
        ])
        
        # 2. FPN 路径 (Top-Down): 深 -> 浅
        # 这里的卷积用于在融合后平滑特征
        self.fpn_convs = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim) for _ in range(len(in_channels_list))
        ])

        # 3. PAN 路径 (Bottom-Up): 浅 -> 深
        # 用于将低层的精细定位信息传回高层
        # 注意：因为ViT特征图大小相同，这里不需要下采样(stride=2)，直接融合即可
        self.pan_convs = nn.ModuleList([
            ConvBlock(hidden_dim, hidden_dim) for _ in range(len(in_channels_list))
        ])
        
        # 4. 最终融合输出
        # 将三层 output 拼接后压缩
        total_channels = hidden_dim * len(in_channels_list)
        self.final_conv = nn.Sequential(
            ConvBlock(total_channels, total_channels),
            nn.Conv2d(total_channels, out_dim, kernel_size=1)
        )

    def forward(self, features):
        """
        Args:
            features: list of tensors [feat_shallow, feat_mid, feat_deep]
        """
        # 1. 统一通道维度
        projs = [conv(f) for conv, f in zip(self.laterals, features)]
        p_shallow, p_mid, p_deep = projs[0], projs[1], projs[2]
        
        # --- FPN 阶段 (Top-Down) ---
        # 深层特征加到中层，中层加到浅层
        p_mid_fpn = p_mid + p_deep
        p_shallow_fpn = p_shallow + p_mid_fpn
        
        # 对融合后的特征进行卷积平滑
        f_shallow = self.fpn_convs[0](p_shallow_fpn)
        f_mid = self.fpn_convs[1](p_mid_fpn)
        f_deep = self.fpn_convs[2](p_deep) # 深层作为起点，直接过卷积
        
        # --- PAN 阶段 (Bottom-Up) ---
        # 浅层特征加回中层，中层加回深层
        n_shallow = f_shallow # 浅层作为起点
        n_mid = self.pan_convs[1](f_mid + n_shallow)
        n_deep = self.pan_convs[2](f_deep + n_mid)
        
        # --- 最终输出 ---
        # 拼接所有层级的特征，兼顾不同尺度的语义
        combined = torch.cat([n_shallow, n_mid, n_deep], dim=1)
        output = self.final_conv(combined)
        
        return output


class LightweightBackFusion(nn.Module):
    """
    专为手背图像设计的轻量级特征融合模块。
    采用极简的 Top-Down FPN 和双线性插值上采样，降低计算量并防止对噪声过拟合。
    """
    def __init__(self, in_channels, hidden_dim=128, out_dim=512):
        super().__init__()
        # 1. 降维映射 (减少计算量)
        self.projs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1) for _ in range(3)
        ])
        
        # 2. 轻量级平滑卷积 (使用深度可分离卷积或组卷积降低参数)
        self.smooth = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim) 
            for _ in range(3)
        ])
        
        # 3. 最终的简单上采样 (抛弃沉重的 Deconv，直接使用双线性插值 + 1x1 Conv)
        # 假设输入特征图是 16x16，我们需要放大 4 倍到 64x64
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: [feat_shallow, feat_mid, feat_deep]
        p_shallow = self.projs[0](features[0])
        p_mid = self.projs[1](features[1])
        p_deep = self.projs[2](features[2])
        
        # 极简 Top-Down 融合 (只取大尺度特征，忽略 PANet 的 Bottom-Up)
        f_mid = p_mid + p_deep
        f_shallow = p_shallow + f_mid
        
        # 平滑处理
        f_shallow = self.smooth[0](f_shallow)
        
        # 直接对融合了全局信息的高分辨率浅层特征进行上采样
        out = self.final_upsample(f_shallow)
        return out

# class PoseEstimationModel(nn.Module):
#     def __init__(self, num_joints=21, feature_dim=768, img_size=224, patch_size=14, layers=[3, 6, -1]):
#         """
#         Args:
#             num_joints: 关节数量 (默认 21)
#             feature_dim: ViT 模型的特征维度 (DINOv2-Base 为 768, Small 为 384)
#             img_size: 输入图片大小 (默认 224)
#             patch_size: ViT Patch 大小 (默认 14)
#             layers: 使用的层索引 (必须与 ViTFeatureExtractor 中的 layers_to_extract 一致)
#         """
#         super(PoseEstimationModel, self).__init__()
        
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = img_size // patch_size # 例如 224/14 = 16
#         self.layers = layers
        
#         # 1. 特征压缩层 (Projectors)
#         # 将不同层的特征维度统一压缩到一个较小的维度 (例如 256)，减少计算量
#         reduced_dim = 256
#         self.projectors = nn.ModuleDict({
#             str(l): nn.Conv2d(feature_dim, reduced_dim, kernel_size=1) 
#             for l in layers
#         })
        
#         # 2. 特征融合层 (Fusion)
#         # 这里采用简单的拼接 (Concat) 后卷积融合
#         # 输入通道数 = 层数 * reduced_dim
#         concat_dim = len(layers) * reduced_dim
        
#         self.fusion_block = nn.Sequential(
#             nn.Conv2d(concat_dim, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True)
#         )
        
#         # 3. 回归头 (Regression Head)
#         # 展平后通过 MLP 回归坐标
#         # 特征图大小为 grid_size * grid_size (例如 16*16)
#         self.flatten_dim = 512 * (self.grid_size ** 2)
        
#         self.regressor = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(self.flatten_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3), # 防止过拟合
#             nn.Linear(1024, num_joints * 3)
#         )
        
#         self.num_joints = num_joints

#     def forward(self, features_dict):
#         """
#         Args:
#             features_dict: {layer_idx: [Batch, Seq_Len, Dim]} 
#                            来自 ViTFeatureExtractor 的输出
#         """
#         processed_features = []
        
#         # 遍历指定的层进行处理
#         for layer_id in self.layers:
#             # 获取特征: [Batch, N_patches+1, Dim]
#             feat = features_dict[layer_id] 
            
#             # 1. 剔除 CLS Token (通常是第0个)，只保留 Patch Tokens
#             # shape: [Batch, N_patches, Dim]
#             patch_tokens = feat[:, 1:, :] 
            
#             # 2. 转换维度顺序以适应 Conv2d: [Batch, Dim, N_patches]
#             patch_tokens = patch_tokens.transpose(1, 2)
            
#             # 3. Reshape 回二维特征图: [Batch, Dim, Grid, Grid]
#             # 例如 [Batch, 768, 16, 16]
#             B, C, N = patch_tokens.shape
#             H = W = int(math.sqrt(N)) # 自动计算，通常是 16
#             feature_map = patch_tokens.view(B, C, H, W)
            
#             # 4. 通过 1x1 卷积压缩通道
#             proj_feat = self.projectors[str(layer_id)](feature_map)
#             processed_features.append(proj_feat)
        
#         # 5. 在通道维度拼接: [Batch, 256*3, 16, 16]
#         fused_map = torch.cat(processed_features, dim=1)
        
#         # 6. 卷积融合
#         fused_map = self.fusion_block(fused_map)
        
#         # 7. 回归坐标
#         output = self.regressor(fused_map)
        
#         # Reshape: [Batch, Num_Joints, 3]
#         output = output.view(-1, self.num_joints, 3)
        
#         return output



class PoseEstimationModel(nn.Module):
    def __init__(self, num_joints=21, feature_dim=768, img_size=224, patch_size=14, layers=[3, 6, -1]):
        super(PoseEstimationModel, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.layers = layers
        
        # --- 改进点 1: 移除分散的 projectors，集成到 Fusion 模块中 ---
        
        # --- 改进点 2: 使用 PANet 风格的融合模块 ---
        # 假设 layers 有 3 层，我们将它们视为 [浅, 中, 深]
        # 输入维度列表: [768, 768, 768] (对于 DINOv2 Base)
        in_channels_list = [feature_dim] * len(layers)
        hidden_dim = 256
        fusion_out_dim = 512
        
        self.fusion_block = ViTPoseFusionBlock(
            in_channels_list=in_channels_list,
            hidden_dim=hidden_dim,
            out_dim=fusion_out_dim
        )
        
        # --- 改进点 3: 回归头 ---
        self.flatten_dim = fusion_out_dim * (self.grid_size ** 2)
        self.num_joints = num_joints
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.LayerNorm(1024), # BN 在 MLP 中通常不如 LayerNorm 稳定
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), 
            nn.Linear(1024, num_joints * 3)
        )

    def forward(self, features_dict):
        # 1. 提取并预处理特征
        extracted_features = []
        for layer_id in self.layers:
            feat = features_dict[layer_id]
            patch_tokens = feat[:, 1:, :] # 去除 CLS
            patch_tokens = patch_tokens.transpose(1, 2) # [B, C, N]
            
            B, C, N = patch_tokens.shape
            H = W = int(math.sqrt(N))
            feature_map = patch_tokens.view(B, C, H, W)
            
            extracted_features.append(feature_map)
        
        # 2. 特征融合 (核心改进)
        # 输入: [Shallow, Mid, Deep]
        fused_map = self.fusion_block(extracted_features)
        
        # 3. 回归坐标
        output = self.regressor(fused_map)
        output = output.view(-1, self.num_joints, 3)
        
        return output
