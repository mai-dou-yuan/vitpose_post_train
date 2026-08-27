# pl_system.py
import os
import pdb
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pretrain_vitpose_pkl_and_call.load_vitpose_plus_backbone import (
    ViTPosePlusBackbone,
)
from models.HandBackResidualNet import HandBackResidualNet
from models.model_PoseEstimationModel import PoseEstimationModel

from models.GatedMultiHeadAttention import GatedMultiHeadAttention
from models.ChannelWiseGatedFusion import ChannelWiseGatedFusion
from utils.PositionEmbeddingSine import PositionEmbeddingSine
from models.Pose3DRegressionHead import Pose3DRegressionHead
from utils.project_3d_to_2d_batch_with_distortion import project_3d_to_2d_batch_with_distortion
from models.LightweightBackFeatureExtractor import LightweightBackFeatureExtractor
from torch.utils.checkpoint import checkpoint
from mano_joints_package import MeshToJoints
from mesh_regressor_package import MeshRegressor


class ViTPosePlusBBackbone(nn.Module):
    """ViTPose++-B backbone adapter for the existing Graphormer input contract."""

    model_name = "vitpose_plus_base"

    def __init__(self, config_path, checkpoint_path, dataset_source=5):
        super().__init__()
        self.backbone = ViTPosePlusBackbone(
            config_path=Path(config_path).expanduser().resolve(),
            checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
            freeze=False,
            device="cpu",
            interpolate_pos_encoding=False,
        )
        self.dataset_source = int(dataset_source)
        if not 0 <= self.dataset_source < self.backbone.backbone.num_expert:
            raise ValueError(
                f"dataset_source must be in [0, {self.backbone.backbone.num_expert - 1}], "
                f"got {self.dataset_source}"
            )
        self.feature_dim = int(self.backbone.out_channels)
        self.config_path = str(self.backbone.config_path)
        self.checkpoint_path = str(self.backbone.checkpoint_path)

    def forward(self, image):
        # FreiHAND supplies top-down RGB crops in [0, 1]. Reuse the checkpoint's
        # canonical 256x192 resize and ImageNet normalization exactly once here.
        model_device = self.backbone.backbone.patch_embed.proj.weight.device
        self.backbone.device = model_device
        normalized_image = self.backbone.preprocess(
            image, input_color="RGB", input_range="0_1"
        )
        return self.backbone(
            normalized_image, dataset_source=self.dataset_source
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, bias=False):
        super().__init__()
        # 对应图中的 Gate 路径 和 Up 路径
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=bias)
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=bias)
        
        # 对应图中的 Down 路径
        self.w_down = nn.Linear(hidden_dim, d_model, bias=bias)
        self.act = nn.SiLU() 

    def forward(self, x):
        return self.w_down(self.act(self.w_gate(x)) * self.w_up(x))


def refinement_attention_mode(layer_index):
    """Return the repeating local, local, full refinement schedule."""
    return "full" if layer_index % 3 == 2 else "local"


def project_camera_joints_to_image(joints_3d, cam_k, image_size, eps=1e-6):
    """Project camera-coordinate XYZ joints to input-image pixel coordinates.

    Invalid/behind-camera predictions are mapped to the image centre so that an
    unstable early prediction cannot feed NaNs into ``grid_sample``. Valid
    projections remain differentiable with respect to the 3D joints.
    """
    if joints_3d.ndim != 3 or joints_3d.shape[-1] != 3:
        raise ValueError(f"joints_3d must be [B, N, 3], got {tuple(joints_3d.shape)}")
    if cam_k.ndim != 3 or cam_k.shape[-2:] != (3, 3):
        raise ValueError(f"cam_k must be [B, 3, 3], got {tuple(cam_k.shape)}")

    image_h, image_w = image_size
    points_homo = torch.matmul(joints_3d, cam_k.transpose(1, 2))
    depth = points_homo[..., 2:3]
    valid = torch.isfinite(points_homo).all(dim=-1, keepdim=True) & (depth > eps)
    safe_depth = torch.where(valid, depth, torch.ones_like(depth))
    points_2d = points_homo[..., :2] / safe_depth
    image_centre = points_2d.new_tensor(
        [(float(image_w) - 1.0) / 2.0, (float(image_h) - 1.0) / 2.0]
    )
    return torch.where(valid.expand_as(points_2d), points_2d, image_centre)


def normalized_crop_to_pixel(points, image_size):
    """Convert normalized crop xy coordinates to crop pixel coordinates.

    ``points[..., 0]`` and ``points[..., 1]`` are normalized independently by
    crop width and height.  This is intentionally separate from the ViTPose
    internal 256x192 resize and its usually 16x12 feature grid.
    """
    if points.ndim < 2 or points.shape[-1] != 2:
        raise ValueError(f"points must end in xy coordinates, got {tuple(points.shape)}")
    image_h, image_w = image_size
    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Invalid image_size: {image_size}")
    scale = points.new_tensor((float(image_w) - 1.0, float(image_h) - 1.0))
    return points * scale


class Initial2DCoordinateHead(nn.Module):
    """Regress crop-normalized joint coordinates from a compact spatial grid.

    The adaptive grid retains coarse image layout without constructing one
    heatmap per joint.  Output coordinates are ordered as (x, y).
    """

    def __init__(self, in_channels=256, num_joints=21,
                 bottleneck_channels=16, pooled_size=(4, 3), hidden_dim=32,
                 dropout=0.1):
        super().__init__()
        if bottleneck_channels < 1:
            raise ValueError("bottleneck_channels must be positive")
        if bottleneck_channels % 4 != 0:
            raise ValueError("bottleneck_channels must be divisible by 4")
        if len(pooled_size) != 2 or any(int(size) < 1 for size in pooled_size):
            raise ValueError("pooled_size must contain two positive integers")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_joints = int(num_joints)
        self.pooled_size = tuple(int(size) for size in pooled_size)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            ),
            nn.GroupNorm(4, bottleneck_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=1,
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.GroupNorm(4, bottleneck_channels),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(self.pooled_size),
        )
        flattened_dim = (
            bottleneck_channels * self.pooled_size[0] * self.pooled_size[1]
        )
        self.coordinate_regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_joints * 2),
        )
        nn.init.normal_(
            self.coordinate_regressor[-1].weight, mean=0.0, std=0.001
        )
        nn.init.zeros_(self.coordinate_regressor[-1].bias)

    def forward(self, feature_map):
        if feature_map.ndim != 4:
            raise ValueError(
                f"feature_map must be [B,C,H,W], got {tuple(feature_map.shape)}"
            )
        features = self.spatial_encoder(feature_map)
        coordinates = self.coordinate_regressor(features)
        coordinates = coordinates.view(feature_map.shape[0], self.num_joints, 2)
        return coordinates.sigmoid()

class PoseRefinementLayer(nn.Module):
    """
    Cross Attention Layer: 
    Query = Joint Tokens
    Key/Value = Global Feature Map (Memory)
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.2,
                 attention_mode="full", local_grid_size=5,
                 local_grid_radius=2.0):
        super().__init__()
        if attention_mode not in ("local", "full"):
            raise ValueError(f"Unsupported cross-attention mode: {attention_mode}")
        if local_grid_size < 1:
            raise ValueError("local_grid_size must be positive")
        if local_grid_radius < 0:
            raise ValueError("local_grid_radius must be non-negative")

        self.attention_mode = attention_mode
        self.local_grid_size = int(local_grid_size)
        self.local_grid_radius = float(local_grid_radius)
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
        
        self.norm_attn_in = RMSNorm(d_model)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)

        self.cross_attention = GatedMultiHeadAttention(
            d_model=d_model, 
            n_head=n_head, 
            dropout=dropout
        )
        
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def _sample_local_memory(self, memory, reference_points, image_size):
        """Sample a feature-space neighbourhood for every joint.

        Args:
            memory: [B, C, Hf, Wf]
            reference_points: [B, N, 2] input-image pixels in (x, y) order
        Returns:
            [B, N, K, C], where K = local_grid_size ** 2
        """
        if memory.ndim != 4:
            raise ValueError("Local cross-attention requires memory [B, C, H, W]")
        if reference_points is None or reference_points.ndim != 3:
            raise ValueError("Local cross-attention requires reference_points [B, N, 2]")
        if reference_points.shape[0] != memory.shape[0] or reference_points.shape[-1] != 2:
            raise ValueError(
                "reference_points must match memory batch and end in xy coordinates"
            )

        _, _, feature_h, feature_w = memory.shape
        image_h, image_w = image_size
        if image_h <= 0 or image_w <= 0:
            raise ValueError(f"Invalid image_size: {image_size}")

        # With align_corners=False this maps input pixel centres to the same
        # normalized image coordinates used by the resized ViTPose feature map.
        points = reference_points.to(device=memory.device, dtype=memory.dtype)
        finite = torch.isfinite(points)
        centre_x = points.new_tensor((float(image_w) - 1.0) / 2.0)
        centre_y = points.new_tensor((float(image_h) - 1.0) / 2.0)
        x = torch.where(finite[..., 0], points[..., 0], centre_x)
        y = torch.where(finite[..., 1], points[..., 1], centre_y)
        x = x.clamp(0.0, float(image_w) - 1.0)
        y = y.clamp(0.0, float(image_h) - 1.0)
        norm_x = 2.0 * (x + 0.5) / float(image_w) - 1.0
        norm_y = 2.0 * (y + 0.5) / float(image_h) - 1.0
        centres = torch.stack((norm_x, norm_y), dim=-1)  # [B, N, 2]

        offsets_1d = torch.linspace(
            -self.local_grid_radius,
            self.local_grid_radius,
            self.local_grid_size,
            device=memory.device,
            dtype=memory.dtype,
        )
        offset_y, offset_x = torch.meshgrid(offsets_1d, offsets_1d, indexing="ij")
        offsets = torch.stack(
            (2.0 * offset_x / float(feature_w), 2.0 * offset_y / float(feature_h)),
            dim=-1,
        ).reshape(1, 1, -1, 2)
        sample_grid = centres.unsqueeze(2) + offsets  # [B, N, K, 2]

        sampled = F.grid_sample(
            memory,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )  # [B, C, N, K]
        return sampled.permute(0, 2, 3, 1).contiguous()

    def forward(self, tgt, memory, query_pos=None, memory_pos=None,
                reference_points=None, image_size=None):
        """
        tgt:        [B, N, C]       - Joint Tokens
        memory:     [B, C, H, W]    - Global Feature Map
        query_pos:  [B, N, C]       - Joint Token Positional Encoding
        memory_pos: [B, C, H, W]    - Feature Map Positional Encoding
        """
        # ====== Block 1: Attention (Cross-Attention) ======
        tgt_norm = self.norm_attn_in(tgt)
        
        # Query Path
        q_content = self.norm_q(tgt_norm) 
        query = q_content + (query_pos if query_pos is not None else 0)
        
        # Key / Value Path
        if self.attention_mode == "local":
            if image_size is None:
                raise ValueError("Local cross-attention requires image_size=(H, W)")
            memory_local = self._sample_local_memory(
                memory, reference_points, image_size
            )  # [B, N, K, C]
            if memory_pos is not None:
                memory_pos_local = self._sample_local_memory(
                    memory_pos, reference_points, image_size
                )
            else:
                memory_pos_local = 0

            bsz, num_joints, num_samples, channels = memory_local.shape
            memory_flatten = memory_local.reshape(
                bsz * num_joints, num_samples, channels
            )
            if isinstance(memory_pos_local, torch.Tensor):
                m_pos_flatten = memory_pos_local.reshape(
                    bsz * num_joints, num_samples, channels
                )
            else:
                m_pos_flatten = 0
        elif memory.dim() == 4:
            # [B, C, H, W] -> [B, H*W, C]
            memory_flatten = memory.permute(0, 2, 3, 1).flatten(1, 2)
            if memory_pos is not None:
                m_pos_flatten = memory_pos.permute(0, 2, 3, 1).flatten(1, 2)
            else:
                m_pos_flatten = 0
        else:
            memory_flatten = memory
            m_pos_flatten = memory_pos if memory_pos is not None else 0

        k_content = self.norm_k(memory_flatten) 
        key = k_content + m_pos_flatten
        value = memory_flatten
        
        if self.attention_mode == "local":
            # Each joint is its own one-query attention problem. Heads remain
            # internal to GatedMultiHeadAttention: Q=[B*N,1,C], KV=[B*N,K,C].
            query_local = query.reshape(bsz * num_joints, 1, channels)
            attn_out = self.cross_attention(
                query=query_local, key=key, value=value, is_causal=False
            ).reshape(bsz, num_joints, channels)
        else:
            attn_out = self.cross_attention(
                query=query,
                key=key,
                value=value,
                is_causal=False
            )
        
        tgt = tgt + self.dropout(attn_out)
        
        # ====== Block 2: MLP (SwiGLU) ======
        mlp_in = self.norm_mlp_in(tgt)
        mlp_out = self.mlp(mlp_in)
        tgt = tgt + self.dropout(mlp_out)
        
        return tgt

class PoseSelfAttentionLayer(nn.Module):
    """
    Self Attention Layer:
    Joint Tokens attend to themselves
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.2):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)

        self.norm_attn_in = RMSNorm(d_model)
        self.norm_q = RMSNorm(d_model)
        self.norm_k = RMSNorm(d_model)

        self.self_attention = GatedMultiHeadAttention(
            d_model=d_model, 
            n_head=n_head, 
            dropout=dropout
        )
        
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None):
        """
        x:   [B, N, C] - Joint Tokens
        pos: [B, N, C] - Joint Token Positional Encoding
        """
        # Block 1: Self-Attention ---
        residual = x
        x = self.norm_attn_in(x)
        
        q = self.norm_q(x) + (pos if pos is not None else 0)
        k = self.norm_k(x) + (pos if pos is not None else 0)
        v = x
        
        attn_out = self.self_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False
        )
        x = residual + self.dropout(attn_out)

        # Block 2: MLP (SwiGLU) ---
        residual = x
        x = self.norm_mlp_in(x)
        x = self.mlp(x)
        x = residual + self.dropout(x)
        
        return x


class HandGraphormerLayer(nn.Module):
    """
    结合了 Graphormer 空间距离偏置、同指运动学偏置与门控机制的增强版全局注意力层。
    针对混合精度训练和早期训练稳定性进行了深度优化，并使用 PyTorch 原生 SDPA 加速。
    """
    def __init__(self, d_model, n_head=8, dim_feedforward=None, dropout=0.1, num_joints=21):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = int(d_model * 4)
            
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"


        # 1. 结构偏置 A：最短路径距离 (Shortest Path Distance)

        edges = [
            (0,1), (1,2), (2,3), (3,4),       # Thumb
            (0,5), (5,6), (6,7), (7,8),       # Index
            (0,9), (9,10), (10,11), (11,12),  # Middle
            (0,13), (13,14), (14,15), (15,16),# Ring
            (0,17), (17,18), (18,19), (19,20) # Pinky
        ]
        
        spd = torch.full((num_joints, num_joints), float('inf'))
        for i in range(num_joints): spd[i, i] = 0
        for i, j in edges:
            spd[i, j] = 1
            spd[j, i] = 1
            
        for k in range(num_joints):
            for i in range(num_joints):
                for j in range(num_joints):
                    if spd[i, k] + spd[k, j] < spd[i, j]:
                        spd[i, j] = spd[i, k] + spd[k, j]
                        
        max_dist = int(spd.max().item()) 
        self.register_buffer('spatial_distance', spd.long())
        
        # 【优化】0 初始化，保证训练初期行为等价于标准 Self-Attention
        self.spatial_bias_table = nn.Embedding(max_dist + 1, n_head)
        nn.init.zeros_(self.spatial_bias_table.weight)


        # 2. 结构偏置 B：同指偏置 (Same-Finger Bias)

        fingers = [
            [1, 2, 3, 4],         # Thumb
            [5, 6, 7, 8],         # Index
            [9, 10, 11, 12],      # Middle
            [13, 14, 15, 16],     # Ring
            [17, 18, 19, 20]      # Pinky
        ]
        
        same_finger = torch.zeros((num_joints, num_joints), dtype=torch.long)
        for finger in fingers:
            for i in finger:
                for j in finger:
                    same_finger[i, j] = 1
        # 手腕 (0) 与所有手指不被视为"同指"
        for i in range(num_joints): same_finger[i, i] = 1 
        
        self.register_buffer('same_finger_index', same_finger)
        self.same_finger_bias = nn.Embedding(2, n_head)
        nn.init.zeros_(self.same_finger_bias.weight)

        # ==========================================
        # 3. Attention 分支
        # ==========================================
        self.norm_attn_in = RMSNorm(d_model) # 需确保外部已定义
        self.pos_scale = nn.Parameter(torch.tensor(1.0)) # 控制位置编码的强度
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        
        # 语义分离的 Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # ===== 输出门控投影 =====
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        # 初始 bias 从 2.0 降至 0.0，初始 Gate ≈ 0.5，更平稳
        nn.init.constant_(self.gate_proj.bias, 0.0) 
        
        self.linear_out = nn.Linear(d_model, d_model)

        # ==========================================
        # 4. FFN / MLP 分支
        # ==========================================
        self.norm_mlp_in = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, dim_feedforward) # 需确保外部已定义
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x, pos=None, attn_mask=None):
        """
        x:   [B, N, C]
        pos: [B, N, C]
        attn_mask: [B, N, N] 布尔型，True 代表被遮罩 (可选)
        """
        B, N, C = x.size()
        
        # 安全断言，预防未来添加 Token 时维度崩溃
        assert N <= self.spatial_distance.size(0), f"Sequence length N={N} exceeds maximum joint bias map size."
        
        residual = x
        
        # ==========================================
        # Block 1: Graphormer Global Attention
        # ==========================================
        h_in = self.norm_attn_in(x)
        if pos is not None:
            # 独立缩放，防止 pos 方差压倒内容特征
            h_in = h_in + self.pos_scale * pos
            
        gate_input = h_in 
        
        # 生成 Q, K, V
        qkv = self.qkv(h_in).chunk(3, dim=-1)
        # shape 转换: [B, N, n_head, head_dim] -> [B, n_head, N, head_dim]
        q, k, v = map(lambda t: t.view(B, N, self.n_head, self.head_dim).transpose(1, 2), qkv) 
        
        # 计算图结构偏置 
        # 获取偏置 (N < 21 的切片操作)
        bias_spd = self.spatial_bias_table(self.spatial_distance[:N, :N]) 
        bias_finger = self.same_finger_bias(self.same_finger_index[:N, :N])
        
        # 合并偏置并调整形状: [N, N, n_head] -> [1, n_head, N, N]
        bias = (bias_spd + bias_finger).permute(2, 0, 1).unsqueeze(0)
        # bias = (bias_finger).permute(2, 0, 1).unsqueeze(0)
        # bias = None
        
        # ----- 处理 Mask -----
        if attn_mask is not None:
            # attn_mask 形状: [B, N, N] -> [B, 1, N, N]
            # 你的约定是 True 代表被遮罩，因此将被遮罩位置替换为 -inf
            bias = bias.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        
        # ----- PyTorch SDPA 核心调用 -----
        # 使用 F.scaled_dot_product_attention 替代手动计算的 scores, softmax 和 matmul
        out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=bias, # 传入计算好的加性结构偏置 (内部会自动做加上 bias 的操作)
            dropout_p=self.attn_dropout.p if self.training else 0.0, # 动态控制 dropout
        )
        
        # 恢复维度: [B, n_head, N, head_dim] -> [B, N, n_head, head_dim] -> [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        
        # ----- 门控机制 -----
        gate = torch.sigmoid(self.gate_proj(gate_input)) 
        out_gated = out * gate
        
        out_attn = self.linear_out(out_gated)
        x = residual + self.proj_dropout(out_attn)

        # ==========================================
        # Block 2: FFN (SwiGLU)
        # ==========================================
        residual = x
        
        mlp_in = self.norm_mlp_in(x)
        mlp_out = self.mlp(mlp_in)
        
        x = residual + self.mlp_dropout(mlp_out)
        
        return x

class PoseLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, backbone_lr=None, backbone_freeze_epochs=0,
                 lr_warmup_epochs=5,
                 num_joints=21, vitpose_config_path=None, vitpose_checkpoint_path=None,
                 vitpose_dataset_source=5, feature_dim=None, layers=[3, 6, -1],
                 upsample_dim=256, num_refine_layers=3, use_gradient_checkpointing=True,
                 local_grid_size=5, local_grid_radius=2.0,
                 initial_2d_loss_weight=0.1,
                 joint_2d_loss_weight=0.02,
                 joint_3d_loss_weight=1.0,
                 stage_supervision_weights=(0.1, 0.3, 1.0),
                 vertices_loss_weight=10.0,
                 initial_2d_bottleneck_channels=16,
                 initial_2d_pooled_size=(4, 3),
                 initial_2d_hidden_dim=32,
                 initial_2d_dropout=0.1): # 默认层数建议设为3
        super().__init__()
        self.save_hyperparameters() 
        loss_weights = {
            "initial_2d_loss_weight": initial_2d_loss_weight,
            "joint_2d_loss_weight": joint_2d_loss_weight,
            "joint_3d_loss_weight": joint_3d_loss_weight,
            "vertices_loss_weight": vertices_loss_weight,
        }
        for name, value in loss_weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if num_joints != 21:
            raise ValueError("MeshRegressor requires exactly 21 joint tokens")
        if num_refine_layers < 1:
            raise ValueError("num_refine_layers must be positive")
        stage_supervision_weights = tuple(
            float(weight) for weight in stage_supervision_weights
        )
        if len(stage_supervision_weights) < num_refine_layers:
            raise ValueError(
                "stage_supervision_weights must provide at least one weight per "
                f"refinement stage, got {len(stage_supervision_weights)} weights "
                f"for {num_refine_layers} stages"
            )
        if any(weight < 0 for weight in stage_supervision_weights):
            raise ValueError("stage_supervision_weights must be non-negative")
        if sum(stage_supervision_weights[-num_refine_layers:]) <= 0:
            raise ValueError("active stage_supervision_weights must have a positive sum")
        self.upsample_dim = upsample_dim
        self.num_joints = num_joints
        self.num_refine_layers = num_refine_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.initial_2d_loss_weight = float(initial_2d_loss_weight)
        self.joint_2d_loss_weight = float(joint_2d_loss_weight)
        self.joint_3d_loss_weight = float(joint_3d_loss_weight)
        self.stage_supervision_weights = stage_supervision_weights
        self.vertices_loss_weight = float(vertices_loss_weight)
        
        # 1. Backbone
        self.vitmodel = ViTPosePlusBBackbone(
            config_path=vitpose_config_path,
            checkpoint_path=vitpose_checkpoint_path,
            dataset_source=vitpose_dataset_source,
        )
        self.backbone_lr = float(backbone_lr if backbone_lr is not None else lr)
        self.backbone_freeze_epochs = int(backbone_freeze_epochs)
        self.lr_warmup_epochs = int(lr_warmup_epochs)
        if self.lr_warmup_epochs < 0:
            raise ValueError("lr_warmup_epochs must be non-negative")
        self._backbone_is_frozen = False
        if self.backbone_freeze_epochs > 0:
            self._set_backbone_trainable(False)

        actual_feature_dim = self.vitmodel.feature_dim
        if feature_dim is not None and int(feature_dim) != actual_feature_dim:
            raise ValueError(
                f"Configured feature_dim={feature_dim}, but {self.vitmodel.model_name} "
                f"outputs {actual_feature_dim} channels"
            )
        # Project ViTPose++-B's final spatial feature map into decoder memory.
        self.backbone_projection = nn.Conv2d(
            actual_feature_dim, self.upsample_dim, kernel_size=1
        )

        # Preserve a compact 4x3 spatial grid and regress all crop-normalized
        # joint coordinates directly; no per-joint heatmaps are constructed.
        self.initial_2d_head = Initial2DCoordinateHead(
            in_channels=self.upsample_dim,
            num_joints=self.num_joints,
            bottleneck_channels=initial_2d_bottleneck_channels,
            pooled_size=initial_2d_pooled_size,
            hidden_dim=initial_2d_hidden_dim,
            dropout=initial_2d_dropout,
        )
        
        # 3. Positional Embedding for Feature Map (Memory)
        self.pos_embed_layer = PositionEmbeddingSine(num_pos_feats=upsample_dim // 2, normalize=True)
        
        # 4. Learnable Joint Tokens and Queries
        self.joint_tokens = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        self.joint_token_pos = nn.Parameter(torch.randn(1, num_joints, upsample_dim))
        
        # 初始化策略
        nn.init.normal_(self.joint_tokens, std=0.02)
        nn.init.normal_(self.joint_token_pos, std=0.02)

        # 5. Transformer Layers
        self.layers_sa = nn.ModuleList()
        for i in range(self.num_refine_layers):
            if i < 2:
                self.layers_sa.append(
                    HandGraphormerLayer(
                        d_model=upsample_dim,
                        n_head=8, 
                        dim_feedforward=1024,
                        dropout=0.1,
                    )
                )
            else: # 最后一层使用全局 Self-Attention
                self.layers_sa.append(
                    PoseSelfAttentionLayer(
                        d_model=upsample_dim,
                        n_head=8, 
                        dim_feedforward=1024,
                        dropout=0.1,
                    )
                )

        self.layers_ca = nn.ModuleList([
            PoseRefinementLayer(
                d_model=upsample_dim,
                n_head=8, 
                dim_feedforward=1024,
                dropout=0.1,
                attention_mode=refinement_attention_mode(i),
                local_grid_size=local_grid_size,
                local_grid_radius=local_grid_radius,
            )
            for i in range(self.num_refine_layers)
        ])
        

        # 6. Regression Head (Shared or Independent, here shared for simplicity)
        self.pose_3d_head_PR = Pose3DRegressionHead(
            in_channels=upsample_dim, 
            mid_channels=128, 
            out_channels=6, # 3 wrist-relative coordinates + 3 log-variances
            dropout=0.1,
        )

        # The portable SimpleHand mesh head consumes [B, 21, 256].  Production
        # uses 256-D decoder tokens; the projection also keeps small test
        # configurations valid without changing the packaged implementation.
        self.mesh_token_projection = (
            nn.Identity()
            if upsample_dim == 256
            else nn.Linear(upsample_dim, 256)
        )
        self.mesh_regressor = MeshRegressor()
        self.mesh_to_joints = MeshToJoints()

    def _set_backbone_trainable(self, trainable):
        for parameter in self.vitmodel.parameters():
            parameter.requires_grad = trainable
        self._backbone_is_frozen = not trainable

    def on_train_epoch_start(self):
        should_train_backbone = self.current_epoch >= self.backbone_freeze_epochs
        if should_train_backbone and self._backbone_is_frozen:
            self._set_backbone_trainable(True)
        elif not should_train_backbone and not self._backbone_is_frozen:
            self._set_backbone_trainable(False)

        # A frozen backbone must also keep its BatchNorm running statistics
        # fixed. Lightning has already put the whole module in train mode when
        # this hook runs, so override only the backbone here.
        if self._backbone_is_frozen:
            self.vitmodel.eval()
        else:
            self.vitmodel.train()
        
        
    # def _compute_gnll_loss(self, pred_mu, pred_logvar, gt):
    #     """
    #     计算 Gaussian Negative Log Likelihood Loss
    #     """
    #     mse_term = (pred_mu - gt) ** 2
    #     precision = torch.exp(-pred_logvar)
    #     loss = 0.5 * (precision * mse_term + pred_logvar)
    #     return loss.mean()
    @staticmethod
    def _masked_coordinate_mean(loss, valid_mask=None):
        if valid_mask is None:
            return loss.mean()
        valid_mask = valid_mask.to(device=loss.device, dtype=torch.bool)
        while valid_mask.ndim < loss.ndim:
            valid_mask = valid_mask.unsqueeze(-1)
        valid_mask = valid_mask.expand_as(loss)
        valid_count = valid_mask.sum()
        if valid_count.item() == 0:
            return loss.nan_to_num().sum() * 0.0
        return loss.masked_select(valid_mask).sum() / valid_count

    def _compute_gnll_loss(
        self, pred_mu, pred_logvar, gt, valid_mask=None, warmup_epochs=999, beta=10.0
    ):
        """
        组合拳版 GNLL Loss: 延迟不确定性学习 + Smooth L1 鲁棒回归
        """
        # 1. 计算鲁棒的回归距离 (注意 reduction='none' 确保后续能逐元素乘以权重)
        # beta 是 L1 和 L2 的平滑过渡阈值
        robust_dist = F.smooth_l1_loss(pred_mu, gt, reduction='none', beta=beta)
        
        # 2. 阶段控制：利用 Lightning 内置的 self.current_epoch
        if self.current_epoch < warmup_epochs:
            # 【先学走】退化为纯 Smooth L1，完全不给 pred_logvar 传导梯度
            # 让模型专心致志地拟合 3D 坐标
            return self._masked_coordinate_mean(robust_dist, valid_mask)
        else:
            # 【再学跑】放开 logvar 权重，进行动态不确定性学习
            # 引入 clamp 防止 logvar 极端暴走 (可选，但推荐作为最后一道防线)
            pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)
            
            precision = torch.exp(-pred_logvar)
            loss = precision * robust_dist + 0.5 * pred_logvar
            
            return self._masked_coordinate_mean(loss, valid_mask)
    def forward(self, x, hand_back=None, cam_k=None, root_3d=None):
        results = {}
        B = x.shape[0]
        image_size = x.shape[-2:]

        needs_projected_3d_reference = any(
            index > 0 and layer.attention_mode == "local"
            for index, layer in enumerate(self.layers_ca)
        )
        if needs_projected_3d_reference:
            if cam_k is None or root_3d is None:
                raise ValueError(
                    "Local cross-attention after Stage 0 requires cam_k [B,3,3] "
                    "and root_3d [B,3] (camera-coordinate wrist)."
                )
            cam_k = cam_k.to(device=x.device, dtype=x.dtype)
            root_3d = root_3d.to(device=x.device, dtype=x.dtype)
            if root_3d.ndim == 2:
                root_3d = root_3d.unsqueeze(1)
            if root_3d.shape != (B, 1, 3):
                raise ValueError(f"root_3d must be [B,3] or [B,1,3], got {tuple(root_3d.shape)}")
        
        feature_map = self.vitmodel(x)
        if feature_map.ndim != 4:
            raise ValueError(
                f"ViTPose++-B backbone must return [B, C, H, W], got "
                f"{tuple(feature_map.shape)}"
            )
        global_feature_map = self.backbone_projection(feature_map)

        initial_pose_2d_normalized = self.initial_2d_head(global_feature_map)
        initial_pose_2d = normalized_crop_to_pixel(
            initial_pose_2d_normalized, image_size
        )

        # Memory Pos Embed
        pos_embed_map = self.pos_embed_layer(global_feature_map)
        
        # Query Tokens (Learnable) - 扩展到 Batch 维度
        curr_tokens = self.joint_tokens.expand(B, -1, -1)   # [B, 21, 256]
        query_pos   = self.joint_token_pos.expand(B, -1, -1) # [B,hand_back=None 21, 256]

        # 5. Transformer Loop: Self-Attn -> Cross-Attn -> Predict
        all_stage_preds = []
        all_stage_logvars = []
        all_stage_vertices = []
        all_stage_mesh_joints = []
        all_stage_gates = [] # 用于记录 gate 值，分析手背利用率
        stage_reference_points = []
        current_pose = None
        
        for i in range(self.num_refine_layers):
            layer_ca = self.layers_ca[i]
            reference_points = None
            if layer_ca.attention_mode == "local":
                if i == 0:
                    # Stage 0 consumes the new direct 2D estimate.  Local CA
                    # always receives crop pixels, never feature-map indices.
                    reference_points = initial_pose_2d
                else:
                    camera_pose = current_pose + root_3d
                    reference_points = project_camera_joints_to_image(
                        camera_pose, cam_k, image_size
                    )
            stage_reference_points.append(reference_points)

            if self.training and self.use_gradient_checkpointing:
                curr_tokens = checkpoint(
                    self.layers_sa[i],
                    curr_tokens,   
                    query_pos,     
                    use_reentrant=False
                )

                curr_tokens = checkpoint(
                    layer_ca,
                    curr_tokens,         
                    global_feature_map,  
                    query_pos,           
                    pos_embed_map,       
                    reference_points,
                    image_size,
                    use_reentrant=False
                )
            else:
                curr_tokens = self.layers_sa[i](x=curr_tokens, pos=query_pos)
                curr_tokens = layer_ca(
                    tgt=curr_tokens,
                    memory=global_feature_map,
                    query_pos=query_pos,
                    memory_pos=pos_embed_map,
                    reference_points=reference_points,
                    image_size=image_size,
                )

            mesh_tokens = self.mesh_token_projection(curr_tokens)
            if self.training and self.use_gradient_checkpointing:
                raw_vertices = checkpoint(
                    self.mesh_regressor,
                    mesh_tokens,
                    use_reentrant=False,
                )
            else:
                raw_vertices = self.mesh_regressor(mesh_tokens)
            raw_mesh_joints = self.mesh_to_joints(raw_vertices)
            mesh_wrist = raw_mesh_joints[:, 0:1, :]
            stage_vertices = raw_vertices - mesh_wrist
            stage_mesh_joints = raw_mesh_joints - mesh_wrist
            all_stage_vertices.append(stage_vertices)
            all_stage_mesh_joints.append(stage_mesh_joints)

            raw_pred = self.pose_3d_head_PR(curr_tokens)
            stage_pred_mu = raw_pred[..., :3]
            # FreiHAND uses joint 0 as wrist; the model predicts a full root-relative set.
            stage_pred_mu = stage_pred_mu - stage_pred_mu[:, 0:1, :]
            stage_pred_logvar = raw_pred[..., 3:]
            current_pose = stage_pred_mu

            all_stage_preds.append(stage_pred_mu)
            all_stage_logvars.append(stage_pred_logvar)
                

        # 取最后一个 Stage 作为最终结果
        reference_pose_3d = all_stage_preds[-1]
        pred_logvar_3d = all_stage_logvars[-1]
        pred_vertices = all_stage_vertices[-1]
        pred_mesh_joints = all_stage_mesh_joints[-1]
        
        
        # Public predictions come from the supervised mesh branch.  The legacy
        # token-coordinate branch is retained only to generate later-stage
        # local-attention reference points.
        results["pose3d"] = pred_mesh_joints
        results["pred_mesh_joints"] = pred_mesh_joints
        results["pred_vertices"] = pred_vertices
        results["reference_pose3d"] = reference_pose_3d
        results["pose3d_logvar"] = pred_logvar_3d
        results["all_stage_pose3d"] = all_stage_preds 
        results["all_stage_logvars"] = all_stage_logvars
        results["all_stage_vertices"] = all_stage_vertices
        results["all_stage_mesh_joints"] = all_stage_mesh_joints
        results["initial_pose2d_normalized"] = initial_pose_2d_normalized
        results["initial_pose2d"] = initial_pose_2d
        results["stage_reference_points"] = stage_reference_points
        
        return results

    def _compute_mpjpe_3d(self, pred, gt, valid_mask=None):
        per_joint_error = torch.linalg.norm(pred - gt, dim=-1)
        finite = torch.isfinite(per_joint_error)
        if valid_mask is not None:
            finite = finite & valid_mask.to(device=finite.device, dtype=torch.bool)
        if not torch.any(finite):
            safe_error = torch.where(
                torch.isfinite(per_joint_error),
                per_joint_error,
                torch.zeros_like(per_joint_error),
            )
            return safe_error.sum() * 0.0
        return per_joint_error[finite].mean()

    def _compute_pa_mpjpe_3d(self, pred, gt):
        """
        计算 Procrustes 对齐后的 MPJPE (PA-MPJPE)
        pred: [B, N, 3]
        gt: [B, N, 3]
        """
        with torch.no_grad():
            # 1. 均值中心化 (Translation Alignment)
            mu_pred = pred.mean(dim=1, keepdim=True)
            mu_gt = gt.mean(dim=1, keepdim=True)
            X = pred - mu_pred
            Y = gt - mu_gt
            
            # 2. 尺度归一化 (Scale Alignment)
            norm_X = torch.norm(X, dim=(1, 2), keepdim=True)
            norm_Y = torch.norm(Y, dim=(1, 2), keepdim=True)
            
            X_norm = X / (norm_X + 1e-8)
            Y_norm = Y / (norm_Y + 1e-8)
            
            # 3. 计算最佳旋转矩阵 (Rotation Alignment via SVD)
            H = torch.bmm(X_norm.transpose(1, 2), Y_norm)
            U, S, Vh = torch.linalg.svd(H)
            V = Vh.transpose(1, 2)
            R = torch.bmm(V, U.transpose(1, 2))
            
            # 防止反射变换 (Reflection)
            det = torch.linalg.det(R)
            V_valid = V.clone()
            flip_idx = det < 0
            if flip_idx.any():
                V_valid[flip_idx, :, 2] *= -1
                R = torch.bmm(V_valid, U.transpose(1, 2))
            
            # 4. 空间对齐: 旋转 -> 恢复 GT 尺度 -> 加上 GT 平移
            pred_aligned = torch.bmm(X_norm, R.transpose(1, 2)) * norm_Y + mu_gt
            
            # 5. 计算对齐后的欧氏距离
            dist = torch.norm(pred_aligned - gt, dim=-1)
            return dist.mean()
    def _compute_root_rigid_mpjpe_3d(self, pred, gt):
        """
        计算根节点对齐 + 旋转对齐（无缩放）后的 MPJPE
        pred: [B, N, 3]
        gt: [B, N, 3]
        """
        with torch.no_grad():
            # 1. 根节点平移对齐 (假设索引 0 为根节点 Wrist)
            root_pred = pred[:, 0:1, :]
            root_gt = gt[:, 0:1, :]
            
            X = pred - root_pred
            Y = gt - root_gt
            
            # 2. 计算最佳旋转矩阵 (Rotation Alignment via SVD)
            H = torch.bmm(X.transpose(1, 2), Y)
            U, S, Vh = torch.linalg.svd(H)
            V = Vh.transpose(1, 2)
            R = torch.bmm(V, U.transpose(1, 2))
            
            # 防止反射变换 (Reflection)
            det = torch.linalg.det(R)
            V_valid = V.clone()
            flip_idx = det < 0
            if flip_idx.any():
                V_valid[flip_idx, :, 2] *= -1
                R = torch.bmm(V_valid, U.transpose(1, 2))
            
            # 3. 空间对齐: 仅应用旋转，然后加上 GT 的根节点位置
            pred_aligned = torch.bmm(X, R.transpose(1, 2)) + root_gt
            
            # 4. 计算对齐后的欧氏距离
            dist = torch.norm(pred_aligned - gt, dim=-1)
            return dist.mean()
    
    def training_step(self, batch, batch_idx):
        imgs = batch['img'] 
        gt_pose = batch['gt_pose'] 
        gt_pose_2d = batch['gt_pose_2d'] # 来自 dataset，已经包含了畸变
        cam_k = batch['cam_k']
        dist_coeffs = batch['dist_coeffs'] 
        hand_back = batch["hand_back"]
        root_3d = batch.get("origin_3d", gt_pose[:, 0, :])
        results = self(imgs, hand_back, cam_k=cam_k, root_3d=root_3d)
        pred_pose_3d = results['pose3d']
        
        # 计算 Loss (MPJPE + Deep Supervision)
        all_preds = results['all_stage_pose3d']
        loss_3d_pose = 0
        for pred_mu in all_preds:
            loss_3d_pose += self._compute_mpjpe_3d(pred_mu, gt_pose)
        
        
        self.log('train_loss', loss_3d_pose, prog_bar=True)
        
        with torch.no_grad():
            mpjpe_3d = self._compute_mpjpe_3d(results['pose3d'], gt_pose)
            pa_mpjpe_3d = self._compute_pa_mpjpe_3d(results['pose3d'], gt_pose)
            
            self.log('train_mpjpe_3d', mpjpe_3d, prog_bar=True)
            self.log('train_pa_mpjpe_3d', pa_mpjpe_3d, prog_bar=True)

        return loss_3d_pose

    def validation_step(self, batch, batch_idx):
        imgs = batch['img']
        gt_pose = batch['gt_pose']
        hand_back = batch["hand_back"]
        cam_k = batch['cam_k']
        root_3d = batch.get("origin_3d", gt_pose[:, 0, :])
        results = self(imgs, hand_back, cam_k=cam_k, root_3d=root_3d)
        # self._save_batch_visuals(batch, results, batch_idx, mode='val')
        pred_mu = results['pose3d']
        val_loss = self._compute_mpjpe_3d(pred_mu, gt_pose)
        val_mpjpe_3d = self._compute_mpjpe_3d(pred_mu, gt_pose)
        val_pa_mpjpe_3d = self._compute_pa_mpjpe_3d(pred_mu, gt_pose)
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mpjpe_3d', val_mpjpe_3d, prog_bar=True)
        self.log('val_pa_mpjpe_3d', val_pa_mpjpe_3d, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        imgs = batch['img']
        gt_pose = batch['gt_pose']
        hand_back = batch["hand_back"]
        cam_k = batch['cam_k']
        root_3d = batch.get("origin_3d", gt_pose[:, 0, :])
        results = self(imgs, hand_back, cam_k=cam_k, root_3d=root_3d)
        # self._save_batch_visuals(batch, results, batch_idx, mode='test')
        pred_mu = results['pose3d']
        test_loss = self._compute_mpjpe_3d(pred_mu, gt_pose)
        test_mpjpe_3d = self._compute_mpjpe_3d(pred_mu, gt_pose)
        test_pa_mpjpe_3d = self._compute_pa_mpjpe_3d(pred_mu, gt_pose)
        
        # 新增：计算根节点+旋转对齐（无缩放）的 MPJPE
        test_root_rigid_mpjpe_3d = self._compute_root_rigid_mpjpe_3d(pred_mu, gt_pose)
        
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_mpjpe_3d', test_mpjpe_3d, prog_bar=True)
        self.log('test_pa_mpjpe_3d', test_pa_mpjpe_3d, prog_bar=True)
        
        # 新增：记录新指标
        # self.log('test_root_rigid_mpjpe_3d', test_root_rigid_mpjpe_3d, prog_bar=True)
    def _save_batch_visuals(self, batch, results, batch_idx, mode='val'):
        """
        保存前10个batch的数据用于可视化分析
        """
        if batch_idx >= 10:
            return

        # 创建保存目录
        save_dir = "vis_result"
        os.makedirs(save_dir, exist_ok=True)

        # 创建字典
        # 注意：将 Tensor 转为 numpy 
        data_to_save = {
            "img": batch['img'].detach().cpu().numpy(),              # [B, 3, H, W]
            "gt_pose_3d": batch['gt_pose'].detach().cpu().numpy(),   # [B, 21, 3]
            "gt_pose_2d": batch['gt_pose_2d'].detach().cpu().numpy(),# [B, 21, 2]
            
            # "pred_pose_2d": results['pose2d'].detach().cpu().numpy(), # [B, 21, 2]
        }
        
        # 如果有 3D 预测结果，保存
        if 'pose3d' in results:
            data_to_save["pred_pose_3d"] = results['pose3d'].detach().cpu().numpy() # [B, 21, 3]

        # 生成文件名: vis_result/val_epoch_0_batch_1.pkl
        filename = f"{mode}_epoch_{self.current_epoch}_batch_{batch_idx}.pkl"
        save_path = os.path.join(save_dir, filename)

        # 保存为 pickle 文件
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
    
    def configure_optimizers(self):
        # Keep frozen ViTPose parameters in the optimizer so they can be
        # unfrozen without rebuilding optimizer state after the warm-up phase.
        backbone_params = list(self.vitmodel.parameters())
        backbone_param_ids = {id(parameter) for parameter in backbone_params}
        task_params = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in backbone_param_ids
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": task_params, "lr": self.hparams.lr, "name": "pose_head"},
                {"params": backbone_params, "lr": self.backbone_lr, "name": "vitpose"},
            ],
            weight_decay=0.04
        )
        
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 100
        warmup_epochs = self.lr_warmup_epochs
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=1e-6
        )
        if warmup_epochs > 0:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.001,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_cosine],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = scheduler_cosine
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss"
            }
        }
