import torch
import torch.nn.functional as F

def sample_features_at_points(features, points, img_size=(224, 224)):
    """
    根据像素坐标采样对应的图片特征。
    
    参数:
        features: 特征图, shape [Batch, Channels, H_feat, W_feat] 
        points: 2D像素坐标, shape [Batch, Num_Points, 2] (x, y)
        img_size: 原始图片尺寸 (H, W), 默认为 (224, 224)
    
    返回:
        joint_tokens: 采样后的特征, shape [Batch, Num_Points, Channels]
    """
    H_img, W_img = img_size
    
    # 获取 x, y 坐标
    x = points[..., 0]
    y = points[..., 1]
    
    # 1. 归一化坐标 (Normalize coordinates)
    # 使用 align_corners=True 的配套公式
    # 范围映射: 0 -> -1, size-1 -> 1
    norm_x = 2.0 * (x / (W_img - 1)) - 1.0
    norm_y = 2.0 * (y / (H_img - 1)) - 1.0
    
    # 拼接坐标, shape: [Batch, Num_Points, 2]
    grid = torch.stack((norm_x, norm_y), dim=-1)
    
    # 2. 调整 Grid 维度以适配 grid_sample
    # grid_sample 要求 grid 为 [B, H_out, W_out, 2]
    # 我们将其视为 [B, 1, Num_Points, 2]
    grid = grid.unsqueeze(1) 
    
    # 3. 双线性插值采样
    # 修正点: 必须设置 align_corners=True 以匹配上面的归一化公式
    sampled_features = F.grid_sample(
        features, 
        grid, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True 
    )
    
    # 4. 调整输出维度
    # [B, C, 1, Num_Points] -> [B, C, Num_Points] -> [B, Num_Points, C]
    joint_tokens = sampled_features.squeeze(2).transpose(1, 2)
    
    return joint_tokens

def verify_alignment():
    print("--- 开始严格对齐测试 ---")
    
    # 创建一个 1x1x3x3 的特征图
    # 值为:
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    features = torch.tensor([[[
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]]]) # shape [1, 1, 3, 3]
    
    # 定义图像尺寸为 3x3 (方便对应)
    img_size = (3, 3)
    
    # 测试点：采样左上角 (0,0) 和 右下角 (2,2)
    # 在 align_corners=True 下，(0,0) 应该得到 1.0，(2,2) 应该得到 9.0
    points = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]) # shape [1, 2, 2]
    
    # 运行函数
    output = sample_features_at_points(features, points, img_size=img_size)
    
    print(f"输入特征图:\n{features.squeeze()}")
    print(f"采样点: (0,0) 和 (2,2)")
    print(f"采样结果: {output.squeeze().tolist()}")
    
    # 验证
    val_top_left = output[0, 0, 0].item()
    val_bottom_right = output[0, 1, 0].item()
    
    # 允许微小误差
    if abs(val_top_left - 1.0) < 1e-5 and abs(val_bottom_right - 9.0) < 1e-5:
        print("✅ 验证通过：端点严格对齐 (align_corners=True 生效)")
    else:
        print("❌ 验证失败：端点未对齐！")
        print(f"   期望: 1.0 和 9.0")
        print(f"   实际: {val_top_left} 和 {val_bottom_right}")
        if abs(val_top_left - 1.0) > 0.1:
            print("   (提示: 这种偏差通常是因为 align_corners=False 导致的)")

if __name__ == "__main__":
    verify_alignment()