import torch

def generate_target_heatmaps(joints_2d, input_size, heatmap_size, sigma=2):
    """
    根据 GT 坐标生成高斯热图
    joints_2d: [Batch, Num_Joints, 2] (x, y) 原始图像坐标
    input_size: (H, W) 原始输入图像大小 (例如 224, 224)
    heatmap_size: (h, w) 特征图大小 (例如 64, 64)
    """
    B, K, _ = joints_2d.shape
    H, W = heatmap_size
    
    # 计算缩放比例 (例如 224 -> 64，stride=3.5)
    stride_x = input_size[1] / W
    stride_y = input_size[0] / H
    
    target_heatmaps = torch.zeros((B, K, H, W), device=joints_2d.device)
    
    # 生成网格
    x = torch.arange(0, W, 1, dtype=torch.float32, device=joints_2d.device)
    y = torch.arange(0, H, 1, dtype=torch.float32, device=joints_2d.device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    for b in range(B):
        for k in range(K):
            # 将 GT 坐标缩放到 Heatmap 尺度
            mu_x = joints_2d[b, k, 0] / stride_x
            mu_y = joints_2d[b, k, 1] / stride_y
            
            # 检查关键点是否在可视范围内 (假设 GT 有 visibility 标志更好，这里简单判断)
            # 如果坐标全是 0 或者出界，通常保持全 0 热图
            if mu_x < 0 or mu_x >= W or mu_y < 0 or mu_y >= H:
                continue
                
            # 生成高斯
            # 这里的 Gaussian 公式没有归一化系数，最高点为 1，这是 Heatmap 回归的惯例
            target_heatmaps[b, k] = torch.exp(
                -((x_grid - mu_x)**2 + (y_grid - mu_y)**2) / (2 * sigma**2)
            )
            
    return target_heatmaps