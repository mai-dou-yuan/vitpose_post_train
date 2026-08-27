# 建议放在 pl_system.py 的顶部，import 之后

import torch

def project_3d_to_2d_batch_with_distortion(points_3d, cam_k, dist_coeffs):
    """
    可微分的 3D 到 2D 投影，包含径向和切向畸变。
    对应 OpenCV 的畸变模型。
    
    Args:
        points_3d:   [B, N, 3]  (x, y, z)
        cam_k:       [B, 3, 3]  (fx, fy, cx, cy)
        dist_coeffs: [B, 5]     (k1, k2, p1, p2, k3) 
                     注意：如果数据里是 8 个参数，这里只取前 5 个即可，通常后 3 个是高阶项或薄棱镜畸变，影响较小。
    
    Returns:
        points_2d:   [B, N, 2]  (u, v)
    """
    # 1. 提取坐标
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    
    # 避免除零错误，给 Z 一个极小值保护
    z = z.clamp(min=1e-5)
    
    # 2. 归一化平面坐标 (Normalized Image Coordinates)
    x_norm = x / z
    y_norm = y / z
    
    # 3. 计算半径 r^2
    r2 = x_norm**2 + y_norm**2
    r4 = r2 * r2
    r6 = r2 * r4
    
    # 4. 准备畸变系数
    # dist_coeffs shape: [B, 5] -> [B, 1] 用于广播到 [B, N]
    # 假设 dist_coeffs 顺序为 (k1, k2, p1, p2, k3)
    k1 = dist_coeffs[:, 0:1]
    k2 = dist_coeffs[:, 1:2]
    p1 = dist_coeffs[:, 2:3]
    p2 = dist_coeffs[:, 3:4]
    k3 = dist_coeffs[:, 4:5]

    # 5. 径向畸变 (Radial Distortion)
    # radial = (1 + k1*r^2 + k2*r^4 + k3*r^6)
    radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    
    # 6. 切向畸变 (Tangential Distortion)
    # x_tan = 2*p1*x*y + p2*(r^2 + 2*x^2)
    # y_tan = p1*(r^2 + 2*y^2) + 2*p2*x*y
    tan_distortion_x = 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm**2)
    tan_distortion_y = p1 * (r2 + 2.0 * y_norm**2) + 2.0 * p2 * x_norm * y_norm
    
    # 7. 应用畸变到归一化坐标
    x_distorted = x_norm * radial_distortion + tan_distortion_x
    y_distorted = y_norm * radial_distortion + tan_distortion_y
    
    # 8. 投影到像素坐标 (Pixel Coordinates)
    # u = fx * x_distorted + cx
    # v = fy * y_distorted + cy
    fx = cam_k[:, 0, 0:1] # [B, 1]
    fy = cam_k[:, 1, 1:1]
    cx = cam_k[:, 0, 2:3]
    cy = cam_k[:, 1, 2:3]
    
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy
    
    # 组合结果 [B, N, 2]
    return torch.stack([u, v], dim=-1)