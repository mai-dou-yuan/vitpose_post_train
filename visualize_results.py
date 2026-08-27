import os
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def save_inference_visualizations(result_dir, target_indices, output_dir="visualizations"):
    """
    读取 pkl 结果并保存可视化图像到指定目录，不进行弹窗显示。
    """
    # 创建可视化结果保存目录
    os.makedirs(output_dir, exist_ok=True)

    # 定义手部骨架连接顺序
    skeleton = [
        [0, 1], [1, 2], [2, 3], [3, 4],       # thumb
        [0, 5], [5, 6], [6, 7], [7, 8],       # index
        [0, 9], [9, 10], [10, 11], [11, 12],  # middle
        [0, 13], [13, 14], [14, 15], [15, 16],# ring
        [0, 17], [17, 18], [18, 19], [19, 20] # pinky
    ]

    for idx in target_indices:
        idx_str = f"{idx:06d}"
        pkl_path = os.path.join(result_dir, f"{idx_str}_result.pkl")
        img_path = os.path.join(result_dir, f"{idx_str}_origin_left.jpg")

        if not os.path.exists(pkl_path) or not os.path.exists(img_path):
            print(f"[跳过] {idx_str}: 找不到结果文件")
            continue

        # 1. 加载数据
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_pose_3d = data["pred_pose_3d"]
        cam_k = data["cam_k_original"]

        # 2. 3D 投影到 2D
        # 投影公式：[u, v, 1]^T = K * [x, y, z]^T / z
        pred_pose_2d = np.dot(cam_k, pred_pose_3d.T).T
        pred_pose_2d[:, 0] /= (pred_pose_2d[:, 2] + 1e-6) # 防止除零
        pred_pose_2d[:, 1] /= (pred_pose_2d[:, 2] + 1e-6)
        points_2d = pred_pose_2d[:, :2]

        # 3. 绘图 (使用 Figure 对象)
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f"Result Visualization: {idx_str}", fontsize=14)

        # --- 左子图: 2D 投影 ---
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img_rgb)
        for connection in skeleton:
            p1, p2 = points_2d[connection[0]], points_2d[connection[1]]
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lime', linewidth=2)
        ax1.scatter(points_2d[:, 0], points_2d[:, 1], color='red', s=15)
        ax1.axis('off') # 隐藏坐标轴让图更干净

        # --- 右子图: 3D 骨架 ---
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        for connection in skeleton:
            p1, p2 = pred_pose_3d[connection[0]], pred_pose_3d[connection[1]]
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='blue', linewidth=2)
        ax2.scatter(pred_pose_3d[:, 0], pred_pose_3d[:, 1], pred_pose_3d[:, 2], color='red', s=20)
        
        # 统一 3D 坐标轴比例
        max_range = np.array([pred_pose_3d[:,0].max()-pred_pose_3d[:,0].min(),
                             pred_pose_3d[:,1].max()-pred_pose_3d[:,1].min(),
                             pred_pose_3d[:,2].max()-pred_pose_3d[:,2].min()]).max() / 2.0
        mid = pred_pose_3d.mean(axis=0)
        ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax2.invert_yaxis() # 垂直翻转以匹配图像坐标系

        # 4. 保存并关闭
        save_path = os.path.join(output_dir, f"{idx_str}_vis.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150) # 设置 dpi 提高清晰度
        plt.close(fig) # 释放内存占用
        
        print(f"[成功] 已保存可视化结果至: {save_path}")

if __name__ == "__main__":
    # 配置参数
    RESULT_DIR = "specific_images_results" # 之前推理保存 pkl 的路径
    TARGET_INDICES = [2856, 7000]          # 索引
    OUTPUT_VIS_DIR = "vis_outputs"         # 可视化图片保存路径

    save_inference_visualizations(RESULT_DIR, TARGET_INDICES, OUTPUT_VIS_DIR)