import pdb
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 标准手部骨架连接
HAND_SKELETON = [
    [0, 1, 2, 3, 4],        # Thumb
    [0, 5, 6, 7, 8],        # Index
    [0, 9, 10, 11, 12],     # Middle
    [0, 13, 14, 15, 16],    # Ring
    [0, 17, 18, 19, 20]     # Pinky
]

def visualize_3d_comparison(gt_3d, pred_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def plot_skeleton_3d(pose, color, label_prefix):
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, label=f'{label_prefix} Joints', s=20)
        first_line = True
        for chain in HAND_SKELETON:
            chain_pts = pose[chain]
            label = f'{label_prefix} Skeleton' if first_line else None
            ax.plot(chain_pts[:, 0], chain_pts[:, 1], chain_pts[:, 2], c=color, linewidth=2, label=label)
            first_line = False

    plot_skeleton_3d(gt_3d, color='green', label_prefix='GT')
    plot_skeleton_3d(pred_3d, color='red', label_prefix='Pred')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Comparison: GT (Green) vs Pred (Red)')
    ax.legend()
    plt.show()

def visualize_2d_projection(img, gt_2d, pred_2d):
    img_display = img.transpose(1, 2, 0)
    img_min = img_display.min()
    img_max = img_display.max()
    if img_max > img_min:
        img_display = (img_display - img_min) / (img_max - img_min)
    else:
        img_display = np.clip(img_display, 0, 1)  # 避免除零

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_display)

    def plot_skeleton_2d(pose, color, label_prefix, linestyle='-'):
        ax.scatter(pose[:, 0], pose[:, 1], c=color, s=15, zorder=10)
        first_line = True
        for chain in HAND_SKELETON:
            chain_pts = pose[chain]
            label = f'{label_prefix}' if first_line else None
            ax.plot(chain_pts[:, 0], chain_pts[:, 1], c=color, linewidth=2, linestyle=linestyle, label=label)
            first_line = False

    plot_skeleton_2d(gt_2d, color='lime', label_prefix='GT', linestyle='-')
    plot_skeleton_2d(pred_2d, color='red', label_prefix='Pred', linestyle='--')

    ax.set_title('2D Projection: GT (Green) vs Pred (Red)')
    ax.legend()
    plt.axis('off')
    plt.show()

def main(epoch, batch, sample_idx, base_dir="vis_result"):
    # 构造文件路径
    filename = f"test_epoch_{epoch}_batch_{batch}.pkl"
    data_path = os.path.join(base_dir, filename)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print("Available keys:", data.keys())

    # 提取指定样本
    img_sample = data['img'][sample_idx]         # [3, H, W]
    import numpy as np
    from PIL import Image

    arr = img_sample
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255)
    img_pil = Image.fromarray(arr.transpose(1, 2, 0).astype(np.uint8))
    img_pil.save("output_image.png")
    
    gt_2d = data['gt_pose_2d'][sample_idx]       # [21, 2]
    # pred_2d = data['pred_pose_2d'][sample_idx]   # [21, 2]
    gt_3d = data['gt_pose_3d'][sample_idx]       # [21, 3]
    pred_3d = data['pred_pose_3d'][sample_idx]   # [21, 3]

    print(f"Visualizing epoch={epoch}, batch={batch}, sample={sample_idx}...")

    visualize_3d_comparison(gt_3d, pred_3d)
    visualize_2d_projection(img_sample, gt_2d, gt_2d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize hand pose estimation results.")
    parser.add_argument('--epoch', type=int, default=0, help='Epoch index (e.g., 0)')
    parser.add_argument('--batch', type=int, default=0, help='Batch index (e.g., 0)')
    parser.add_argument('--sample', type=int, default=0, help='Sample index within the batch (e.g., 0)')
    parser.add_argument('--dir', type=str, default='vis_result', help='Directory containing .pkl files')

    args = parser.parse_args()

    main(epoch=args.epoch, batch=args.batch, sample_idx=args.sample, base_dir=args.dir)