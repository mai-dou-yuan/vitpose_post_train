import os
import torch
import cv2
import pickle
import numpy as np

# 导入你的 Lightning 模型
from pl_system_v6_graphormer import PoseLightningModule

def infer_and_save_specific_images(
    ckpt_path, 
    data_root, 
    target_indices, 
    save_dir="inference_results", 
    img_size=518
):
    """
    对指定的图片进行推理，并保存预测的 3D 坐标、原始图片和相机内参。
    
    参数:
        ckpt_path: 你的模型 checkpoint 路径
        data_root: 数据集根目录 (包含 .jpg 和 .pkl 文件的路径)
        target_indices: 你想要测试的图片索引列表，例如 [0, 15, 123]
        save_dir: 结果保存的目录
        img_size: 模型输入的分辨率，默认 518
    """
    # 1. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 2. 确定设备并加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载模型权重: {ckpt_path}")
    # strict=False 可以在某些不匹配的情况下强制加载，如果确定结构完全一致可以去掉
    model = PoseLightningModule.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.to(device)
    print("模型加载完成！\n")

    # 3. 遍历指定的索引进行推理
    for idx in target_indices:
        # 格式化索引为 6 位数字字符串
        idx_str = f"{idx:06d}"
        
        left_path = os.path.join(data_root, f"{idx_str}_origin_left.jpg")
        right_path = os.path.join(data_root, f"{idx_str}_origin_right.jpg")
        pkl_path = os.path.join(data_root, f"{idx_str}.pkl")

        # 检查文件是否存在
        if not os.path.exists(left_path) or not os.path.exists(pkl_path) or not os.path.exists(right_path):
            print(f"[警告] 找不到索引 {idx_str} 对应的文件，跳过...")
            continue

        print(f"正在处理: {idx_str} ...")

        # ================= 1. 读取原始数据 =================
        # 读取原始图片 (BGR 格式)
        img_left_orig = cv2.imread(left_path)
        img_right_orig = cv2.imread(right_path)

        # 转换为 RGB 格式用于模型输入
        img_left_rgb = cv2.cvtColor(img_left_orig, cv2.COLOR_BGR2RGB)
        img_right_rgb = cv2.cvtColor(img_right_orig, cv2.COLOR_BGR2RGB)
        
        old_h, old_w = img_left_rgb.shape[:2]

        # 读取 PKL 获取相机的原始 cam_k
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

        # 提取当前视角的 camera ID (和 dataset.py 保持一致)
        if 2 in pkl_data["cam_info"]:
            current_cam_id = 2
        elif 6 in pkl_data["cam_info"]:
            current_cam_id = 6
        elif 4 in pkl_data["cam_info"]:
            current_cam_id = 4
        elif 1 in pkl_data["cam_info"]:
            current_cam_id = 1
        
        cam_k_original = pkl_data["cam_info"][current_cam_id]['cam_k'].copy()

        # ================= 2. 数据预处理 =================
        # Resize 到模型需要的尺寸
        img_left_resized = cv2.resize(img_left_rgb, (img_size, img_size))
        img_right_resized = cv2.resize(img_right_rgb, (img_size, img_size))
        
        # 调整 cam_k (根据 Resize 的比例)
        scale_x = img_size / old_w
        scale_y = img_size / old_h
        cam_k_new = cam_k_original.copy()
        cam_k_new[0, 0] *= scale_x 
        cam_k_new[0, 2] *= scale_x 
        cam_k_new[1, 1] *= scale_y 
        cam_k_new[1, 2] *= scale_y 

        # 转换为 Tensor: [H, W, C] -> [C, H, W] -> 归一化 -> 增加 Batch 维度 [1, C, H, W]
        img_tensor = torch.from_numpy(img_left_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0).to(device)
        hand_back_tensor = torch.from_numpy(img_right_resized.transpose((2, 0, 1)).astype(np.float32) / 255.0).unsqueeze(0).to(device)

        # ================= 3. 模型推理 =================
        with torch.no_grad():
            results = model(img_tensor, hand_back_tensor)
            # 提取 3D 预测结果并转换为 numpy array，去掉 batch 维度
            pred_pose_3d = results['pose3d'].squeeze(0).cpu().numpy()  # shape: [21, 3]

        # ================= 4. 保存结果 =================
        # 保存原始左图
        save_img_path = os.path.join(save_dir, f"{idx_str}_origin_left.jpg")
        cv2.imwrite(save_img_path, img_left_orig)

        # 构建需要保存的数据字典
        save_data = {
            "pred_pose_3d": pred_pose_3d,        # 模型预测的 3D 坐标
            "cam_k_original": cam_k_original,    # 原始的相机内参
            "cam_k_resized": cam_k_new,          # 经过 Resize 调整后的相机内参
            "gt_pose_3d": pkl_data["frames_info"][:, :] # 如果需要对比，顺便把 GT 也存下来
        }

        # 保存为 .pkl 文件
        save_pkl_path = os.path.join(save_dir, f"{idx_str}_result.pkl")
        with open(save_pkl_path, "wb") as f:
            pickle.dump(save_data, f)
            
    print(f"\n所有指定图片处理完毕！结果已保存至: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    # ================= 配置参数 =================
    
    # 1. 指定你的 Checkpoint 路径
    CKPT_PATH = 'checkpoints/pose-epoch=52-val_mpjpe_3d=14.8132.ckpt' 
    
    # 2. 指定数据集根目录 (存放 000000.pkl, 000000_origin_left.jpg 的目录)
    DATA_ROOT = '/home/duanmu/data/vitpose_v3/vitpose_v3/save_path_15'  # 例如: '/data/unrealego/'
    
    # 3. 指定你想单独提取的图片索引列表
    # 比如你想处理 000010.jpg, 000150.jpg, 001024.jpg，就填入 [10, 150, 1024]
    TARGET_INDICES = [2856, 7000] 
    
    # 4. 指定结果保存的文件夹
    SAVE_DIR = "specific_images_results"
    
    # ============================================

    infer_and_save_specific_images(
        ckpt_path=CKPT_PATH,
        data_root=DATA_ROOT,
        target_indices=TARGET_INDICES,
        save_dir=SAVE_DIR
    )