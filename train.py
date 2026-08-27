import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml

# 导入类
from datasets.dataset import Unrealego3DPoseDataset
# from pl_system_v6_mutil_simple_vit import PoseLightningModule
# from pl_system_v6_graphormer_wopan import PoseLightningModule
from pl_system_v6_graphormer import PoseLightningModule
# from pl_system import PoseLightningModule

def load_config(config_path="configs/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # ================= 加载配置 =================
    config = load_config()

    # 从配置文件中读取参数
    DATA_ROOT = config['data']['root']
    TRAIN_DIR = os.path.join(DATA_ROOT, config['data']['train_dir'])
    VAL_DIR = os.path.join(DATA_ROOT, config['data']['val_dir'])
    TEST_DIR = os.path.join(DATA_ROOT, config['data']['test_dir'])
    
    
    BATCH_SIZE = config['training']['batch_size']
    MAX_EPOCHS = config['training']['max_epochs']
    NUM_WORKERS = config['training']['num_workers']
    LEARNING_RATE = config['training']['learning_rate']
    
    local_model_dir = config['model']['local_model_dir']
    fastvit_ckpt_path = config['model'].get('fastvit_ckpt_path', None)
    image_size = config['data']['image_size']
    ckpt_path = config['model']['ckpt_path'] # 如果 config 中有定义断点路径

    # pl.seed_everything(config['seed'], workers=True)
    pl.seed_everything(config['seed'])

    # ================= 准备数据 =================
    # print(f"正在加载训练集: {DATA_ROOT}")
    # train_dataset = Unrealego3DPoseDataset(
    #     data_root=TRAIN_DIR, 
    #     img_size=image_size, 
    #     is_train=True, 
    #     target_user_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # )
    
    # print(f"正在加载验证集: {VAL_DIR}")
    # val_dataset = Unrealego3DPoseDataset(
    #     data_root=VAL_DIR, 
    #     img_size=image_size, 
    #     is_train=False, 
    #     target_user_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # )

    # print(f"正在加载测试集: {TEST_DIR}")
    # test_dataset = Unrealego3DPoseDataset(
    #     data_root=TEST_DIR, 
    #     img_size=image_size, 
    #     is_train=False, 
    #     target_user_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # )
   
   # [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12,14]
   # [4, 8, 13]
    print(f"正在加载训练集: {DATA_ROOT}")
    train_dataset = Unrealego3DPoseDataset(
        data_root=DATA_ROOT, 
        img_size=image_size, 
        is_train=True, 
        # target_user_ids=[0, 1, 2, 3, 4, 8, 9, 10, 11, 12,13, 14],
        target_user_ids=[0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 14], 
    )
    
    print(f"正在加载验证集: {DATA_ROOT}")
    val_dataset = Unrealego3DPoseDataset(
        data_root=DATA_ROOT, 
        img_size=image_size, 
        is_train=False, 
        target_user_ids=[4, 8, 13]
    )

    print(f"正在加载测试集: {DATA_ROOT}")
    test_dataset = Unrealego3DPoseDataset(
        data_root=DATA_ROOT, 
        img_size=image_size, 
        is_train=False, 
        target_user_ids=[1, 2, 3]
    )
    

    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    

    print(f"数据加载完成 -> 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

    # ================= 初始化模型 =================
    # 注意：不再传入 train_stage，因为模型现在是纯 3D 架构
    system = PoseLightningModule(
        lr=LEARNING_RATE, 
        local_model_dir=local_model_dir,
        num_joints=21,
        # backbone_ckpt_path=fastvit_ckpt_path, 
    )

    # ================= 配置 Trainer =================
    # 监控 3D MPJPE 指标
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='pose-{epoch:02d}-{val_mpjpe_3d:.4f}', # 文件名带上指标
        save_top_k=3,
        monitor='val_mpjpe_3d',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_mpjpe_3d',
        patience=15,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=20, 
        gradient_clip_val=1.0,
    )

    # # ================= 开始训练 =================
    # print("\n开始训练...")
    
    # # 检查是否存在断点以恢复训练
    # resume_ckpt = None
    # if ckpt_path and os.path.exists(ckpt_path):
    #     print(f"发现检查点，准备恢复训练: {ckpt_path}")
    #     resume_ckpt = ckpt_path
    # else:
    #     print("未指定检查点或检查点不存在，开始重新训练。")

    # trainer.fit(
    #     model=system, 
    #     train_dataloaders=train_loader, 
    #     val_dataloaders=val_loader,
    #     ckpt_path=resume_ckpt
    # )

    # ================= 开始测试 =================
    # 训练结束后，自动加载表现最好的 checkpoint 进行测试
    print("\n训练结束，开始在 Test 集上评估最佳模型...")
    # 'best' 会自动使用 checkpoint_callback 保存的最好模型
    # trainer.test(model=system, dataloaders=test_loader, ckpt_path='best') 
    # trainer.test(model=system, dataloaders=test_loader, ckpt_path='checkpoints/pose-epoch=71-val_mpjpe_3d=21.2636.ckpt') 
    trainer.test(model=system, dataloaders=test_loader, ckpt_path='checkpoints/pose-epoch=52-val_mpjpe_3d=14.8132.ckpt') 

if __name__ == "__main__":
    main()