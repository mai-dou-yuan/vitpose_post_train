import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(file_path, loss_ylim=None, mpjpe_ylim=None):
    # 读取CSV文件
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return

    # 打印列名以供检查 (防止列名中有意外的空格)
    print("CSV中的列名:", df.columns.tolist())

    # 配置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid')  # 如果报错可改为 plt.style.use('ggplot')
    plt.figure(figsize=(16, 6))

    # --- 图表 1: Loss 对比 ---
    plt.subplot(1, 2, 1)
    
    # 提取并绘制训练 Loss (过滤掉 NaN 值)
    train_loss = df.dropna(subset=['loss_3d'])
    plt.plot(train_loss['step'], train_loss['loss_3d'], 
             label='Train Loss (loss_3d)', alpha=0.6, linewidth=1)
    
    # 提取并绘制验证 Loss (过滤掉 NaN 值)
    val_loss = df.dropna(subset=['val_loss'])
    plt.plot(val_loss['step'], val_loss['val_loss'], 
             label='Val Loss (val_loss)', marker='o', linestyle='-', linewidth=2, color='orange')
    
    plt.title('Loss Curve (loss_3d vs val_loss)')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    if loss_ylim is not None:
        plt.ylim(loss_ylim)  # 设置 y 轴范围
    plt.legend()
    plt.grid(True)

    # --- 图表 2: MPJPE 3D 对比 ---
    plt.subplot(1, 2, 2)
    
    # 提取并绘制训练 MPJPE
    train_mpjpe = df.dropna(subset=['train_mpjpe_3d'])
    plt.plot(train_mpjpe['step'], train_mpjpe['train_mpjpe_3d'], 
             label='Train MPJPE (train_mpjpe_3d)', alpha=0.6, linewidth=1)
    
    # 提取并绘制验证 MPJPE
    val_mpjpe = df.dropna(subset=['val_mpjpe_3d'])
    plt.plot(val_mpjpe['step'], val_mpjpe['val_mpjpe_3d'], 
             label='Val MPJPE (val_mpjpe_3d)', marker='o', linestyle='-', linewidth=2, color='orange')
    
    plt.title('MPJPE 3D Curve')
    plt.xlabel('Step')
    plt.ylabel('MPJPE 3D')
    if mpjpe_ylim is not None:
        plt.ylim(mpjpe_ylim)  # 设置 y 轴范围
    plt.legend()
    plt.grid(True)

    # 调整布局并保存/显示
    plt.tight_layout()
    plt.savefig('metrics_plot.png')  # 保存为图片
    plt.show()

if __name__ == "__main__":
    csv_file = 'lightning_logs/version_3/metrics.csv'
    # 示例：指定 y 轴范围，如 loss 在 [0, 100]，MPJPE 在 [20, 80]
    plot_metrics(csv_file, loss_ylim=None, mpjpe_ylim=(0, 20))