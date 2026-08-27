import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv("lightning_logs/version_14/metrics.csv")

# 创建一个包含所有数据的图表
plt.figure(figsize=(15, 10))

# 1. MPJPE 可视化 (训练步骤 vs 验证)
plt.subplot(2, 1, 1)
# 绘制每个训练步骤的MPJPE
plt.plot(df['step'], df['train_mpjpe_step'], 'b-', alpha=0.3, label='Train MPJPE (per step)')
# 提取验证数据点（非空的val_mpjpe）
val_data = df[df['val_mpjpe'].notna()]
plt.plot(val_data['step'], val_data['val_mpjpe'], 'ro-', label='Validation MPJPE')
# 提取每个epoch的训练MPJPE
epoch_train_data = df[df['train_mpjpe_epoch'].notna()]
plt.plot(epoch_train_data['step'], epoch_train_data['train_mpjpe_epoch'], 'bo-', label='Train MPJPE (epoch avg)')

plt.title('MPJPE Comparison During Training', fontsize=14)
plt.ylabel('MPJPE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 2. Loss 可视化 (训练步骤 vs 验证)
plt.subplot(2, 1, 2)
# 绘制每个训练步骤的loss
plt.plot(df['step'], df['train_loss_step'], 'g-', alpha=0.3, label='Train Loss (per step)')
# 绘制验证loss
plt.plot(val_data['step'], val_data['val_loss'], 'mo-', label='Validation Loss')
# 绘制每个epoch的训练loss
plt.plot(epoch_train_data['step'], epoch_train_data['train_loss_epoch'], 'go-', label='Train Loss (epoch avg)')

plt.title('Loss Comparison During Training', fontsize=14)
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_metrics.png', dpi=300)
plt.show()

# 3. 创建按epoch的MPJPE对比图
plt.figure(figsize=(10, 6))
# 每个epoch的训练MPJPE
epochs = epoch_train_data['epoch'].unique()
train_mpjpe_per_epoch = [epoch_train_data[epoch_train_data['epoch'] == e]['train_mpjpe_epoch'].values[-1] for e in epochs]
# 每个epoch的验证MPJPE
val_mpjpe_per_epoch = [val_data[val_data['epoch'] == e]['val_mpjpe'].values[-1] if len(val_data[val_data['epoch'] == e]) > 0 else np.nan for e in epochs]

plt.plot(epochs, train_mpjpe_per_epoch, 'bo-', label='Train MPJPE')
plt.plot(epochs, val_mpjpe_per_epoch, 'ro-', label='Validation MPJPE')

plt.title('MPJPE by Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MPJPE', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('mpjpe_by_epoch.png', dpi=300)
plt.show()

# 4. 创建按epoch的Loss对比图
plt.figure(figsize=(10, 6))
# 每个epoch的训练loss
train_loss_per_epoch = [epoch_train_data[epoch_train_data['epoch'] == e]['train_loss_epoch'].values[-1] for e in epochs]
# 每个epoch的验证loss
val_loss_per_epoch = [val_data[val_data['epoch'] == e]['val_loss'].values[-1] if len(val_data[val_data['epoch'] == e]) > 0 else np.nan for e in epochs]

plt.plot(epochs, train_loss_per_epoch, 'go-', label='Train Loss')
plt.plot(epochs, val_loss_per_epoch, 'mo-', label='Validation Loss')

plt.title('Loss by Epoch', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('loss_by_epoch.png', dpi=300)
plt.show()

# 基本统计信息
print("训练最终指标:")
final_epoch = int(epochs[-1])
print(f"最终Epoch: {final_epoch}")
print(f"训练MPJPE: {train_mpjpe_per_epoch[-1]:.4f}")
print(f"验证MPJPE: {val_mpjpe_per_epoch[-1]:.4f}")
print(f"训练Loss: {train_loss_per_epoch[-1]:.4f}")
print(f"验证Loss: {val_loss_per_epoch[-1]:.4f}")

# 找出最佳验证性能的epoch
best_epoch_idx = np.nanargmin(val_mpjpe_per_epoch)
best_epoch = epochs[best_epoch_idx]
print(f"\n最佳验证MPJPE在 Epoch {best_epoch}: {val_mpjpe_per_epoch[best_epoch_idx]:.4f}")