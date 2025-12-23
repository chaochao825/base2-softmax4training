import json
import matplotlib.pyplot as plt
import numpy as np

# 读取两个JSON文件
with open('results_v3/results_cifar10_vit-s_base2_fp32.json', 'r') as f:
    data_base2 = json.load(f)

with open('results_v3/results_cifar10_vit-s_standard_fp32.json', 'r') as f:
    data_standard = json.load(f)

# 提取前50个epoch的数据
epochs = 50
history_base2 = data_base2['history']
history_standard = data_standard['history']

epochs_base2 = history_base2['epoch'][:epochs]
train_loss_base2 = history_base2['train_loss'][:epochs]
val_loss_base2 = history_base2['val_loss'][:epochs]
val_acc_base2 = history_base2['val_accuracy'][:epochs]

epochs_standard = history_standard['epoch'][:epochs]
train_loss_standard = history_standard['train_loss'][:epochs]
val_loss_standard = history_standard['val_loss'][:epochs]
val_acc_standard = history_standard['val_accuracy'][:epochs]

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形，使用更大的尺寸
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ViT-Small on CIFAR-10: Base-2 vs Standard Softmax (First 50 Epochs)', 
             fontsize=16, fontweight='bold', y=0.98)

# 添加模型和数据集信息
model_info = 'Model: ViT-Small | Dataset: CIFAR-10'
fig.text(0.5, 0.02, model_info, ha='center', fontsize=11, 
         style='italic', color='#555555')

# 绘制训练损失
ax1 = axes[0]
ax1.plot(epochs_base2, train_loss_base2, 'o-', linewidth=3, markersize=6, 
         label='Base-2', color='#2E86AB', markerfacecolor='#2E86AB', 
         markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(epochs_standard, train_loss_standard, 's-', linewidth=3, markersize=6, 
         label='Standard', color='#A23B72', markerfacecolor='#A23B72', 
         markeredgecolor='white', markeredgewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax1.set_title('Training Loss', fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.legend(fontsize=11, framealpha=0.9, loc='upper right')
ax1.tick_params(labelsize=11)

# 绘制验证准确率
ax2 = axes[1]
# 转换为百分比
val_acc_base2_pct = [acc * 100 for acc in val_acc_base2]
val_acc_standard_pct = [acc * 100 for acc in val_acc_standard]
ax2.plot(epochs_base2, val_acc_base2_pct, 'o-', linewidth=3, markersize=6, 
         label='Base-2', color='#2E86AB', markerfacecolor='#2E86AB', 
         markeredgecolor='white', markeredgewidth=1.5)
ax2.plot(epochs_standard, val_acc_standard_pct, 's-', linewidth=3, markersize=6, 
         label='Standard', color='#A23B72', markerfacecolor='#A23B72', 
         markeredgecolor='white', markeredgewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Validation Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.legend(fontsize=11, framealpha=0.9, loc='lower right')
ax2.tick_params(labelsize=11)

# 调整布局，为底部信息留出空间
plt.tight_layout(rect=[0, 0.05, 1, 0.98])

# 保存图像
output_path = 'results_v3/comparison_cifar10_vit-s_50epochs.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图像已保存至: {output_path}")

plt.close()

