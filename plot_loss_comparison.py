import json
import matplotlib.pyplot as plt
import numpy as np

# 读取两个JSON文件
with open('results_v3/gpt2_xl/history_gpt2_base2.json', 'r') as f:
    data_base2 = json.load(f)

with open('results_v3/gpt2_xl/history_gpt2_standard.json', 'r') as f:
    data_standard = json.load(f)

# 提取前8个epoch的数据
epochs = 8
epochs_base2 = data_base2['epoch'][:epochs]
train_loss_base2 = data_base2['train_loss'][:epochs]
val_loss_base2 = data_base2['val_loss'][:epochs]

epochs_standard = data_standard['epoch'][:epochs]
train_loss_standard = data_standard['train_loss'][:epochs]
val_loss_standard = data_standard['val_loss'][:epochs]

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形，使用更大的尺寸
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Training and Validation Loss Comparison (First 8 Epochs)', 
             fontsize=16, fontweight='bold', y=0.98)

# 添加模型和数据集信息
model_info = 'Model: GPT-2 XL (1.5B parameters) | Dataset: wikitext-2-raw-v1'
fig.text(0.5, 0.02, model_info, ha='center', fontsize=11, 
         style='italic', color='#555555')

# 绘制训练损失
ax1.plot(epochs_base2, train_loss_base2, 'o-', linewidth=3, markersize=8, 
         label='Base-2', color='#2E86AB', markerfacecolor='#2E86AB', 
         markeredgecolor='white', markeredgewidth=1.5)
ax1.plot(epochs_standard, train_loss_standard, 's-', linewidth=3, markersize=8, 
         label='Standard', color='#A23B72', markerfacecolor='#A23B72', 
         markeredgecolor='white', markeredgewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Training Loss', fontsize=13, fontweight='bold')
ax1.set_title('Training Loss', fontsize=14, fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax1.legend(fontsize=11, framealpha=0.9, loc='upper right')
ax1.set_xticks(range(1, epochs + 1))
ax1.tick_params(labelsize=11)

# 绘制验证损失
ax2.plot(epochs_base2, val_loss_base2, 'o-', linewidth=3, markersize=8, 
         label='Base-2', color='#2E86AB', markerfacecolor='#2E86AB', 
         markeredgecolor='white', markeredgewidth=1.5)
ax2.plot(epochs_standard, val_loss_standard, 's-', linewidth=3, markersize=8, 
         label='Standard', color='#A23B72', markerfacecolor='#A23B72', 
         markeredgecolor='white', markeredgewidth=1.5)
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
ax2.set_title('Validation Loss', fontsize=14, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2.legend(fontsize=11, framealpha=0.9, loc='upper right')
ax2.set_xticks(range(1, epochs + 1))
ax2.tick_params(labelsize=11)

# 调整布局，为底部信息留出空间
plt.tight_layout(rect=[0, 0.05, 1, 0.98])

# 保存图像（新文件名，不覆盖原来的）
output_path = 'results_v3/gpt2_xl/loss_comparison_e008_with_info.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图像已保存至: {output_path}")

plt.show()

