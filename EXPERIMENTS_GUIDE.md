# Base-2 Softmax vs Standard Softmax实验指南

## 📋 实验概览

本项目系统性地对比了在BitNet 1.58-bit量化场景下，**Standard Softmax (base-e)** 与 **Base-2 Softmax** 的性能差异。

### 核心研究问题

1. **稳定性**: Base-2 Softmax是否能在超低比特量化环境中提供更好的训练稳定性？
2. **性能**: 两种Softmax对最终模型准确率的影响如何？
3. **梯度**: Base-2 Softmax是否产生更平滑的梯度？

---

## 🚀 快速开始

### 环境设置

```bash
# 激活conda环境
conda activate base-2-bitnet

# 验证GPU可用性
nvidia-smi

# 验证依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 运行单个实验

#### 实验1: ResNet-18 on CIFAR-10

```bash
# Standard Softmax
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-3 \
    --softmax standard \
    --scheduler cosine \
    --warmup_epochs 5 \
    --track_grads \
    --save_checkpoints

# Base-2 Softmax
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-3 \
    --softmax base2 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --track_grads \
    --save_checkpoints
```

#### 实验2: ViT-Small on CIFAR-10

```bash
# Standard Softmax
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model vit-s \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --softmax standard \
    --scheduler cosine \
    --warmup_epochs 10 \
    --track_grads \
    --save_checkpoints

# Base-2 Softmax
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model vit-s \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --softmax base2 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --track_grads \
    --save_checkpoints
```

#### 实验3: ImageNet-100 (需要下载数据)

```bash
# 下载ImageNet-100到 /amax/storage/nfs/spco/data/imagenet/

# ViT-Base on ImageNet-100, Standard Softmax
python scripts/train_enhanced.py \
    --dataset ImageNet-100 \
    --data_path /amax/storage/nfs/spco/data \
    --model vit-b \
    --batch_size 64 \
    --epochs 150 \
    --lr 3e-4 \
    --softmax standard \
    --scheduler cosine \
    --warmup_epochs 15 \
    --track_grads \
    --save_checkpoints

# ViT-Base on ImageNet-100, Base-2 Softmax
python scripts/train_enhanced.py \
    --dataset ImageNet-100 \
    --data_path /amax/storage/nfs/spco/data \
    --model vit-b \
    --batch_size 64 \
    --epochs 150 \
    --lr 3e-4 \
    --softmax base2 \
    --scheduler cosine \
    --warmup_epochs 15 \
    --track_grads \
    --save_checkpoints
```

#### 实验4: LLM (TinyLlama on TinyStories)

```bash
# Standard Softmax
python scripts/train_llm.py \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --dataset roneneldan/TinyStories \
    --softmax standard \
    --batch_size 4 \
    --epochs 1 \
    --num_train_samples 5000 \
    --max_length 512

# Base-2 Softmax
python scripts/train_llm.py \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --dataset roneneldan/TinyStories \
    --softmax base2 \
    --batch_size 4 \
    --epochs 1 \
    --num_train_samples 5000 \
    --max_length 512
```

### 批量运行所有实验

```bash
# 运行所有图像分类实验
bash scripts/run_quick_experiments.sh

# 查看进度
tail -f logs/full_experiments.log
```

---

## 📊 结果分析

### 生成对比报告

```bash
python scripts/generate_report.py
```

这将生成：
- `results/comparison_report_accuracy_comparison.png` - 准确率对比曲线
- `results/comparison_report_loss_comparison.png` - 损失对比曲线
- `results/comparison_report_gradient_norms.png` - 梯度范数对比
- `results/comparison_report_final_comparison.png` - 最终性能柱状图
- `results/summary.txt` - 文本格式总结

### 查看单个实验结果

```bash
# 查看JSON结果
cat results/results_cifar10_resnet18_standard.json
cat results/results_cifar10_resnet18_base2.json

# 查看训练曲线
ls results/curves_*.png
```

---

## 🔧 高级选项

### 多GPU训练

```bash
# 使用torchrun进行DDP训练
torchrun --nproc_per_node=3 scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --batch_size 256 \
    --epochs 50 \
    --softmax base2 \
    --track_grads
```

### WandB集成

```bash
# 启用Weights & Biases日志
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --softmax base2 \
    --wandb
```

### 恢复训练

```bash
# 从checkpoint恢复
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --softmax base2 \
    --resume results/best_resnet18_CIFAR-10_base2.pth
```

### 超参数调优

```bash
# 调整温度参数
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --softmax base2 \
    --temperature 0.5  # 尝试不同温度

# 调整学习率和调度器
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model vit-s \
    --softmax base2 \
    --lr 1e-3 \
    --scheduler step \
    --warmup_epochs 5
```

---

## 📈 预期结果

根据初步假设：

### 假设1: 训练稳定性
- **预期**: Base-2 Softmax产生更平滑的梯度
- **验证**: 查看 `gradient_norms.png` 图表
- **指标**: 梯度L2范数方差更小

### 假设2: 模型性能
- **预期**: 两种Softmax性能相当，或Base-2在某些情况下因稳定性更好而略优
- **验证**: 查看 `final_comparison.png` 柱状图
- **指标**: Top-1准确率差异 < 1%

### 假设3: 收敛速度
- **预期**: Base-2可能收敛稍慢但更稳定
- **验证**: 查看 `accuracy_comparison.png` 曲线
- **指标**: 达到90%最佳准确率所需epoch数

---

## 🐛 故障排除

### CUDA错误

```bash
# 检查CUDA可用性
python -c "import torch; print(torch.cuda.is_available())"

# 指定单个GPU
CUDA_VISIBLE_DEVICES=0 python scripts/train_enhanced.py ...
```

### 内存不足

```bash
# 减小batch size
python scripts/train_enhanced.py --batch_size 64 ...

# 使用梯度累积
python scripts/train_enhanced.py --gradient_accumulation_steps 4 ...
```

### 数据集下载失败

```bash
# CIFAR-10/100已在 /amax/storage/nfs/spco/data/
ls /amax/storage/nfs/spco/data/

# 手动下载ImageNet（如需要）
# 解压到 /amax/storage/nfs/spco/data/imagenet/
```

---

## 📝 引用

如果使用本实验框架，请引用：

```bibtex
@software{base2_softmax_bitnet_2025,
    title={Base-2 Softmax in Ultra-Low-Bit Quantization: An Empirical Study},
    author={Your Name},
    year={2025},
    url={https://github.com/yourusername/base-2-bitnet}
}
```

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License


