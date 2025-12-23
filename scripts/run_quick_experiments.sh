#!/bin/bash
# Quick experiments for base-2 softmax comparison

set -e

RESULTS_DIR="/home/spco/base-2-bitnet/results"
LOGS_DIR="/home/spco/base-2-bitnet/logs"

mkdir -p $RESULTS_DIR $LOGS_DIR

echo "========================================================================"
echo "Quick Experiment Suite: Base-2 Softmax vs Standard Softmax"
echo "========================================================================"
echo ""

# Experiment 1: ResNet-18 CIFAR-10 Standard
echo "[1/4] Running ResNet-18 on CIFAR-10 with Standard Softmax..."
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
    --save_checkpoints \
    2>&1 | tee $LOGS_DIR/resnet18_cifar10_standard.log

echo "✓ Experiment 1 complete"
echo ""

# Experiment 2: ResNet-18 CIFAR-10 Base-2
echo "[2/4] Running ResNet-18 on CIFAR-10 with Base-2 Softmax..."
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
    --save_checkpoints \
    2>&1 | tee $LOGS_DIR/resnet18_cifar10_base2.log

echo "✓ Experiment 2 complete"
echo ""

# Experiment 3: ViT-Small CIFAR-10 Standard
echo "[3/4] Running ViT-Small on CIFAR-10 with Standard Softmax..."
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
    --save_checkpoints \
    2>&1 | tee $LOGS_DIR/vit_small_cifar10_standard.log

echo "✓ Experiment 3 complete"
echo ""

# Experiment 4: ViT-Small CIFAR-10 Base-2
echo "[4/4] Running ViT-Small on CIFAR-10 with Base-2 Softmax..."
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
    --save_checkpoints \
    2>&1 | tee $LOGS_DIR/vit_small_cifar10_base2.log

echo "✓ Experiment 4 complete"
echo ""

# Generate comparison report
echo "========================================================================"
echo "Generating Comparison Report..."
echo "========================================================================"
python scripts/generate_report.py

echo ""
echo "========================================================================"
echo "All experiments completed successfully!"
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOGS_DIR"
echo "========================================================================"


