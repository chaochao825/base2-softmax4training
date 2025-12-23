#!/bin/bash
# Run LLM experiments comparing Standard vs Base-2 Softmax

set -e

echo "========================================"
echo "LLM Experiments: Base-2 vs Standard Softmax"
echo "========================================"
echo ""

# Configuration
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2
NUM_GPUS=3
OUTPUT_DIR="/home/spco/base-2-bitnet/results"
LOG_DIR="/home/spco/base-2-bitnet/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Experiment 1: GPT-2 with Standard Softmax
echo "========================================"
echo "Experiment 1: GPT-2 with Standard Softmax"
echo "========================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    scripts/train_llm_enhanced.py \
    --model_name gpt2 \
    --softmax standard \
    --batch_size 8 \
    --epochs 103 \
    --lr 5e-5 \
    --max_length 256 \
    --num_train_samples 10000 \
    --num_val_samples 1000 \
    --track_grads \
    --wandb \
    --output_dir "$OUTPUT_DIR" \
    --save_every 10 \
    2>&1 | tee "$LOG_DIR/llm_gpt2_standard.log"

echo ""
echo "✓ Experiment 1 completed"
echo ""
echo "Waiting 10 seconds before next experiment..."
sleep 10

# Experiment 2: GPT-2 with Base-2 Softmax
echo "========================================"
echo "Experiment 2: GPT-2 with Base-2 Softmax"
echo "========================================"
echo ""

torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    scripts/train_llm_enhanced.py \
    --model_name gpt2 \
    --softmax base2 \
    --batch_size 8 \
    --epochs 103 \
    --lr 5e-5 \
    --max_length 256 \
    --num_train_samples 10000 \
    --num_val_samples 1000 \
    --track_grads \
    --wandb \
    --output_dir "$OUTPUT_DIR" \
    --save_every 10 \
    2>&1 | tee "$LOG_DIR/llm_gpt2_base2.log"

echo ""
echo "✓ Experiment 2 completed"
echo ""

echo "========================================"
echo "All LLM experiments completed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"

