#!/bin/bash
# 并行运行剩余的实验
# 视觉实验使用GPU 0,1 - LLM使用GPU 2

set -e

echo "========================================"
echo "并行执行剩余实验"
echo "========================================"
echo ""
echo "GPU分配:"
echo "  GPU 0, 1: 视觉实验 (ViT-Small base2)"
echo "  GPU 2:    LLM实验 (GPT-2 standard + base2)"
echo ""
echo "已完成的实验:"
ls -1 /home/spco/base-2-bitnet/results/results_*.json 2>/dev/null | wc -l | xargs echo "  "
echo ""
echo "========================================"
echo ""

export WANDB_MODE=offline
BASE_DIR="/home/spco/base-2-bitnet"
LOG_DIR="$BASE_DIR/logs"
RESULTS_DIR="$BASE_DIR/results"

mkdir -p "$LOG_DIR"

# 函数: 运行视觉实验（如果需要）
run_vision_remaining() {
    echo ""
    echo "【视觉实验】检查 ViT-Small base2..."
    
    if [ ! -f "$RESULTS_DIR/results_cifar10_vit-s_base2.json" ]; then
        echo "→ 需要运行 ViT-Small base2"
        echo "  使用 GPU 0,1"
        echo ""
        
        export CUDA_VISIBLE_DEVICES=0,1
        
        torchrun --nproc_per_node=2 \
            --master_port=29500 \
            "$BASE_DIR/scripts/train_enhanced.py" \
            --dataset=CIFAR-10 \
            --model=vit-s \
            --softmax=base2 \
            --batch_size=256 \
            --epochs=100 \
            --lr=5e-4 \
            --scheduler=cosine \
            --warmup_epochs=10 \
            --track_grads \
            --save_checkpoints \
            --wandb \
            2>&1 | tee "$LOG_DIR/parallel_vit-s_base2.log"
        
        echo "✓ ViT-Small base2 完成"
    else
        echo "✓ ViT-Small base2 已完成，跳过"
    fi
}

# 函数: 运行LLM实验
run_llm_experiments() {
    echo ""
    echo "【LLM实验】开始..."
    echo "  使用 GPU 2"
    echo ""
    
    export CUDA_VISIBLE_DEVICES=2
    
    # GPT-2 Standard
    if [ ! -f "$RESULTS_DIR/results_llm_gpt2_standard.json" ]; then
        echo "→ 运行 GPT-2 Standard"
        
        python "$BASE_DIR/scripts/train_llm_enhanced.py" \
            --model_name=gpt2 \
            --softmax=standard \
            --batch_size=8 \
            --epochs=3 \
            --lr=5e-5 \
            --max_length=256 \
            --num_train_samples=10000 \
            --num_val_samples=1000 \
            --track_grads \
            --wandb \
            2>&1 | tee "$LOG_DIR/parallel_llm_gpt2_standard.log"
        
        echo "✓ GPT-2 Standard 完成"
    else
        echo "✓ GPT-2 Standard 已完成，跳过"
    fi
    
    echo ""
    sleep 5
    
    # GPT-2 Base-2
    if [ ! -f "$RESULTS_DIR/results_llm_gpt2_base2.json" ]; then
        echo "→ 运行 GPT-2 Base-2"
        
        python "$BASE_DIR/scripts/train_llm_enhanced.py" \
            --model_name=gpt2 \
            --softmax=base2 \
            --batch_size=8 \
            --epochs=3 \
            --lr=5e-5 \
            --max_length=256 \
            --num_train_samples=10000 \
            --num_val_samples=1000 \
            --track_grads \
            --wandb \
            2>&1 | tee "$LOG_DIR/parallel_llm_gpt2_base2.log"
        
        echo "✓ GPT-2 Base-2 完成"
    else
        echo "✓ GPT-2 Base-2 已完成，跳过"
    fi
}

# 并行执行
echo "启动并行执行..."
echo ""

# 后台运行视觉实验
run_vision_remaining &
VISION_PID=$!

# 后台运行LLM实验
run_llm_experiments &
LLM_PID=$!

# 等待两个进程完成
echo "等待进程完成..."
echo "  视觉实验 PID: $VISION_PID"
echo "  LLM实验 PID: $LLM_PID"
echo ""

wait $VISION_PID
VISION_STATUS=$?

wait $LLM_PID
LLM_STATUS=$?

echo ""
echo "========================================"
echo "并行执行完成"
echo "========================================"
echo "  视觉实验状态: $VISION_STATUS"
echo "  LLM实验状态: $LLM_STATUS"
echo ""

if [ $VISION_STATUS -eq 0 ] && [ $LLM_STATUS -eq 0 ]; then
    echo "✓ 所有实验成功完成！"
    exit 0
else
    echo "✗ 部分实验失败"
    exit 1
fi

