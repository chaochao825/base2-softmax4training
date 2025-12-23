#!/bin/bash
# Quick status checker for all experiments

echo "========================================"
echo "Experiment Status Dashboard"
echo "========================================"
echo ""
echo "Time: $(date)"
echo ""

# Check running processes
echo "Running Processes:"
echo "------------------"
if pgrep -f "run_comprehensive_experiments.py" > /dev/null; then
    echo "✓ Vision experiments: RUNNING"
else
    echo "✗ Vision experiments: NOT RUNNING"
fi

if pgrep -f "auto_run_llm_after_vision.sh" > /dev/null; then
    echo "✓ LLM auto-runner: ACTIVE (waiting/running)"
else
    echo "✗ LLM auto-runner: NOT ACTIVE"
fi

if pgrep -f "train_llm_enhanced.py" > /dev/null; then
    echo "✓ LLM experiments: RUNNING"
else
    echo "✗ LLM experiments: NOT RUNNING"
fi

echo ""
echo "GPU Status:"
echo "-----------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s%% util, %sMB/%sMB\n", $1, $3, $4, $5}'

echo ""
echo "Completed Results:"
echo "------------------"
RESULTS_DIR="/home/spco/base-2-bitnet/results"

# Vision results
VISION_COUNT=$(ls -1 $RESULTS_DIR/results_cifar10*.json 2>/dev/null | wc -l)
echo "Vision experiments: $VISION_COUNT/4"
if [ -f "$RESULTS_DIR/results_cifar10_resnet18_standard.json" ]; then echo "  ✓ ResNet-18 Standard"; fi
if [ -f "$RESULTS_DIR/results_cifar10_resnet18_base2.json" ]; then echo "  ✓ ResNet-18 Base-2"; fi
if [ -f "$RESULTS_DIR/results_cifar10_vit-s_standard.json" ]; then echo "  ✓ ViT-Small Standard"; fi
if [ -f "$RESULTS_DIR/results_cifar10_vit-s_base2.json" ]; then echo "  ✓ ViT-Small Base-2"; fi

# LLM results
LLM_COUNT=$(ls -1 $RESULTS_DIR/results_llm*.json 2>/dev/null | wc -l)
echo "LLM experiments: $LLM_COUNT/2"
if [ -f "$RESULTS_DIR/results_llm_gpt2_standard.json" ]; then echo "  ✓ GPT-2 Standard"; fi
if [ -f "$RESULTS_DIR/results_llm_gpt2_base2.json" ]; then echo "  ✓ GPT-2 Base-2"; fi

echo ""
echo "Latest Activity:"
echo "----------------"

# Check vision experiment log
if [ -f "/home/spco/base-2-bitnet/logs/comprehensive_experiments.log" ]; then
    echo "Vision experiments (last line):"
    tail -n 1 "/home/spco/base-2-bitnet/logs/comprehensive_experiments.log" | sed 's/^/  /'
fi

# Check LLM log
if [ -f "/home/spco/base-2-bitnet/logs/auto_llm_experiments.log" ]; then
    echo "LLM experiments (last line):"
    tail -n 1 "/home/spco/base-2-bitnet/logs/auto_llm_experiments.log" | sed 's/^/  /'
fi

echo ""
echo "========================================"

# Overall progress
TOTAL_COMPLETE=$((VISION_COUNT + LLM_COUNT))
echo "Overall Progress: $TOTAL_COMPLETE/6 experiments complete"

if [ $TOTAL_COMPLETE -eq 6 ]; then
    echo "🎉 ALL EXPERIMENTS COMPLETE! 🎉"
    echo ""
    echo "Run the following to generate final report:"
    echo "  cd /home/spco/base-2-bitnet"
    echo "  conda activate base-2-bitnet"
    echo "  python scripts/generate_final_report.py"
fi

echo "========================================"

