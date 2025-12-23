#!/bin/bash
# Automatically run LLM experiments after vision experiments complete

echo "========================================"
echo "Waiting for Vision Experiments to Complete"
echo "========================================"
echo ""

# Wait for comprehensive experiments to finish
while pgrep -f "run_comprehensive_experiments.py" > /dev/null; do
    echo "Vision experiments still running..."
    echo "Current time: $(date)"
    
    # Show latest progress
    if [ -f "/home/spco/base-2-bitnet/logs/comprehensive_experiments.log" ]; then
        echo "Latest progress:"
        tail -n 5 "/home/spco/base-2-bitnet/logs/comprehensive_experiments.log"
    fi
    
    echo ""
    echo "Checking again in 5 minutes..."
    sleep 300  # Wait 5 minutes
done

echo ""
echo "========================================"
echo "✓ Vision Experiments Completed!"
echo "========================================"
echo ""

# Check results
echo "Checking results..."
ls -lh /home/spco/base-2-bitnet/results/*.json

echo ""
echo "Waiting 30 seconds before starting LLM experiments..."
sleep 30

# Start LLM experiments
echo ""
echo "========================================"
echo "Starting LLM Experiments"
echo "========================================"
echo ""

cd /home/spco/base-2-bitnet
source ~/.bashrc
conda activate base-2-bitnet
export WANDB_MODE=offline

bash scripts/run_llm_experiments.sh

echo ""
echo "========================================"
echo "✓ All Experiments Completed!"
echo "========================================"

