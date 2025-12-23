#!/bin/bash
# Monitor running experiments

echo "=================================="
echo "Experiment Progress Monitor"
echo "=================================="
echo ""

# Check if comprehensive experiment is running
if pgrep -f "run_comprehensive_experiments.py" > /dev/null; then
    echo "✓ Comprehensive experiments are RUNNING"
    echo ""
    echo "GPU Usage:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU %s (%s): %s%% utilization, %sMB / %sMB\n", $1, $2, $3, $4, $5}'
    echo ""
else
    echo "✗ No experiments currently running"
    echo ""
fi

# Check latest log
LOG_FILE="/home/spco/base-2-bitnet/logs/comprehensive_experiments.log"
if [ -f "$LOG_FILE" ]; then
    echo "Latest Log Entries (last 30 lines):"
    echo "===================================="
    tail -n 30 "$LOG_FILE"
else
    echo "No log file found yet"
fi

echo ""
echo "===================================="

# Check results directory
RESULTS_DIR="/home/spco/base-2-bitnet/results"
if [ -d "$RESULTS_DIR" ]; then
    echo ""
    echo "Completed Results:"
    echo "===================="
    ls -lh "$RESULTS_DIR"/*.json 2>/dev/null | awk '{print "  " $9}' || echo "  No results yet"
fi

