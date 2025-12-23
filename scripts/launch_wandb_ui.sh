#!/bin/bash
# Launch WandB UI for offline monitoring

echo "=================================================="
echo "Starting WandB Offline Monitoring"
echo "=================================================="
echo ""

cd /home/spco/base-2-bitnet

# Find an available port
PORT=8080
while lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo "WandB UI will be available at:"
echo "  http://localhost:$PORT"
echo "  http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=================================================="

# Launch wandb local server
wandb server start --port $PORT

