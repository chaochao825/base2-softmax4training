#!/usr/bin/env bash
# Wait for a free GPU and then launch the LLM base-2 vs standard softmax eval in a detached session.
# Criteria: utilization <= UTIL_THRESH and memory used <= MEM_THRESH.

set -euo pipefail

PREFERRED_GPUS=(2 3 1 0)
UTIL_THRESH=30        # percent
MEM_THRESH=12000      # MiB
SLEEP_SEC=60

PROJECT_DIR="/home/spco/base-2-bitnet"
CMD="python scripts/run_llm_base2_eval.py \
  --models Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-8B-Instruct meta-llama/Llama-3.2-3B \
  --quant_modes fp16 fp8 int8 4bit \
  --samples 256 \
  --max_length 512 \
  --output_dir /home/spco/base-2-bitnet/new_result/llm_base2_eval"

echo "[wait-run] Starting wait loop at $(date '+%F %T')"

pick_gpu() {
  local idx util mem tot
  while IFS=',' read -r idx util mem tot; do
    idx="${idx//[!0-9]/}"
    util="${util//[!0-9]/}"
    mem="${mem//[!0-9]/}"
    tot="${tot//[!0-9]/}"
    printf "%s %s %s %s\n" "$idx" "$util" "$mem" "$tot"
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
}

while true; do
  mapfile -t gpu_stats < <(pick_gpu)
  chosen=""
  for g in "${PREFERRED_GPUS[@]}"; do
    for line in "${gpu_stats[@]}"; do
      read -r idx util mem tot <<<"$line"
      if [[ "$idx" == "$g" ]]; then
        if (( util <= UTIL_THRESH && mem <= MEM_THRESH )); then
          chosen="$idx"
        fi
        break
      fi
    done
    [[ -n "$chosen" ]] && break
  done

  if [[ -n "$chosen" ]]; then
    echo "[wait-run] GPU ${chosen} is free enough (util<=${UTIL_THRESH}%, mem<=${MEM_THRESH}MiB). Launching..."
    export CUDA_VISIBLE_DEVICES="$chosen"
    cd "$PROJECT_DIR"
    echo "[wait-run] Command: $CMD"
    exec bash -lc "$CMD"
  else
    echo "[wait-run] No GPU free (<=${UTIL_THRESH}% util & <=${MEM_THRESH}MiB mem). Sleeping ${SLEEP_SEC}s..."
    sleep "$SLEEP_SEC"
  fi
done
