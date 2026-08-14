#!/bin/bash
#SBATCH --job-name=adam-all
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=adam_all.%j.out

# Go to project directory
cd ~/adam-gcg/Adam-GCG-Universal-Prompt-Injection || exit 1

# Activate environment
source .venv/bin/activate

echo "Using Python:"
which python

# Quick sanity check (fail early if broken)
python - <<'EOF'
import numpy, torch
print("Environment OK")
EOF

LOG_FILE="run_status.log"
echo "Run started at $(date)" > "$LOG_FILE"

run_job () {
    NAME=$1
    shift

    echo "===== STARTING $NAME ====="
    echo "[$(date)] START $NAME" >> "$LOG_FILE"

    "$@" >> "${NAME}.log" 2>&1
    STATUS=$?

    if [ $STATUS -eq 0 ]; then
        echo "[$(date)] SUCCESS $NAME" >> "$LOG_FILE"
        echo "===== $NAME SUCCESS ====="
    else
        echo "[$(date)] FAILED $NAME (exit code $STATUS)" >> "$LOG_FILE"
        echo "===== $NAME FAILED ====="
    fi
}

echo "GPU Info:" >> "$LOG_FILE"
nvidia-smi >> "$LOG_FILE" 2>&1

# ---------------- RUNS ---------------- #

run_job "STATIC" \
python universal_prompt_injection.py \
  --optimizer adam \
  --save_suffix dgx_adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_eps 1e-8 \
  --model llama2 \
  --injection static \
  --target 0 \
  --tokens 120 \
  --start 0 \
  --end 5 \
  --batch_size 128 \
  --topk 64 \
  --num_steps 500

run_job "SEMI_DYNAMIC" \
python universal_prompt_injection.py \
  --optimizer adam \
  --save_suffix dgx_adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_eps 1e-8 \
  --model llama2 \
  --injection semi-dynamic \
  --target 0 \
  --tokens 120 \
  --start 0 \
  --end 5 \
  --batch_size 128 \
  --topk 64 \
  --num_steps 500

run_job "DYNAMIC" \
python universal_prompt_injection.py \
  --optimizer adam \
  --save_suffix dgx_adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_eps 1e-8 \
  --model llama2 \
  --injection dynamic \
  --target 0 \
  --tokens 120 \
  --start 0 \
  --end 5 \
  --batch_size 128 \
  --topk 64 \
  --num_steps 500

echo "Run completed at $(date)" >> "$LOG_FILE"
echo "===== ALL JOBS COMPLETED ====="