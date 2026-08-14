#!/bin/bash
#SBATCH --job-name=mgcg-a40
#SBATCH --account=25m0842
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=mgcg_a40.%j.out
#SBATCH --error=mgcg_a40.%j.err

set -euo pipefail
set -x

echo "===== JOB START (A40 - MOMENTUM) ====="
date
hostname

PROJECT_DIR="$HOME/adam-gcg/Adam-GCG-Universal-Prompt-Injection"
VENV_DIR="$PROJECT_DIR/.venv"

# ---- SAFETY CHECKS ----
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found"
    exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: Virtual environment python not found"
    exit 1
fi

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

echo "Python:"
python -c "import sys; print(sys.executable)"

echo "===== GPU CHECK ====="
nvidia-smi || { echo "GPU not available"; exit 1; }

# ---- RUN FUNCTION ----
run_job () {
    NAME="$1"
    shift

    echo "===== STARTING $NAME ====="
    date

    LOG_FILE="${NAME}.log"

    if ! python universal_prompt_injection.py "$@" > "$LOG_FILE" 2>&1; then
        echo "FAILED $NAME"
        tail -n 20 "$LOG_FILE"
        exit 1
    fi

    echo "SUCCESS $NAME"
}

# ---- COMMON ARGS ----
COMMON_ARGS=(
  --optimizer momentum
  --save_suffix a40_momentum
  --momentum 0.9
  --model llama2
  --target 0
  --tokens 120
  --start 0
  --end 5
  --batch_size 128
  --topk 64
  --num_steps 500
)

# ---- RUNS ----
run_job STATIC "${COMMON_ARGS[@]}" --injection static
run_job SEMI_DYNAMIC "${COMMON_ARGS[@]}" --injection semi-dynamic
run_job DYNAMIC "${COMMON_ARGS[@]}" --injection dynamic

echo "===== JOB COMPLETED ====="
date
