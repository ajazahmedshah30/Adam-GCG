#!/bin/bash
#SBATCH --job-name=adam-a40-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=adam_a40_smoke.%j.out
#SBATCH --error=adam_a40_smoke.%j.err

set -euo pipefail

echo "===== JOB START (A40 - ADAM SMOKE) ====="
date
hostname

PROJECT_DIR="$HOME/adam-gcg/Adam-GCG-Universal-Prompt-Injection"
VENV_DIR="$PROJECT_DIR/.venv"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: Virtual environment python not found: $VENV_DIR/bin/python"
    exit 1
fi

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

echo "Python executable:"
python -c "import sys; print(sys.executable)"

echo "===== ENV CHECK ====="
python - <<'PY'
import numpy, pandas, torch, tqdm, transformers
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
PY

echo "===== GPU CHECK ====="
nvidia-smi || { echo "GPU not available"; exit 1; }

LOG_DIR="$PROJECT_DIR/logs_a40_smoke"
mkdir -p "$LOG_DIR"

run_job () {
    NAME="$1"
    INJECTION="$2"

    echo "===== STARTING $NAME ====="
    date

    LOG_FILE="${LOG_DIR}/${NAME}_${SLURM_JOB_ID}.log"

    python universal_prompt_injection.py \
      --optimizer adam \
      --save_suffix smoke \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --adam_eps 1e-8 \
      --model llama2 \
      --injection "$INJECTION" \
      --target 0 \
      --tokens 80 \
      --start 0 \
      --end 2 \
      --batch_size 16 \
      --topk 32 \
      --num_steps 10 > "$LOG_FILE" 2>&1

    echo "===== COMPLETED $NAME ====="
}

run_job STATIC static
run_job SEMI_DYNAMIC semi-dynamic
run_job DYNAMIC dynamic

echo "===== JOB COMPLETED ====="
date