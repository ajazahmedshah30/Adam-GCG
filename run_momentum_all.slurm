#!/bin/bash
#SBATCH --job-name=mgcg-momentum
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=mgcg_momentum.%j.out
#SBATCH --error=mgcg_momentum.%j.err

# Strict mode (no silent bugs)
set -euo pipefail
set -x

echo "===== JOB START ====="
date
hostname

# ---------------- PATHS ---------------- #
PROJECT_DIR="$HOME/adam-gcg/Adam-GCG-Universal-Prompt-Injection"
VENV_DIR="$PROJECT_DIR/.venv"

# ---------------- SAFETY CHECKS ---------------- #
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found: $VENV_DIR"
    exit 1
fi

# ---------------- ENV SETUP ---------------- #
source "$VENV_DIR/bin/activate"

echo "===== PYTHON INFO ====="
which python
python -c "import sys; print(sys.executable)"

echo "===== PACKAGE CHECK ====="
python - <<EOF
import numpy, torch, pandas, tqdm
print("NumPy:", numpy.__version__)
print("Torch:", torch.__version__)
print("Pandas:", pandas.__version__)
print("TQDM:", tqdm.__version__)
EOF

# GPU check
echo "===== GPU INFO ====="
nvidia-smi || { echo "ERROR: GPU not available"; exit 1; }

echo "===== ENV READY ====="

# ---------------- RUN FUNCTION ---------------- #
run_job () {
    NAME="$1"
    shift

    echo "===== STARTING $NAME ====="
    date

    LOG_FILE="${NAME}.log"

    python universal_prompt_injection.py "$@" > "$LOG_FILE" 2>&1
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "===== $NAME FAILED (exit $STATUS) ====="
        echo "Check log: $LOG_FILE"
        tail -n 20 "$LOG_FILE"
        exit $STATUS
    fi

    echo "===== $NAME SUCCESS ====="
}

# ---------------- RUNS (MOMENTUM) ---------------- #

COMMON_ARGS=(
  --optimizer momentum
  --save_suffix dgx_momentum
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

run_job STATIC "${COMMON_ARGS[@]}" --injection static
run_job SEMI_DYNAMIC "${COMMON_ARGS[@]}" --injection semi-dynamic
run_job DYNAMIC "${COMMON_ARGS[@]}" --injection dynamic

echo "===== JOB COMPLETED SUCCESSFULLY ====="
date
