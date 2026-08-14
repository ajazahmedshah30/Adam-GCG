#!/bin/bash
#SBATCH --job-name=eval-mgcg-a40
#SBATCH --account=25m0842
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --output=eval_momentum_a40.%j.out
#SBATCH --error=eval_momentum_a40.%j.err

set -euo pipefail

echo "===== JOB START (A40 - EVAL MOMENTUM) ====="
date
hostname

PROJECT_DIR="$HOME/adam-gcg/Adam-GCG-Universal-Prompt-Injection"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs_eval_a40"

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
mkdir -p "$LOG_DIR"

echo "Python:"
python -c "import sys; print(sys.executable)"

echo "===== GPU CHECK ====="
nvidia-smi || { echo "GPU not available"; exit 1; }

TASKS=(
  duplicate_sentence_detection
  summarization
  grammar_correction
  natural_language_inference
)

OPTIMIZER_DIR="momentum_0.9"
TRAINING_LOG="0_5_20_a40_momentum.json"
SAVE_DIR="a40_momentum"

run_eval () {
    ATTACK="$1"
    TRAIN_LOG_PATH="$PROJECT_DIR/results/eval/llama2/${ATTACK}/${OPTIMIZER_DIR}/token_length_120/target_0/${SAVE_DIR}/${TRAINING_LOG}"

    if [ ! -f "$TRAIN_LOG_PATH" ]; then
        echo "ERROR: Training log not found: $TRAIN_LOG_PATH"
        exit 1
    fi

    for TASK in "${TASKS[@]}"; do
        RESULT_JSON="${TRAIN_LOG_PATH%.json}_${TASK}.json"
        
        echo "===== CHECKING ${ATTACK} :: ${TASK} ====="
        # Quick check: if the result file exists, we let the python script handle resuming or skipping.
        # This allows us to load the model only if work remains.
        
        echo "===== STARTING ${ATTACK} :: ${TASK} ====="
        date

        LOG_FILE="${LOG_DIR}/eval_momentum_${ATTACK}_${TASK}_${SLURM_JOB_ID}.log"
        if ! python get_responses_universal.py \
            --device 0 \
            --evaluate "$TASK" \
            --path "$TRAIN_LOG_PATH" \
            --max_samples 100 \
            --max_attempts 5 > "$LOG_FILE" 2>&1; then
            echo "FAILED ${ATTACK} :: ${TASK}"
            tail -n 20 "$LOG_FILE"
            exit 1
        fi

        echo "SUCCESS ${ATTACK} :: ${TASK}"
    done
}

run_eval static
run_eval semi-dynamic
run_eval dynamic

echo "===== JOB COMPLETED ====="
date
