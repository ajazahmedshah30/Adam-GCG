#!/bin/bash
#SBATCH --job-name=eval-adam-a40-full
#SBATCH --account=25m0842
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4-00:00:00
#SBATCH --output=eval_adam_a40_full.%j.out
#SBATCH --error=eval_adam_a40_full.%j.err

set -euo pipefail

echo "===== JOB START (A40 - FULL EVAL ADAM) ====="
date
hostname

PROJECT_DIR="$HOME/adam-gcg/Adam-GCG-Universal-Prompt-Injection"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs_eval_full_a40_corrected"
OUTPUT_ROOT="$PROJECT_DIR/results_full_corrected/eval/llama2"

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
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

echo "Python:"
python -c "import sys; print(sys.executable)"

echo "===== GPU CHECK ====="
nvidia-smi || { echo "GPU not available"; exit 1; }

TASKS=(
  duplicate_sentence_detection
  summarization
  grammar_correction
  natural_language_inference
  hate_detection
  sentiment_analysis
  spam_detection
)

OPTIMIZER_DIR="adam_b1_0.9_b2_0.999_eps_1e-08"
TRAINING_LOG="0_5_20_a40_adam.json"
SAVE_DIR="a40_adam"

run_eval () {
    ATTACK="$1"
    TRAIN_LOG_PATH="$PROJECT_DIR/results/eval/llama2/${ATTACK}/${OPTIMIZER_DIR}/token_length_120/target_0/${SAVE_DIR}/${TRAINING_LOG}"

    if [ ! -f "$TRAIN_LOG_PATH" ]; then
        echo "ERROR: Training log not found: $TRAIN_LOG_PATH"
        exit 1
    fi

    for TASK in "${TASKS[@]}"; do
        echo "===== STARTING ${ATTACK} :: ${TASK} ====="
        date

        LOG_FILE="${LOG_DIR}/eval_adam_full_${ATTACK}_${TASK}_${SLURM_JOB_ID}.log"
        if ! python get_responses_universal.py \
            --device 0 \
            --evaluate "$TASK" \
            --path "$TRAIN_LOG_PATH" \
            --max_samples 200 \
            --max_attempts 10 \
            --output_root "$OUTPUT_ROOT" > "$LOG_FILE" 2>&1; then
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
