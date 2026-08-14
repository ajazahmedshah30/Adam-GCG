#!/bin/bash
#SBATCH --job-name=adam-semi-paper
#SBATCH --account=25m0842
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --output=adam_semi_paper.%j.out
#SBATCH --error=adam_semi_paper.%j.err

set -euo pipefail

PROJECT_DIR="$HOME/adam-gcg/Adam-GCG-Universal-Prompt-Injection"
VENV_DIR="$PROJECT_DIR/.venv"
LOG_DIR="$PROJECT_DIR/logs_paper_style"
OUTPUT_ROOT="$PROJECT_DIR/results_paper_style"

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

# Export for protobuf issues and OOM prevention
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "===== STARTING OPTIMIZATION (ADAM SEMI-DYNAMIC PAPER STYLE) ====="
# Reduced batch_size to 256 to fit on A40, keeping other paper hyperparameters
python universal_prompt_injection.py \
    --optimizer adam \
    --injection semi-dynamic \
    --tokens 150 \
    --batch_size 256 \
    --topk 256 \
    --num_steps 1000 \
    --device 0 \
    --start 0 \
    --end 5 \
    --save_suffix "paper_style"

echo "===== OPTIMIZATION COMPLETED ====="

# Find the generated JSON
TRAIN_LOG=$(find results/eval/llama2/semi-dynamic -name "*_paper_style.json" | head -n 1)

if [ -z "$TRAIN_LOG" ]; then
    echo "ERROR: Training log not found"
    exit 1
fi

echo "===== STARTING EVALUATION ====="

TASKS=(
  duplicate_sentence_detection
  summarization
  grammar_correction
  natural_language_inference
  hate_detection
  sentiment_analysis
  spam_detection
)

for TASK in "${TASKS[@]}"; do
    echo "Evaluating $TASK..."
    python get_responses_universal.py \
        --device 0 \
        --evaluate "$TASK" \
        --path "$TRAIN_LOG" \
        --max_samples 200 \
        --max_attempts 10 \
        --output_root "$OUTPUT_ROOT" > "${LOG_DIR}/eval_${TASK}.log" 2>&1
done

echo "===== ALL DONE ====="
python check_answers.py --root "$OUTPUT_ROOT" --token_length 150 --output_csv results_paper_style_asr.csv
