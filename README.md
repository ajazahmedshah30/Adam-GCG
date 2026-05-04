# Adam-GCG Universal Prompt Injection

This repository is an RnD adaptation of the paper [Automatic and Universal Prompt Injection Attacks against Large Language Models](https://arxiv.org/abs/2403.04957). The original public code uses a momentum-style gradient aggregation inside the universal prompt optimization loop. This project keeps the data pipeline, target construction, prompt formatting, candidate sampling, and evaluation flow unchanged, and replaces that optimizer step with an Adam-style update.

The result is an `Adam+GCG` variant that is suitable for reporting as:

- the same universal prompt injection framework as the paper
- the same greedy coordinate gradient (GCG) candidate search
- a different optimizer state update: Adam instead of momentum

<img src="Universal-Prompt-Injection.png" width="1000"/>

## What Changed

Compared with the original implementation:

- `universal_prompt_injection.py` now supports `--optimizer adam` and `--optimizer momentum`
- the default optimizer is `adam`
- Adam uses first and second moments over the aggregated coordinate gradients
- result folders are optimizer-aware, for example:
  - `results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/...`
  - `results/eval/llama2/semi-dynamic/momentum_1.0/...`
- `check_answers.py` is now configurable from the command line for Adam runs

## Project Layout

```text
Adam-GCG-Universal-Prompt-Injection/
|- data/
|- models/
|- utils/
|- universal_prompt_injection.py
|- get_responses_universal.py
|- check_answers.py
`- README.md
```

## Environment Setup

```bash
cd /Users/mohdhussain/Desktop/Adam_GCG/Adam-GCG-Universal-Prompt-Injection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer conda:

```bash
conda create -n adam-gcg python=3.9 -y
conda activate adam-gcg
pip install -r requirements.txt
```

## Download Models

Modify `models/download_models.py` if you want to use different Hugging Face checkpoints, then run:

```bash
cd models
python download_models.py
cd ..
```

## Run Adam+GCG

Default run with Adam:

```bash
python universal_prompt_injection.py \
  --optimizer adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_eps 1e-8 \
  --model llama2 \
  --injection semi-dynamic \
  --target 0 \
  --tokens 150 \
  --start 0 \
  --end 5 \
  --batch_size 256 \
  --topk 128 \
  --num_steps 1000
```

The training log is saved under:

```text
results/eval/<model>/<injection>/adam_b1_<beta1>_b2_<beta2>_eps_<eps>/token_length_<tokens>/target_<target>/
```

Example:

```text
results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
```

## Run Original Baseline

To reproduce the original behavior in the same codebase:

```bash
python universal_prompt_injection.py \
  --optimizer momentum \
  --momentum 1.0 \
  --model llama2 \
  --injection semi-dynamic
```

This makes comparison between `M-GCG` and `Adam+GCG` straightforward for your report.

## Generate Responses

Use the produced training log path when generating responses for downstream tasks:

```bash
python get_responses_universal.py --evaluate duplicate_sentence_detection --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
python get_responses_universal.py --evaluate summarization --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
python get_responses_universal.py --evaluate grammar_correction --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
python get_responses_universal.py --evaluate natural_language_inference --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
python get_responses_universal.py --evaluate hate_detection --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
python get_responses_universal.py --evaluate sentiment_analysis --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
python get_responses_universal.py --evaluate spam_detection --path results/eval/llama2/semi-dynamic/adam_b1_0.9_b2_0.999_eps_1e-08/token_length_150/target_0/0_5_20_normal.json
```

`get_responses_universal.py` now auto-detects the injection type from the result path. You can still pass `--injection` manually if you want to override it.

## Compute Keyword ASR

```bash
python check_answers.py \
  --root ./results/eval/llama2 \
  --optimizer_dir adam_b1_0.9_b2_0.999_eps_1e-08 \
  --token_length 150 \
  --target_number 1 \
  --training_log 0_5_20_normal.json \
  --output_csv ./results_adam.csv
```

## Run Readiness Notes

- `requirements.txt` includes the libraries used by the scripts, including `pandas`
- `models/download_models.py` downloads Llama-2, which requires Hugging Face access approval and login
- full end-to-end execution also requires a CUDA-capable GPU with enough memory for the selected model

## Suggested Report Framing

For your RnD report or presentation, describe the contribution as:

1. Reimplementation of the paper's universal prompt injection optimization pipeline.
2. Controlled optimizer substitution from momentum aggregation to Adam moment estimation.
3. Empirical comparison under identical datasets, targets, token budget, and candidate search settings.

Recommended ablations:

1. `momentum=1.0` vs `adam(beta1=0.9, beta2=0.999)`
2. different token budgets such as `100`, `150`, and `200`
3. different injection modes: `static`, `semi-dynamic`, `dynamic`, `refuse`
4. multiple target models if GPU resources allow

## Acknowledgement

This codebase is derived from the original `Universal-Prompt-Injection` implementation and builds on ideas from [llm-attacks](https://github.com/llm-attacks/llm-attacks).

## BibTeX

```bibtex
@misc{liu2024automatic,
  title={Automatic and Universal Prompt Injection Attacks against Large Language Models},
  author={Xiaogeng Liu and Zhiyuan Yu and Yizhe Zhang and Ning Zhang and Chaowei Xiao},
  year={2024},
  eprint={2403.04957},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
