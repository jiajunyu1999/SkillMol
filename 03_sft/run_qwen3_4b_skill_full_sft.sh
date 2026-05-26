#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-.}"

BACKBONE="${BACKBONE:-Qwen/Qwen3-4B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-outputs/run_20260509_skillprompt_sft/data/nomoltok_sft_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-outputs/run_20260509_skillprompt_sft/data/nomoltok_sft_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3-4b_skill_lora}"

FINETUNE_MODE="${FINETUNE_MODE:-lora}"
NUM_GPUS="${NUM_GPUS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"

EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
SEARCH_WIDTH="${SEARCH_WIDTH:-8}"
SEARCH_DEPTH="${SEARCH_DEPTH:-1}"
EXTRA_PROMPT_PATH="${EXTRA_PROMPT_PATH:-skill.md}"

python 03_sft/run_sft_train_eval.py \
  --backbone "$BACKBONE" \
  --train_jsonl "$TRAIN_JSONL" \
  --val_jsonl "$VAL_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --finetune_mode "$FINETUNE_MODE" \
  --num_gpus "$NUM_GPUS" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --max_length "$MAX_LENGTH" \
  --batch_size "$BATCH_SIZE" \
  --micro_batch_size "$MICRO_BATCH_SIZE" \
  --eval_max_rows "$EVAL_MAX_ROWS" \
  --n_samples "$N_SAMPLES" \
  --search_width "$SEARCH_WIDTH" \
  --search_depth "$SEARCH_DEPTH" \
  --extra_prompt_path "$EXTRA_PROMPT_PATH" \
  "$@"
