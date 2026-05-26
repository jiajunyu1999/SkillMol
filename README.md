# SkillMol

Clean code release for SkillMol.

This repository only keeps the code, the skill prompt, and the test set needed to run molecular editing evaluation. It does not include model weights, checkpoints, training outputs, paper files, result tables, or large generated data.

## Contents

- `onerec_mol/`: core code for prompts, molecular edits, rewards, SFT, GRPO/PPO, and inference.
- `01_tokenizer/`, `02_dataset/`, `03_sft/`, `04_grpo/`, `05_infer_rerank/`: command-line entrypoints.
- `configs/`: minimal representative SFT/GRPO configs.
- `data/test_chatdrug.csv`: test set.
- `data/fg_list_small.json`: functional group vocabulary used by edit decoding.
- `skill.md`: task-specific molecular editing skills.

## Install

```bash
pip install -r requirements.txt
conda install -c conda-forge rdkit
```

Use a PyTorch build matching your CUDA version for GPU training or inference.

## Example Evaluation

```bash
PYTHONPATH=. python 05_infer_rerank/eval_chatdrug_14tasks_unified.py \
  --model_path Qwen/Qwen3-0.6B \
  --adapter_dir path/to/adapter \
  --test_csv data/test_chatdrug.csv \
  --output_dir outputs/eval_14tasks \
  --max_rows 200 \
  --n_samples 8 \
  --search_width 8 \
  --search_depth 1 \
  --extra_prompt_path skill.md \
  --no_mol_tokens
```

## Example SFT

```bash
PYTHONPATH=. python 03_sft/train_sft.py \
  --model_name Qwen/Qwen3-0.6B \
  --train_jsonl path/to/train.jsonl \
  --val_jsonl path/to/val.jsonl \
  --output_dir outputs/skillmol_sft \
  --config_json configs/sft_clean_full_nomoltok_skillprompt_3epoch_live.json
```
