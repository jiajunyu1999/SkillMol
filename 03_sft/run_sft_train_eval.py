from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BACKBONE = "Qwen/Qwen3-0.6B"
DEFAULT_TOKENIZER_CKPT = "outputs/run_20260412_workflow1/tokenizer/tokenizer.pt"
DEFAULT_TEST_CSV = "data/test_chatdrug.csv"


def _str2bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {raw}")


def _load_json(path: str) -> dict[str, Any]:
    if not str(path).strip():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default_sft_config(mode: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "register_domain_vocab": True,
        "masked_ce": False,
        "codebook_size": 256,
        "num_codebooks": 8,
        "mol_token_format": "shared",
        "max_atom_map": 256,
        "max_fg_id": 64,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "max_steps": -1,
        "weight_decay": 0.0,
        "logging_steps": 10,
        "eval_steps": 999999,
        "save_steps": 100,
        "save_total_limit": 3,
        "seed": 42,
        "gradient_checkpointing": True,
    }
    if mode == "lora":
        return {
            **common,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_train_embeddings": False,
            "max_length": 512,
            "batch_size": 128,
            "micro_batch_size": 16,
            "save_safetensors": False,
        }
    if mode == "full":
        return {
            **common,
            "use_lora": False,
            "max_length": 4096,
            "batch_size": 64,
            "micro_batch_size": 1,
            "save_safetensors": True,
        }
    raise ValueError(f"Unsupported finetune mode: {mode}")


def _update_if_not_none(config: dict[str, Any], **items: Any) -> None:
    for key, value in items.items():
        if value is not None:
            config[key] = value


def _run(cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {' '.join(cmd)}", flush=True)
    print(f"[log] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _build_train_cmd(args: argparse.Namespace, config_path: Path) -> list[str]:
    train_cmd = [
        "03_sft/train_sft.py",
        "--model_name",
        str(args.backbone),
        "--train_jsonl",
        str(args.train_jsonl),
        "--val_jsonl",
        str(args.val_jsonl),
        "--output_dir",
        str(args.output_dir),
        "--config_json",
        str(config_path),
    ]
    if int(args.num_gpus) > 1:
        return [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={int(args.num_gpus)}",
            *train_cmd,
        ]
    return [sys.executable, *train_cmd]


def _build_eval_cmd(args: argparse.Namespace) -> list[str]:
    output_dir = Path(args.output_dir)
    eval_model_path = str(output_dir if args.finetune_mode == "full" else args.backbone)
    adapter_dir = "" if args.finetune_mode == "full" else str(output_dir)

    cmd = [
        sys.executable,
        "05_infer_rerank/eval_chatdrug_14tasks_unified.py",
        "--model_path",
        eval_model_path,
        "--adapter_dir",
        adapter_dir,
        "--tokenizer_ckpt",
        str(args.tokenizer_ckpt),
        "--test_csv",
        str(args.test_csv),
        "--output_dir",
        str(args.eval_output_dir),
        "--max_rows",
        str(int(args.eval_max_rows)),
        "--n_samples",
        str(int(args.n_samples)),
        "--search_width",
        str(int(args.search_width)),
        "--search_depth",
        str(int(args.search_depth)),
        "--device",
        str(args.eval_device),
        "--temperature",
        str(float(args.temperature)),
        "--top_p",
        str(float(args.top_p)),
        "--max_new_tokens",
        str(int(args.max_new_tokens)),
    ]
    if str(args.task_ids).strip():
        cmd.extend(["--task_ids", str(args.task_ids)])
    if not bool(args.eval_include_mol_tokens):
        cmd.append("--no_mol_tokens")
    if not bool(args.eval_include_atom_map_tokens):
        cmd.append("--no_atom_map_tokens")
    if bool(args.no_register_domain_vocab):
        cmd.append("--no_register_domain_vocab")
    if str(args.extra_prompt_path).strip():
        cmd.extend(["--extra_prompt_path", str(args.extra_prompt_path)])
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train OneRec-Mol SFT from a backbone and immediately run the unified "
            "14-task validation/evaluation."
        )
    )
    ap.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE, help="Backbone/full model path.")
    ap.add_argument("--train_jsonl", type=str, required=True, help="SFT training jsonl.")
    ap.add_argument("--val_jsonl", type=str, required=True, help="SFT validation jsonl.")
    ap.add_argument("--output_dir", type=str, required=True, help="Directory for the trained SFT weights.")
    ap.add_argument("--finetune_mode", type=str, choices=["lora", "full"], required=True)
    ap.add_argument("--num_gpus", type=int, default=1, help="Use torchrun when >1.")
    ap.add_argument("--base_config_json", type=str, default="", help="Optional config JSON to override defaults.")

    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--num_train_epochs", type=float, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--micro_batch_size", type=int, default=None)
    ap.add_argument("--save_steps", type=int, default=None)
    ap.add_argument("--save_total_limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--masked_ce", type=_str2bool, default=None)
    ap.add_argument("--register_domain_vocab", type=_str2bool, default=None)
    ap.add_argument("--save_safetensors", type=_str2bool, default=None)
    ap.add_argument("--lora_r", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--lora_dropout", type=float, default=None)
    ap.add_argument("--lora_train_embeddings", type=_str2bool, default=None)
    ap.add_argument("--resume_adapter_path", type=str, default="", help="Optional existing LoRA adapter to continue SFT from.")

    ap.add_argument("--skip_eval", action="store_true", help="Only train; do not run 14-task evaluation.")
    ap.add_argument("--eval_output_dir", type=str, default="", help="Default: <output_dir>/eval_14tasks.")
    ap.add_argument("--test_csv", type=str, default=DEFAULT_TEST_CSV)
    ap.add_argument("--tokenizer_ckpt", type=str, default=DEFAULT_TOKENIZER_CKPT)
    ap.add_argument("--eval_device", type=str, default="cuda:0")
    ap.add_argument("--eval_max_rows", type=int, default=200)
    ap.add_argument("--task_ids", type=str, default="")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--search_width", type=int, default=8)
    ap.add_argument("--search_depth", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--extra_prompt_path", type=str, default="skill.md")
    ap.add_argument("--eval_include_mol_tokens", type=_str2bool, default=False)
    ap.add_argument("--eval_include_atom_map_tokens", type=_str2bool, default=True)
    ap.add_argument("--no_register_domain_vocab", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    if not str(args.eval_output_dir).strip():
        args.eval_output_dir = str(output_dir / "eval_14tasks")
    else:
        eval_output_dir = Path(args.eval_output_dir)
        args.eval_output_dir = str(eval_output_dir if eval_output_dir.is_absolute() else repo_root / eval_output_dir)

    config = _default_sft_config(str(args.finetune_mode))
    config.update(_load_json(str(args.base_config_json)))
    config["use_lora"] = args.finetune_mode == "lora"
    _update_if_not_none(
        config,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        masked_ce=args.masked_ce,
        register_domain_vocab=args.register_domain_vocab,
        save_safetensors=args.save_safetensors,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_train_embeddings=args.lora_train_embeddings,
        resume_adapter_path=str(args.resume_adapter_path).strip() or None,
    )

    config_path = output_dir / "sft_config.effective.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(repo_root)

    _run(_build_train_cmd(args, config_path), cwd=repo_root, log_path=output_dir / "train.wrapper.log", env=env)
    if args.skip_eval:
        print(json.dumps({"trained_model_dir": str(output_dir), "config": str(config_path)}, indent=2))
        return

    eval_output_dir = Path(args.eval_output_dir)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    _run(_build_eval_cmd(args), cwd=repo_root, log_path=eval_output_dir / "eval.wrapper.log", env=env)

    meta_path = eval_output_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(json.dumps({"trained_model_dir": str(output_dir), "eval_meta": str(meta_path), "overall": meta.get("overall")}, indent=2))
    else:
        print(json.dumps({"trained_model_dir": str(output_dir), "eval_output_dir": str(eval_output_dir)}, indent=2))


if __name__ == "__main__":
    main()
