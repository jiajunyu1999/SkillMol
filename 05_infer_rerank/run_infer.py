from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from onerec_mol.inference import infer_with_rerank
from onerec_mol.vocab import register_domain_tokens


@lru_cache(maxsize=4)
def _load_runtime(model_name: str, adapter_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    register_domain_tokens(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token_id is not None else "[PAD]"
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
    if int(base.get_input_embeddings().num_embeddings) != int(len(tokenizer)):
        base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.to(device)
    model.eval()
    return model, tokenizer


def _run_one(
    *,
    data: dict[str, Any],
    model_name: str,
    adapter_dir: str,
    n_samples: int,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, Any]:
    prompt = str(data["prompt"])
    meta = dict(data["meta"])
    model, tokenizer = _load_runtime(str(model_name), str(adapter_dir), str(device))

    return infer_with_rerank(
        model={"model": model, "tokenizer": tokenizer},
        prompt=prompt,
        meta=meta,
        n_samples=int(n_samples),
        gen_config={
            "tokenizer": tokenizer,
            "do_sample": True,
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "constrained_decoding": True,
            "constraint_guidance": True,
            "constraint_sample_multiplier": 2,
            "constraint_max_rounds": 2,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter_dir", type=str, required=True)
    ap.add_argument("--input_json", type=str, default="", help="Single JSON file with prompt/meta.")
    ap.add_argument("--input_jsonl", type=str, default="", help="Batch JSONL input. Reuses one loaded model.")
    ap.add_argument("--output_jsonl", type=str, default="", help="Optional JSONL output path for --input_jsonl.")
    ap.add_argument("--stdin_jsonl", action="store_true", help="Read JSONL requests from stdin and stream JSONL responses.")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    mode_count = int(bool(args.input_json)) + int(bool(args.input_jsonl)) + int(bool(args.stdin_jsonl))
    if mode_count != 1:
        raise ValueError("Exactly one of --input_json, --input_jsonl, or --stdin_jsonl must be specified.")

    common = {
        "model_name": str(args.model_name),
        "adapter_dir": str(args.adapter_dir),
        "n_samples": int(args.n_samples),
        "device": str(args.device),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
    }

    if args.input_json:
        data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        result = _run_one(data=data, **common)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.input_jsonl:
        out_lines: list[str] = []
        with Path(args.input_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                result = _run_one(data=json.loads(line), **common)
                out_lines.append(json.dumps(result, ensure_ascii=False))
        if args.output_jsonl:
            Path(args.output_jsonl).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        else:
            for line in out_lines:
                print(line)
        return

    for line in sys.stdin:
        if not line.strip():
            continue
        result = _run_one(data=json.loads(line), **common)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
