#!/usr/bin/env python3
"""
Evaluate standard vs base-2 softmax on HF LLMs (e.g., Qwen/LLaMA) under low-bit configs.
Produces JSON + PNG outputs without overwriting previous results (timestamped directory).
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # bitsandbytes optional
    BitsAndBytesConfig = None  # type: ignore

from src.ops.base2_softmax import patch_torch_softmax_base2


def build_bnb_config(quant_mode: str):
    if BitsAndBytesConfig is None:
        return None
    quant_mode = quant_mode.lower()
    if quant_mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    if quant_mode in {"4bit", "fp4"}:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return None


def load_model_and_tokenizer(model_name: str, quant_mode: str, device: torch.device, attn_impl: str = 'eager'):
    # Try local files first, fallback to online if not found
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    except OSError:
        # If local files not found, try online (may fail if offline)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qmode = quant_mode.lower()
    bnb_config = build_bnb_config(quant_mode)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    fp8_warn = False
    if qmode == "fp8":
        float8_dtype = getattr(torch, "float8_e4m3fn", None)
        if float8_dtype is not None:
            dtype = float8_dtype
        else:
            fp8_warn = True
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif qmode == "fp32":
        dtype = torch.float32

    if fp8_warn:
        print("   FP8 not supported in this build; falling back to bf16.")

    kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto" if device.type == "cuda" else None,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config

    # Try local files first, fallback to online if not found
    try:
        kwargs["local_files_only"] = True
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except OSError:
        # If local files not found, try online (may fail if offline)
        kwargs.pop("local_files_only", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device.type == "cuda" and bnb_config is None:
        model = model.to(device)

    # Map friendly names
    impl = attn_impl
    if attn_impl == "torch":
        impl = "eager"

    # Force attention impl so base-2 patch applies (avoid fused flash kernels).
    for attr in ["attn_implementation", "_attn_implementation"]:
        if hasattr(model.config, attr):
            setattr(model.config, attr, impl)
    if hasattr(model.config, "use_flash_attn") and impl != "flash_attention_2":
        model.config.use_flash_attn = False
    if impl != "flash_attention_2":
        import os
        os.environ["FLASH_ATTENTION_DISABLE"] = "1"

    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, device: torch.device, max_length: int, max_samples: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for idx, sample in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        text = sample.get("text") or sample.get("content") or ""
        if not text:
            continue
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

        # Only count valid tokens (exclude padding tokens)
        # HuggingFace's loss already accounts for attention_mask, so loss is per valid token
        valid_token_count = inputs["attention_mask"].sum().item()
        total_loss += loss.item() * valid_token_count
        total_tokens += valid_token_count

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 50))
    return {"loss": avg_loss, "ppl": ppl, "tokens": total_tokens}


def plot_results(results: List[Dict], out_path: Path, dataset: str, max_length: int, samples: int):
    sns.set_theme(style="whitegrid", palette="deep")
    groups = {}
    for r in results:
        key = f"{r['model']}\n{r['quant_mode']}"
        groups.setdefault(key, {})[r["softmax"]] = r

    labels = list(groups.keys())
    std_vals, base2_vals = [], []
    for key in labels:
        std_vals.append(groups[key].get("standard", {}).get("ppl", None))
        base2_vals.append(groups[key].get("base2", {}).get("ppl", None))

    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, std_vals, width, label="Softmax (base-e)", color="#3498db", alpha=0.85)
    ax.bar([i + width for i in x], base2_vals, width, label="Softmax (base-2)", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(f"LLM perplexity | dataset={dataset} | max_length={max_length} | samples={samples}")
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()

    for i, (s, b) in enumerate(zip(std_vals, base2_vals)):
        if s is not None:
            ax.text(i, s, f"{s:.2f}", ha="center", va="bottom", fontsize=8)
        if b is not None:
            ax.text(i + width, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Base-2 vs standard softmax eval on Qwen/LLaMA")
    parser.add_argument("--models", nargs="+", default=[
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-8B-Instruct",
        "meta-llama/Llama-3.2-3B",
    ])
    parser.add_argument("--quant_modes", nargs="+", default=["fp16", "fp8", "int8", "4bit"], help="fp16|fp8|int8|4bit")
    parser.add_argument("--attn_impl", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"], help="Force attention implementation so base-2 softmax is actually applied (use eager/sdpa to avoid fused kernels)")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--dataset_dir", type=str, default="/home/spco/base-2-bitnet/data/wikitext-2-raw-v1", help="Local dir with train/validation/test txt to avoid remote downloads")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/new_result/llm_base2_eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_dir and Path(args.dataset_dir).exists():
        data_files = {
            'train': str(Path(args.dataset_dir) / 'train.txt'),
            'validation': str(Path(args.dataset_dir) / 'validation.txt'),
            'test': str(Path(args.dataset_dir) / 'test.txt'),
        }
        dataset = load_dataset('text', data_files=data_files, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)

    if args.samples:
        dataset = dataset.select(range(min(args.samples, len(dataset))))

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []

    for model_name in args.models:
        for quant_mode in args.quant_modes:
            warn_quant = quant_mode.lower() in {"int8", "4bit", "fp4"} and BitsAndBytesConfig is None
            for softmax in ("standard", "base2"):
                print(f"\n==> {model_name} | {quant_mode} | softmax={softmax}")
                if warn_quant:
                    print("   bitsandbytes not available; running in full precision instead.")
                model, tokenizer = load_model_and_tokenizer(model_name, quant_mode, device, attn_impl=args.attn_impl)

                context = patch_torch_softmax_base2(args.temperature) if softmax == "base2" else nullcontext()
                with context:
                    metrics = compute_perplexity(model, tokenizer, dataset, device, args.max_length, args.samples)

                record = {
                    "model": model_name,
                    "quant_mode": quant_mode,
                    "softmax": softmax,
                    "dataset": args.dataset,
                    "split": args.split,
                    "samples": args.samples,
                    "max_length": args.max_length,
                    "temperature": args.temperature,
                }
                record.update(metrics)
                results.append(record)

                del model
                torch.cuda.empty_cache()

    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {json_path}")

    plot_path = out_dir / "perplexity_comparison.png"
    plot_results(results, plot_path, args.dataset, args.max_length, args.samples)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
