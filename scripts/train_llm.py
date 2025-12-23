#!/usr/bin/env python3
"""Train small LLM with BitNet quantization and configurable Softmax."""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.bitnet_llama import convert_llama_to_bitnet


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    softmax_type: str = field(default="standard", metadata={"help": "standard or base2"})
    temperature: float = field(default=1.0)


@dataclass
class DataArguments:
    dataset_name: str = field(default="roneneldan/TinyStories")
    max_length: int = field(default=512)
    num_train_samples: int = field(default=10000, metadata={"help": "Limit training samples for quick experiments"})


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--softmax", type=str, default="standard", choices=["standard", "base2"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/results/llm")
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"BitNet LLM Training with {args.softmax.upper()} Softmax")
    print(f"{'='*80}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use FP32 for BitNet quantization
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Convert to BitNet
    print(f"Converting to BitNet with {args.softmax} softmax...")
    model = convert_llama_to_bitnet(
        model,
        softmax_type=args.softmax,
        temperature=args.temperature,
        replace_embeddings=False
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    
    # Take limited samples for quick experiment
    dataset = dataset.take(args.num_train_samples)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"] if "text" in dataset.column_names else []
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    output_dir = os.path.join(args.output_dir, f"{args.model_name.split('/')[-1]}_{args.softmax}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        eval_strategy="no",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="wandb" if args.wandb else "none",
        run_name=f"bitnet_{args.softmax}_{args.temperature}" if args.wandb else None,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save results
    results = {
        "model": args.model_name,
        "dataset": args.dataset,
        "softmax": args.softmax,
        "temperature": args.temperature,
        "num_samples": args.num_train_samples,
        "final_loss": trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None,
    }
    
    import json
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()


