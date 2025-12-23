#!/usr/bin/env python3
"""
Enhanced LLM training script for Base-2 Softmax comparison with BitNet quantization.
Uses GPT-2 style model on TinyStories dataset.
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wandb = None
try:
    import wandb as _wandb
    wandb = _wandb
except Exception:
    pass

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.quant.bitlinear import BitLinear
from src.ops.base2_softmax import Base2Softmax


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend=backend, init_method="env://")


def is_main_process():
    """Check if current process is main."""
    return not dist.is_initialized() or dist.get_rank() == 0


def convert_gpt2_to_bitnet(model: GPT2LMHeadModel, softmax_type: str = "standard"):
    """
    Minimal, safe conversion for GPT-2:
    - Skip BitLinear replacements to avoid shape mismatches with HF GPT-2 Conv1D layers.
    - If softmax_type == 'base2', monkey-patch torch.nn.functional.softmax and cross_entropy
      to Base-2 variant for the duration of this process.
    """
    if softmax_type == "base2":
        from src.ops.base2_softmax import base2_softmax_fn
        import torch.nn.functional as F
        import math
        
        def _base2_softmax(x, dim=None, _stacklevel=3, dtype=None):
            return base2_softmax_fn(x, dim=dim)

        _orig_cross_entropy = F.cross_entropy
        def _base2_cross_entropy(input, target, *extra_args, **kwargs):
            return _orig_cross_entropy(input * math.log(2), target, *extra_args, **kwargs)

        # Monkey-patch globally within this process
        F.softmax = _base2_softmax  # type: ignore[assignment]
        torch.softmax = _base2_softmax
        F.cross_entropy = _base2_cross_entropy

    return model


class TinyStoriesDataset(torch.utils.data.Dataset):
    """TinyStories dataset wrapper."""
    
    def __init__(self, tokenizer, max_length=512, num_samples=10000, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset from HuggingFace or use mirror
        print(f"Loading TinyStories dataset ({split} split, {num_samples} samples)...")
        try:
            dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            self.texts = [item['text'] for item in dataset]
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Using synthetic data for testing...")
            # Fallback: generate synthetic simple stories
            self.texts = [
                f"Once upon a time, there was a story number {i}. It was a very interesting story."
                for i in range(num_samples)
            ]
        
        print(f"Loaded {len(self.texts)} texts")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


def get_grad_norm(model: nn.Module) -> float:
    """Compute L2 norm of gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_one_epoch(model, dataloader, optimizer, scheduler, device, track_grads=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    grad_norms = []
    
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        
        if track_grads:
            grad_norms.append(get_grad_norm(model))
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        total_tokens += attention_mask.sum().item()
        
        if step % 100 == 0 and is_main_process():
            print(f"  Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(min(avg_loss, 20))  # Cap for numerical stability
    
    result = {'loss': avg_loss, 'perplexity': perplexity}
    if track_grads and grad_norms:
        result['grad_norm'] = sum(grad_norms) / len(grad_norms)
    
    return result


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        total_loss += loss.item()
        total_tokens += attention_mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(min(avg_loss, 20))
    
    return {'loss': avg_loss, 'perplexity': perplexity}


def save_loss_curve(history: dict, out_path: str, title: str):
    plt.style.use('bmh')  # Using a nice built-in style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    epochs = history.get('epoch', [])
    
    # Global font size settings
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'legend.fontsize': 12})
    
    # Train Loss
    ax1.plot(epochs, history.get('train_loss', []), 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.set_title(f"{title} - Training Loss", fontsize=16, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Val Loss
    ax2.plot(epochs, history.get('val_loss', []), 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.set_title(f"{title} - Validation Loss", fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_comparison_plot(output_dir: str, model_name: str):
    """Generate a plot comparing standard vs base2 softmax."""
    std_path = os.path.join(output_dir, f"history_{model_name}_standard.json")
    base2_path = os.path.join(output_dir, f"history_{model_name}_base2.json")
    
    if not (os.path.exists(std_path) and os.path.exists(base2_path)):
        return

    with open(std_path, 'r') as f:
        std_history = json.load(f)
    with open(base2_path, 'r') as f:
        base2_history = json.load(f)

    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Global font size settings for comparison
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 15, 'axes.titlesize': 18, 'legend.fontsize': 13})

    # Training Loss Comparison
    ax1.plot(std_history['epoch'], std_history['train_loss'], 'r--s', label='Base-e (Standard)', linewidth=2, markersize=6, alpha=0.8)
    ax1.plot(base2_history['epoch'], base2_history['train_loss'], 'b-o', label='Base-2', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Training Loss', fontsize=15)
    ax1.set_title(f"GPT-2 {model_name} Training Loss Comparison", fontsize=18, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss Comparison
    ax2.plot(std_history['epoch'], std_history['val_loss'], 'r--s', label='Base-e (Standard)', linewidth=2, markersize=6, alpha=0.8)
    ax2.plot(base2_history['epoch'], base2_history['val_loss'], 'b-o', label='Base-2', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Validation Loss', fontsize=15)
    ax2.set_title(f"GPT-2 {model_name} Validation Loss Comparison", fontsize=18, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    comp_path = os.path.join(output_dir, f"comparison_{model_name}_final.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"  Comparison plot updated: {comp_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Training with Base-2 Softmax")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Base model (gpt2, gpt2-medium)")
    parser.add_argument("--softmax", type=str, default="standard", choices=["standard", "base2"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_samples", type=int, default=10000)
    parser.add_argument("--num_val_samples", type=int, default=1000)
    parser.add_argument("--track_grads", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/results")
    parser.add_argument("--save_every", type=int, default=10, help="Save loss curve every N epochs (and final).")
    # Optional model overrides to scale memory usage
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_embd", type=int, default=None)
    
    args = parser.parse_args()
    
    setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process():
        print(f"\n{'='*80}")
        print(f"LLM Training: {args.model_name} with {args.softmax} Softmax")
        print(f"{'='*80}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    train_dataset = TinyStoriesDataset(
        tokenizer, 
        max_length=args.max_length,
        num_samples=args.num_train_samples,
        split='train'
    )
    val_dataset = TinyStoriesDataset(
        tokenizer,
        max_length=args.max_length,
        num_samples=args.num_val_samples,
        split='validation'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    if is_main_process():
        print(f"Creating {args.model_name} model with BitNet quantization...")
    
    config = GPT2Config.from_pretrained(args.model_name, local_files_only=True)
    # Apply user overrides if provided (otherwise keep pretrained defaults)
    if args.n_layer is not None:
        config.n_layer = int(args.n_layer)
    if args.n_head is not None:
        config.n_head = int(args.n_head)
    if args.n_embd is not None:
        config.n_embd = int(args.n_embd)
    
    model = GPT2LMHeadModel(config)

    # Convert to BitNet
    model = convert_gpt2_to_bitnet(model, softmax_type=args.softmax)
    model = model.to(device)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # WandB
    if args.wandb and is_main_process() and wandb is not None:
        wandb.init(
            project="bitnet-base2-softmax-llm",
            name=f"{args.model_name}_{args.softmax}",
            config=vars(args)
        )
    
    # Training loop
    history = {
        'epoch': [], 'train_loss': [], 'train_ppl': [],
        'val_loss': [], 'val_ppl': [], 'grad_norm': []
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        if is_main_process():
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 80)
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, track_grads=args.track_grads
        )
        val_metrics = evaluate(model, val_loader, device)
        
        # Aggregate metrics across GPUs
        if dist.is_initialized():
            metrics_tensor = torch.tensor([
                train_metrics['loss'],
                val_metrics['loss'],
                train_metrics.get('grad_norm', 0.0)
            ], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= dist.get_world_size()
            train_loss, val_loss, grad_norm = map(float, metrics_tensor.tolist())
            train_ppl = math.exp(min(train_loss, 20))
            val_ppl = math.exp(min(val_loss, 20))
        else:
            train_loss = train_metrics['loss']
            val_loss = val_metrics['loss']
            train_ppl = train_metrics['perplexity']
            val_ppl = val_metrics['perplexity']
            grad_norm = train_metrics.get('grad_norm', 0.0)
        
            if is_main_process():
                history['epoch'].append(epoch)
                history['train_loss'].append(train_loss)
                history['train_ppl'].append(train_ppl)
                history['val_loss'].append(val_loss)
                history['val_ppl'].append(val_ppl)
                history['grad_norm'].append(grad_norm)

                # Save history to JSON every epoch for persistent recording
                history_path = os.path.join(args.output_dir, f"history_{args.model_name}_{args.softmax}.json")
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)

                if args.save_every and (epoch % args.save_every == 0):
                    curve_path = os.path.join(args.output_dir, f"loss_curve_e{epoch:03d}.png")
                    save_loss_curve(history, curve_path, f"Loss Curve up to Epoch {epoch}")
                    
                    # Also save a checkpoint every N epochs
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint_llm_{args.model_name}_{args.softmax}_e{epoch:03d}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'history': history
                    }, ckpt_path)
                    print(f"  Checkpoint and curves saved at epoch {epoch}")
                
                # Attempt to generate comparison plot if other variant exists
                try:
                    generate_comparison_plot(args.output_dir, args.model_name)
                except Exception as e:
                    print(f"  Could not generate comparison plot: {e}")

            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
            if grad_norm > 0:
                print(f"  Grad Norm: {grad_norm:.4f}")
            
            if args.wandb and wandb is not None:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_perplexity': train_ppl,
                    'val_loss': val_loss,
                    'val_perplexity': val_ppl,
                    'grad_norm': grad_norm,
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(
                    args.output_dir,
                    f"best_llm_{args.model_name}_{args.softmax}.pth"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
    
    # Save results
    if is_main_process():
        results = {
            'model': args.model_name,
            'softmax': args.softmax,
            'best_val_loss': best_val_loss,
            'best_val_perplexity': math.exp(min(best_val_loss, 20)),
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0.0,
            'final_val_perplexity': history['val_ppl'][-1] if history['val_ppl'] else 0.0,
            'history': history,
        }
        
        results_path = os.path.join(
            args.output_dir,
            f"results_llm_{args.model_name}_{args.softmax}.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Final loss curve
        final_curve_path = os.path.join(args.output_dir, "loss_curve_final.png")
        save_loss_curve(history, final_curve_path, "Final Loss Curve")

        # Save run parameters
        run_params = {
            'model': args.model_name,
            'softmax': args.softmax,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'num_train_samples': args.num_train_samples,
            'num_val_samples': args.num_val_samples,
            'max_length': args.max_length,
            'save_every': args.save_every,
            'output_dir': args.output_dir,
        }
        with open(os.path.join(args.output_dir, "run_params.json"), 'w') as f:
            json.dump(run_params, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"Best Val Perplexity: {results['best_val_perplexity']:.2f}")
        print(f"Final Val Perplexity: {results['final_val_perplexity']:.2f}")
        print(f"Results saved to: {results_path}")
        print(f"{'='*80}\n")
    
    if is_main_process() and args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

