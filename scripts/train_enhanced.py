#!/usr/bin/env python3
"""Enhanced training script with gradient tracking, checkpointing, and advanced features."""

import os
import sys
import math
import json
from typing import Dict, Optional
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import autocast, GradScaler

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

from src.data.cifar import build_cifar_loaders
from src.data.imagenet import build_imagenet_loaders
from src.models.bitnet_resnet18 import BitNetResNet18
from src.models.bitnet_vit import BitNetViT
from src.utils.plots import plot_training_curves, plot_comparison, plot_gradient_norms


def setup_distributed() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend=backend, init_method="env://")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device, amp_context, grad_scaler: GradScaler | None = None, track_grads: bool = False) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    grad_norms = []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with amp_context:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        if grad_scaler is not None and grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            if track_grads:
                grad_scaler.unscale_(optimizer)
                grad_norms.append(get_grad_norm(model))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            if track_grads:
                grad_norms.append(get_grad_norm(model))
            optimizer.step()
        
        total_loss += float(loss.detach().cpu())
    
    result = {"loss": total_loss / max(1, len(loader))}
    if track_grads and grad_norms:
        result["grad_norm"] = sum(grad_norms) / len(grad_norms)
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device, amp_context) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_context:
            logits = model(images)
        outputs = torch.softmax(logits, dim=-1)
        loss = criterion(logits, targets)
        total_loss += float(loss.detach().cpu())
        pred = outputs.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.size(0))
    return {"loss": total_loss / max(1, len(loader)), "acc": correct / max(1, total)}



def build_amp_context(amp_dtype: str, device: torch.device, verbose: bool = False):
    """Return (autocast context, grad scaler, resolved amp tag)."""
    if device.type != "cuda":
        return nullcontext(), GradScaler(enabled=False), "cpu_fp32"

    amp_dtype = amp_dtype.lower()
    if amp_dtype == "fp32":
        return nullcontext(), GradScaler(enabled=False), "fp32"
    if amp_dtype == "bf16":
        return autocast(device_type="cuda", dtype=torch.bfloat16), GradScaler(enabled=False), "bf16"
    if amp_dtype == "fp16":
        return autocast(device_type="cuda", dtype=torch.float16), GradScaler(enabled=True), "fp16"
    if amp_dtype == "fp8":
        float8_dtype = getattr(torch, "float8_e4m3fn", None)
        if float8_dtype is None:
            if verbose:
                print("FP8 not supported on this build; falling back to bf16.")
            return autocast(device_type="cuda", dtype=torch.bfloat16), GradScaler(enabled=False), "bf16_fallback_fp8"
        return autocast(device_type="cuda", dtype=float8_dtype), GradScaler(enabled=True), "fp8"

    if verbose:
        print(f"Unknown amp_dtype={amp_dtype}, using fp32")
    return nullcontext(), GradScaler(enabled=False), "fp32"


def save_checkpoint(model: nn.Module, optimizer, scheduler, epoch: int, best_acc: float, path: str):
    """Save training checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
    }
    torch.save(state, path)


def load_checkpoint(model: nn.Module, optimizer, scheduler, path: str) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_acc']


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced BitNet Training with Base-2 Softmax Comparison")
    # Dataset & Model
    parser.add_argument("--dataset", type=str, default="CIFAR-10", choices=["CIFAR-10", "CIFAR-100", "ImageNet-100", "ImageNet-1K"])
    parser.add_argument("--data_path", type=str, default="/home/spco/data")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vit-s", "vit-b"])
    
    # Training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    
    # Softmax
    parser.add_argument("--softmax", type=str, default="standard", choices=["standard", "base2"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--amp_dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16", "fp8"], help="Autocast compute dtype; avoids overwriting previous runs via suffix")
    parser.add_argument("--precision", type=str, default="1.58bit")
    
    # Monitoring & Saving
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--track_grads", action="store_true", help="Track gradient norms")
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/results")
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()

    setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_context, grad_scaler, amp_tag = build_amp_context(args.amp_dtype, device, verbose=is_main_process())

    # Load data
    if args.dataset.startswith("CIFAR"):
        train_loader, val_loader, num_classes = build_cifar_loaders(args.dataset, args.batch_size)
        img_size = 32
    elif args.dataset.startswith("ImageNet"):
        subset = "100" if args.dataset == "ImageNet-100" else "1k"
        imagenet_path = os.path.join(args.data_path, "imagenet")
        train_loader, val_loader, num_classes = build_imagenet_loaders(
            imagenet_path, args.batch_size, img_size=224, subset=subset
        )
        img_size = 224
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Create model
    if args.model == "resnet18":
        model = BitNetResNet18(num_classes=num_classes, softmax_type=args.softmax, temperature=args.temperature)
    elif args.model == "vit-s":
        patch_size = 4 if img_size == 32 else 16
        model = BitNetViT(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes,
            embed_dim=384, depth=6, num_heads=6,
            softmax_type=args.softmax, temperature=args.temperature
        )
    elif args.model == "vit-b":
        patch_size = 4 if img_size == 32 else 16
        model = BitNetViT(
            img_size=img_size, patch_size=patch_size, num_classes=num_classes,
            embed_dim=768, depth=12, num_heads=12,
            softmax_type=args.softmax, temperature=args.temperature
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Global patch for base-2 if requested
    if args.softmax == "base2":
        from src.ops.base2_softmax import base2_softmax_fn
        import torch.nn.functional as F
        import math
        
        LOG_2 = math.log(2)
        
        def _base2_softmax_wrapper(x, dim=None, _stacklevel=3, dtype=None):
            return base2_softmax_fn(x, dim=dim, temperature=args.temperature)
        
        _orig_cross_entropy = F.cross_entropy
        def _base2_cross_entropy(input, target, *extra_args, **kwargs):
            return _orig_cross_entropy(input * (LOG_2 / args.temperature), target, *extra_args, **kwargs)
            
        F.softmax = _base2_softmax_wrapper
        torch.softmax = _base2_softmax_wrapper
        F.cross_entropy = _base2_cross_entropy
        
        if is_main_process():
            print("Successfully patched F.softmax, torch.softmax, and F.cross_entropy to Base-2 variant globally.")
    
    model = model.to(device)

    if dist.is_initialized():
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        else:
            model = DDP(model)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        if is_main_process():
            print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    # WandB initialization
    if args.wandb and is_main_process() and wandb is not None:
        wandb.init(
            project="bitnet-base2-softmax-comparison",
            name=f"{args.model}_{args.dataset}_{args.softmax}_{amp_tag}_lr{args.lr}",
            config=vars(args)
        )

    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": [],
        "grad_norm": [], "lr": []
    }

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # Update DistributedSampler
        if hasattr(train_loader, 'sampler') and isinstance(getattr(train_loader, 'sampler'), torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        # Warmup
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_context, grad_scaler=grad_scaler, track_grads=args.track_grads)
        val_metrics = evaluate(model, val_loader, criterion, device, amp_context)
        
        # Scheduler step (after warmup)
        if epoch > args.warmup_epochs and scheduler:
            scheduler.step()

        # Aggregate metrics across GPUs
        if dist.is_initialized():
            metrics_tensor = torch.tensor([
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["acc"],
                train_metrics.get("grad_norm", 0.0)
            ], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            metrics_tensor /= dist.get_world_size()
            train_loss, val_loss, val_acc, grad_norm = map(float, metrics_tensor.tolist())
        else:
            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            grad_norm = train_metrics.get("grad_norm", 0.0)

        current_lr = optimizer.param_groups[0]["lr"]

        if is_main_process():
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["grad_norm"].append(grad_norm)
            history["lr"].append(current_lr)

            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            if args.wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "grad_norm": grad_norm,
                    "lr": current_lr,
                })

            # Save checkpoint
            if args.save_checkpoints and val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = os.path.join(args.output_dir, f"best_{args.model}_{args.dataset}_{args.softmax}_{amp_tag}.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, best_acc, ckpt_path)

            # Plot curves
            ds_name = args.dataset.replace('-', '').lower()
            plot_path = os.path.join(
                args.output_dir,
                f"curves_{ds_name}_{args.model}_{args.softmax}_{amp_tag}_temp{str(args.temperature).replace('.', '')}.png"
            )
            plot_training_curves(history, plot_path)

    # Save final results
    if is_main_process():
        results = {
            "dataset": args.dataset,
            "model": args.model,
            "softmax": args.softmax,
            "temperature": args.temperature,
            "amp_dtype": amp_tag,
            "best_val_acc": max(history["val_accuracy"]) if history["val_accuracy"] else 0.0,
            "final_val_acc": history["val_accuracy"][-1] if history["val_accuracy"] else 0.0,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
            "history": history,
        }
        
        results_path = os.path.join(
            args.output_dir,
            f"results_{ds_name}_{args.model}_{args.softmax}_{amp_tag}.json"
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Dataset: {args.dataset} | Model: {args.model} | Softmax: {args.softmax} | AMP: {amp_tag}")
        print(f"Best Val Accuracy: {results['best_val_acc']:.4f}")
        print(f"Final Val Accuracy: {results['final_val_acc']:.4f}")
        print(f"Results saved to: {results_path}")
        print(f"{'='*60}\n")

    if is_main_process() and args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()


