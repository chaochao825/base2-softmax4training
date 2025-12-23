import os
import sys
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

wandb = None
try:
    import wandb as _wandb
    wandb = _wandb
except Exception:
    pass

# Ensure project root is on sys.path when running from anywhere
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.cifar import build_cifar_loaders  # noqa: E402
from src.models.bitnet_resnet18 import BitNetResNet18  # noqa: E402
from src.models.bitnet_vit import BitNetViT  # noqa: E402
from src.utils.plots import plot_training_curves  # noqa: E402


def setup_distributed() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])  # noqa: F841
        world_size = int(os.environ["WORLD_SIZE"])  # noqa: F841
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend=backend, init_method="env://")


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        outputs = torch.softmax(logits, dim=-1)
        loss = criterion(logits, targets)
        total_loss += float(loss.detach().cpu())
        pred = outputs.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.size(0))
    return {"loss": total_loss / max(1, len(loader)), "acc": correct / max(1, total)}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR-10")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vit-s"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--softmax", type=str, default="standard", choices=["standard", "base2"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="1.58bit")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/results")
    args = parser.parse_args()

    setup_distributed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_classes = build_cifar_loaders(args.dataset, args.batch_size)

    if args.model == "resnet18":
        model = BitNetResNet18(num_classes=num_classes, softmax_type=args.softmax, temperature=args.temperature)
    else:
        # ViT-Small-ish config for CIFAR (patch 4 to keep sequence reasonable)
        model = BitNetViT(img_size=32, patch_size=4, num_classes=num_classes, embed_dim=384, depth=6, num_heads=6, softmax_type=args.softmax, temperature=args.temperature)
    model = model.to(device)

    if dist.is_initialized():
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        else:
            model = DDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.wandb and is_main_process() and wandb is not None:
        wandb.init(project="bitnet-base2-softmax-comparison", config={
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "softmax_type": args.softmax,
            "temperature": args.temperature,
            "precision": args.precision,
        })

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, args.epochs + 1):
        # Update DistributedSampler each epoch for proper shuffling
        if hasattr(train_loader, 'sampler') and isinstance(getattr(train_loader, 'sampler'), torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, criterion, device)

        if dist.is_initialized():
            # Reduce validation metrics across ranks
            t = torch.tensor([train_loss, metrics["loss"], metrics["acc"]], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t /= dist.get_world_size()
            train_loss, val_loss, val_acc = map(float, t.tolist())
        else:
            val_loss, val_acc = metrics["loss"], metrics["acc"]

        if is_main_process():
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            if args.wandb and wandb is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                })

            os.makedirs(args.output_dir, exist_ok=True)
            ds_name = args.dataset.replace('-', '').lower()
            plot_path = os.path.join(
                args.output_dir,
                f"curves_{ds_name}_{args.model}_{args.softmax}_temp{str(args.temperature).replace('.', '')}.png",
            )
            plot_training_curves(history, plot_path)

    if is_main_process() and args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()


