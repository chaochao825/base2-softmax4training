#!/usr/bin/env python3
"""
Comprehensive experiment runner for Base-2 Softmax vs Standard Softmax comparison.
Supports multi-GPU training with wandb logging.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Experiment configurations
EXPERIMENTS = [
    # ResNet-18 on CIFAR-10
    {
        "name": "ResNet18-CIFAR10-Standard",
        "dataset": "CIFAR-10",
        "model": "resnet18",
        "softmax": "standard",
        "batch_size": 256,
        "epochs": 1000,
        "lr": 1e-3,
        "scheduler": "cosine",
        "warmup_epochs": 5,
    },
    {
        "name": "ResNet18-CIFAR10-Base2",
        "dataset": "CIFAR-10",
        "model": "resnet18",
        "softmax": "base2",
        "batch_size": 256,
        "epochs": 1000,
        "lr": 1e-3,
        "scheduler": "cosine",
        "warmup_epochs": 5,
    },
    # ViT-Small on CIFAR-10
    {
        "name": "ViT-S-CIFAR10-Standard",
        "dataset": "CIFAR-10",
        "model": "vit-s",
        "softmax": "standard",
        "batch_size": 256,
        "epochs": 1000,
        "lr": 5e-4,
        "scheduler": "cosine",
        "warmup_epochs": 10,
    },
    {
        "name": "ViT-S-CIFAR10-Base2",
        "dataset": "CIFAR-10",
        "model": "vit-s",
        "softmax": "base2",
        "batch_size": 256,
        "epochs": 1000,
        "lr": 5e-4,
        "scheduler": "cosine",
        "warmup_epochs": 10,
    },
]


def run_experiment(exp_config, args):
    """Run a single experiment using multi-GPU training."""
    # Check if result file already exists
    ds_name = exp_config['dataset'].replace('-', '').lower()
    result_filename = f"results_{ds_name}_{exp_config['model']}_{exp_config['softmax']}_fp32.json"
    result_path = os.path.join(args.output_dir, result_filename)
    
    if os.path.exists(result_path):
        print(f"\nSkipping {exp_config['name']} - results already exist at {result_path}")
        return True

    print(f"\n{'='*80}")
    print(f"Running: {exp_config['name']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.num_gpus}",
        "--master_port=29500",
        f"{args.script_path}",
        f"--dataset={exp_config['dataset']}",
        f"--model={exp_config['model']}",
        f"--softmax={exp_config['softmax']}",
        f"--batch_size={exp_config['batch_size']}",
        f"--epochs={exp_config['epochs']}",
        f"--lr={exp_config['lr']}",
        f"--scheduler={exp_config['scheduler']}",
        f"--warmup_epochs={exp_config['warmup_epochs']}",
        f"--data_path={args.data_path}",
        f"--output_dir={args.output_dir}",
        "--track_grads",
        "--save_checkpoints",
    ]
    
    if args.use_wandb:
        cmd.append("--wandb")
    
    # Set CUDA_VISIBLE_DEVICES to use specified GPUs
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, env=env, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ {exp_config['name']} completed in {elapsed/60:.2f} minutes\n")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {exp_config['name']} failed after {elapsed/60:.2f} minutes")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive Base-2 Softmax experiments")
    parser.add_argument("--num_gpus", type=int, default=3, help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2", help="GPU IDs to use")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Use wandb logging")
    parser.add_argument("--data_path", type=str, default="/home/spco/data")
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/results")
    parser.add_argument("--script_path", type=str, default="/home/spco/base-2-bitnet/scripts/train_enhanced.py")
    parser.add_argument("--experiments", type=str, nargs="+", default=None, 
                        help="Specific experiments to run (by index: 0-3)")
    
    args = parser.parse_args()
    
    print(f"DEBUG: args.data_path = {args.data_path}")
    print(f"DEBUG: args.output_dir = {args.output_dir}")
    
    # Determine which experiments to run
    if args.experiments:
        exp_indices = [int(i) for i in args.experiments]
        experiments_to_run = [EXPERIMENTS[i] for i in exp_indices if i < len(EXPERIMENTS)]
    else:
        experiments_to_run = EXPERIMENTS
    
    print(f"\n{'='*80}")
    print(f"Comprehensive Base-2 Softmax Experiments")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments_to_run)}")
    print(f"GPUs: {args.gpu_ids} ({args.num_gpus} devices)")
    print(f"WandB logging: {args.use_wandb}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments sequentially
    results = {}
    for i, exp_config in enumerate(experiments_to_run):
        success = run_experiment(exp_config, args)
        results[exp_config['name']] = success
        
        # Short pause between experiments
        if i < len(experiments_to_run) - 1:
            print(f"\nWaiting 10 seconds before next experiment...\n")
            time.sleep(10)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}\n")
    for name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")
    print(f"\n{'='*80}\n")
    
    # Count successes
    total = len(results)
    successes = sum(1 for s in results.values() if s)
    print(f"Completed: {successes}/{total} experiments")
    
    if successes == total:
        print("\n🎉 All experiments completed successfully!")
    else:
        print(f"\n⚠️  {total - successes} experiment(s) failed")
    
    return successes == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

