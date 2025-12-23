#!/usr/bin/env python3
"""Batch experiment runner for base-2 softmax comparison."""

import os
import sys
import subprocess
import time
from pathlib import Path

# Experiment configurations
EXPERIMENTS = [
    # Step 2: ResNet-18 on CIFAR-10 (Quick validation)
    {
        "name": "ResNet18-CIFAR10-Standard",
        "args": [
            "--dataset", "CIFAR-10",
            "--model", "resnet18",
            "--batch_size", "256",
            "--epochs", "1000",
            "--lr", "1e-3",
            "--softmax", "standard",
            "--scheduler", "cosine",
            "--warmup_epochs", "5",
            "--track_grads",
            "--save_checkpoints",
        ]
    },
    {
        "name": "ResNet18-CIFAR10-Base2",
        "args": [
            "--dataset", "CIFAR-10",
            "--model", "resnet18",
            "--batch_size", "256",
            "--epochs", "1000",
            "--lr", "1e-3",
            "--softmax", "base2",
            "--scheduler", "cosine",
            "--warmup_epochs", "5",
            "--track_grads",
            "--save_checkpoints",
        ]
    },
    # Step 3: ViT on CIFAR-10
    {
        "name": "ViT-Small-CIFAR10-Standard",
        "args": [
            "--dataset", "CIFAR-10",
            "--model", "vit-s",
            "--batch_size", "128",
            "--epochs", "1000",
            "--lr", "5e-4",
            "--softmax", "standard",
            "--scheduler", "cosine",
            "--warmup_epochs", "10",
            "--track_grads",
            "--save_checkpoints",
        ]
    },
    {
        "name": "ViT-Small-CIFAR10-Base2",
        "args": [
            "--dataset", "CIFAR-10",
            "--model", "vit-s",
            "--batch_size", "128",
            "--epochs", "1000",
            "--lr", "5e-4",
            "--softmax", "base2",
            "--scheduler", "cosine",
            "--warmup_epochs", "10",
            "--track_grads",
            "--save_checkpoints",
        ]
    },
]


def run_experiment(exp_config, use_wandb=False, num_gpus=3):
    """Run a single experiment."""
    print(f"\n{'='*80}")
    print(f"Starting Experiment: {exp_config['name']}")
    print(f"{'='*80}\n")
    
    script_path = Path(__file__).parent / "train_enhanced.py"
    
    # Build command
    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            "--master_port=23456",
            str(script_path),
        ]
    else:
        cmd = ["python", str(script_path)]
    
    cmd.extend(exp_config["args"])
    
    if use_wandb:
        cmd.append("--wandb")
    
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ Experiment {exp_config['name']} completed successfully in {elapsed/60:.2f} minutes")
        print(f"{'='*80}\n")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✗ Experiment {exp_config['name']} failed after {elapsed/60:.2f} minutes")
        print(f"Error: {e}")
        print(f"{'='*80}\n")
        return False


def generate_final_report():
    """Generate comprehensive comparison report after all experiments."""
    print(f"\n{'='*80}")
    print("Generating Final Comparison Report...")
    print(f"{'='*80}\n")
    
    report_script = Path(__file__).parent / "generate_report.py"
    cmd = ["python", str(report_script)]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✓ Final report generated successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to generate report: {e}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run batch experiments")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--num_gpus", type=int, default=3, help="Number of GPUs to use")
    parser.add_argument("--experiments", type=str, nargs="+", help="Specific experiments to run (by index)")
    parser.add_argument("--skip_report", action="store_true", help="Skip final report generation")
    args = parser.parse_args()
    
    # Select experiments to run
    if args.experiments:
        indices = [int(i) for i in args.experiments]
        experiments_to_run = [EXPERIMENTS[i] for i in indices if i < len(EXPERIMENTS)]
    else:
        experiments_to_run = EXPERIMENTS
    
    print(f"\n{'#'*80}")
    print(f"# Base-2 Softmax vs Standard Softmax Comparison Experiments")
    print(f"# Total experiments: {len(experiments_to_run)}")
    print(f"# GPUs: {args.num_gpus}")
    print(f"# W&B: {'Enabled' if args.wandb else 'Disabled'}")
    print(f"{'#'*80}\n")
    
    results = []
    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] Running: {exp['name']}")
        success = run_experiment(exp, use_wandb=args.wandb, num_gpus=args.num_gpus)
        results.append((exp['name'], success))
        
        # Small delay between experiments
        if i < len(experiments_to_run):
            time.sleep(5)
    
    # Generate final report
    if not args.skip_report:
        generate_final_report()
    
    # Print summary
    print(f"\n{'#'*80}")
    print("# Experiment Summary")
    print(f"{'#'*80}\n")
    
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {name}")
    
    success_count = sum(1 for _, s in results if s)
    print(f"\nTotal: {success_count}/{len(results)} experiments completed successfully\n")


if __name__ == "__main__":
    main()


