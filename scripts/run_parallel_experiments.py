#!/usr/bin/env python3
"""
Run Vision and LLM experiments in parallel using different GPU groups.
- Vision experiments: GPU 0, 1 (2 GPUs)
- LLM experiments: GPU 2 (1 GPU)
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Experiment configurations
VISION_EXPERIMENTS = [
    {
        "name": "ResNet18-CIFAR10-Standard",
        "model": "resnet18",
        "softmax": "standard",
        "batch_size": 256,
        "epochs": 100,
    },
    {
        "name": "ResNet18-CIFAR10-Base2",
        "model": "resnet18",
        "softmax": "base2",
        "batch_size": 256,
        "epochs": 100,
    },
    {
        "name": "ViT-S-CIFAR10-Standard",
        "model": "vit-s",
        "softmax": "standard",
        "batch_size": 256,
        "epochs": 100,
    },
    {
        "name": "ViT-S-CIFAR10-Base2",
        "model": "vit-s",
        "softmax": "base2",
        "batch_size": 256,
        "epochs": 100,
    },
]

LLM_EXPERIMENTS = [
    {
        "name": "GPT2-Standard",
        "model": "gpt2",
        "softmax": "standard",
        "batch_size": 8,
        "epochs": 3,
    },
    {
        "name": "GPT2-Base2",
        "model": "gpt2",
        "softmax": "base2",
        "batch_size": 8,
        "epochs": 3,
    },
]


def run_vision_experiments():
    """Run vision experiments on GPU 0,1"""
    print(f"\n{'='*80}")
    print(f"VISION EXPERIMENTS - Using GPU 0,1")
    print(f"{'='*80}\n")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["WANDB_MODE"] = "offline"
    
    for exp in VISION_EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"Running: {exp['name']}")
        print(f"{'='*80}\n")
        
        cmd = [
            "torchrun",
            "--nproc_per_node=2",
            "--master_port=29500",
            "scripts/train_enhanced.py",
            f"--dataset=CIFAR-10",
            f"--model={exp['model']}",
            f"--softmax={exp['softmax']}",
            f"--batch_size={exp['batch_size']}",
            f"--epochs={exp['epochs']}",
            "--lr=1e-3" if exp['model'] == 'resnet18' else "--lr=5e-4",
            "--scheduler=cosine",
            "--warmup_epochs=5" if exp['model'] == 'resnet18' else "--warmup_epochs=10",
            "--track_grads",
            "--save_checkpoints",
            "--wandb",
        ]
        
        log_file = f"logs/vision_{exp['model']}_{exp['softmax']}.log"
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        
        if process.returncode == 0:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed with code {process.returncode}")
        
        time.sleep(5)


def run_llm_experiments():
    """Run LLM experiments on GPU 2"""
    print(f"\n{'='*80}")
    print(f"LLM EXPERIMENTS - Using GPU 2")
    print(f"{'='*80}\n")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["WANDB_MODE"] = "offline"
    
    for exp in LLM_EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"Running: {exp['name']}")
        print(f"{'='*80}\n")
        
        cmd = [
            "python",  # Single GPU for LLM
            "scripts/train_llm_enhanced.py",
            f"--model_name={exp['model']}",
            f"--softmax={exp['softmax']}",
            f"--batch_size={exp['batch_size']}",
            f"--epochs={exp['epochs']}",
            "--lr=5e-5",
            "--max_length=256",
            "--num_train_samples=10000",
            "--num_val_samples=1000",
            "--track_grads",
            "--wandb",
        ]
        
        log_file = f"logs/llm_{exp['model']}_{exp['softmax']}.log"
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
        
        if process.returncode == 0:
            print(f"✓ {exp['name']} completed successfully")
        else:
            print(f"✗ {exp['name']} failed with code {process.returncode}")
        
        time.sleep(5)


def main():
    import argparse
    from multiprocessing import Process
    
    parser = argparse.ArgumentParser(description="Run parallel experiments")
    parser.add_argument("--mode", type=str, default="parallel", 
                       choices=["parallel", "vision", "llm"],
                       help="Execution mode: parallel (both), vision only, or llm only")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"PARALLEL EXPERIMENT EXECUTION")
    print(f"{'='*80}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {args.mode}")
    print(f"{'='*80}\n")
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if args.mode == "parallel":
        # Run vision and LLM in parallel
        vision_process = Process(target=run_vision_experiments)
        llm_process = Process(target=run_llm_experiments)
        
        print("Starting Vision experiments (GPU 0,1)...")
        vision_process.start()
        
        print("Starting LLM experiments (GPU 2)...")
        llm_process.start()
        
        print("\nBoth processes started. Waiting for completion...\n")
        
        vision_process.join()
        llm_process.join()
        
    elif args.mode == "vision":
        run_vision_experiments()
        
    elif args.mode == "llm":
        run_llm_experiments()
    
    print(f"\n{'='*80}")
    print(f"EXECUTION COMPLETE")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

