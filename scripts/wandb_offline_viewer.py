#!/usr/bin/env python3
"""
WandB Offline Viewer - Parse offline runs and create interactive dashboard
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def parse_wandb_run(run_dir):
    """Parse a WandB offline run directory"""
    run_path = Path(run_dir)
    
    # Find wandb history file
    files_dir = run_path / "files"
    if not files_dir.exists():
        return None
    
    wandb_summary = files_dir / "wandb-summary.json"
    wandb_history = files_dir / "wandb-history.jsonl"
    
    run_info = {
        'run_id': run_path.name.split('-')[-1],
        'run_name': run_path.name,
        'path': str(run_path),
        'config': {},
        'history': [],
        'summary': {}
    }
    
    # Parse config
    config_file = files_dir / "config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            import yaml
            try:
                run_info['config'] = yaml.safe_load(f)
            except:
                pass
    
    # Parse summary
    if wandb_summary.exists():
        with open(wandb_summary, 'r') as f:
            try:
                run_info['summary'] = json.load(f)
            except:
                pass
    
    # Parse history
    if wandb_history.exists():
        with open(wandb_history, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    run_info['history'].append(entry)
                except:
                    pass
    
    return run_info if run_info['history'] else None


def create_dashboard(runs, output_dir):
    """Create visualization dashboard from runs"""
    
    # Group runs by experiment type
    vision_runs = []
    llm_runs = []
    
    for run in runs:
        config = run.get('config', {})
        if 'dataset' in config:
            vision_runs.append(run)
        elif 'model_name' in config or 'gpt' in str(config).lower():
            llm_runs.append(run)
    
    print(f"Found {len(vision_runs)} vision runs and {len(llm_runs)} LLM runs")
    
    # Create dashboard
    if vision_runs:
        create_vision_dashboard(vision_runs, output_dir)
    
    if llm_runs:
        create_llm_dashboard(llm_runs, output_dir)
    
    create_summary_page(runs, output_dir)


def create_vision_dashboard(runs, output_dir):
    """Create vision experiments dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot training loss
    ax = axes[0, 0]
    for run in runs:
        history = run['history']
        if not history:
            continue
        
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]
        train_loss = [h.get('train_loss') for h in history]
        
        if any(train_loss):
            label = run['config'].get('model', 'unknown') + '_' + run['config'].get('softmax', 'unknown')
            ax.plot(epochs, train_loss, label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot validation accuracy
    ax = axes[0, 1]
    for run in runs:
        history = run['history']
        if not history:
            continue
        
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]
        val_acc = [h.get('val_accuracy', h.get('val_acc')) for h in history]
        
        if any(val_acc):
            val_acc = [a * 100 if a and a < 1 else a for a in val_acc]  # Convert to percentage
            label = run['config'].get('model', 'unknown') + '_' + run['config'].get('softmax', 'unknown')
            ax.plot(epochs, val_acc, label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot gradient norms
    ax = axes[1, 0]
    for run in runs:
        history = run['history']
        if not history:
            continue
        
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]
        grad_norms = [h.get('grad_norm') for h in history]
        
        if any(grad_norms):
            grad_norms = [g for g in grad_norms if g]
            if grad_norms:
                label = run['config'].get('model', 'unknown') + '_' + run['config'].get('softmax', 'unknown')
                ax.plot(epochs[:len(grad_norms)], grad_norms, label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient L2 Norm')
    ax.set_title('Gradient Norms (log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot learning rate
    ax = axes[1, 1]
    for run in runs:
        history = run['history']
        if not history:
            continue
        
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]
        lr = [h.get('lr') for h in history]
        
        if any(lr):
            label = run['config'].get('model', 'unknown') + '_' + run['config'].get('softmax', 'unknown')
            ax.plot(epochs, lr, label=label, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Vision Experiments - Real-time Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'wandb_vision_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path}")


def create_llm_dashboard(runs, output_dir):
    """Create LLM experiments dashboard"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot perplexity
    ax = axes[0]
    for run in runs:
        history = run['history']
        if not history:
            continue
        
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]
        ppl = [h.get('val_perplexity', h.get('val_ppl')) for h in history]
        
        if any(ppl):
            label = run['config'].get('softmax', 'unknown')
            ax.plot(epochs, ppl, label=label, linewidth=2, marker='o', markersize=8, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title('Validation Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot loss
    ax = axes[1]
    for run in runs:
        history = run['history']
        if not history:
            continue
        
        epochs = [h.get('epoch', i) for i, h in enumerate(history)]
        loss = [h.get('val_loss') for h in history]
        
        if any(loss):
            label = run['config'].get('softmax', 'unknown')
            ax.plot(epochs, loss, label=label, linewidth=2, marker='s', markersize=8, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('LLM Experiments - Real-time Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'wandb_llm_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: {output_path}")


def create_summary_page(runs, output_dir):
    """Create text summary of all runs"""
    summary_path = os.path.join(output_dir, 'wandb_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WandB Offline Runs Summary\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Runs: {len(runs)}\n")
        f.write("="*80 + "\n\n")
        
        for i, run in enumerate(runs, 1):
            f.write(f"\nRun {i}: {run['run_name']}\n")
            f.write("-"*80 + "\n")
            
            config = run.get('config', {})
            if config:
                f.write("Config:\n")
                for key, value in config.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}: {value.get('value', value)}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            summary = run.get('summary', {})
            if summary:
                f.write("\nFinal Metrics:\n")
                for key, value in summary.items():
                    if not key.startswith('_'):
                        f.write(f"  {key}: {value}\n")
            
            history = run.get('history', [])
            if history:
                f.write(f"\nHistory: {len(history)} steps recorded\n")
            
            f.write("\n")
    
    print(f"✓ Created: {summary_path}")


def watch_mode(wandb_dir, output_dir, interval=30):
    """Watch mode - continuously update dashboards"""
    print(f"\nWandB Offline Viewer - Watch Mode")
    print(f"Monitoring: {wandb_dir}")
    print(f"Output: {output_dir}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Find all offline runs
            runs = []
            for run_dir in Path(wandb_dir).glob("offline-run-*"):
                run_info = parse_wandb_run(run_dir)
                if run_info:
                    runs.append(run_info)
            
            if runs:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Updating dashboards... ({len(runs)} runs)")
                create_dashboard(runs, output_dir)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No runs found")
            
            import time
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nStopping watch mode...")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="WandB Offline Viewer")
    parser.add_argument("--wandb_dir", type=str, default="/home/spco/base-2-bitnet/wandb")
    parser.add_argument("--output_dir", type=str, default="/home/spco/base-2-bitnet/results")
    parser.add_argument("--watch", action="store_true", help="Continuously update (watch mode)")
    parser.add_argument("--interval", type=int, default=30, help="Update interval in seconds (watch mode)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.watch:
        watch_mode(args.wandb_dir, args.output_dir, args.interval)
    else:
        # One-time generation
        print(f"\nWandB Offline Viewer")
        print(f"Scanning: {args.wandb_dir}\n")
        
        runs = []
        for run_dir in Path(args.wandb_dir).glob("offline-run-*"):
            run_info = parse_wandb_run(run_dir)
            if run_info:
                runs.append(run_info)
                print(f"  Found: {run_dir.name}")
        
        if runs:
            print(f"\nProcessing {len(runs)} runs...\n")
            create_dashboard(runs, args.output_dir)
            print(f"\n✓ Dashboard created in: {args.output_dir}")
        else:
            print("\n⚠ No offline runs found")


if __name__ == "__main__":
    # Install pyyaml if needed
    try:
        import yaml
    except ImportError:
        print("Installing pyyaml...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
        import yaml
    
    main()

