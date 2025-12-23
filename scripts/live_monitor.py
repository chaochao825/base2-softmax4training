#!/usr/bin/env python3
"""
Live Experiment Monitor - Real-time visualization from JSON results
"""

import os
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_json_results(results_dir):
    """Load all result JSON files"""
    results = {'vision': {}, 'llm': {}}
    
    for json_file in Path(results_dir).glob("results_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if 'llm' in json_file.name:
                key = f"{data.get('model', 'unknown')}_{data.get('softmax', 'unknown')}"
                results['llm'][key] = data
            else:
                key = f"{data.get('model', 'unknown')}_{data.get('softmax', 'unknown')}"
                results['vision'][key] = data
                
        except Exception as e:
            print(f"  Warning: Could not load {json_file.name}: {e}")
    
    return results


def create_live_dashboard(results, output_dir):
    """Create live training dashboard"""
    
    vision_results = results['vision']
    llm_results = results['llm']
    
    # Create figure
    has_both = vision_results and llm_results
    fig = plt.figure(figsize=(20, 12) if has_both else (16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3) if has_both else fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    if not vision_results and not llm_results:
        print("No results to visualize yet")
        return
    
    # Vision: Training Loss
    if vision_results:
        ax = fig.add_subplot(gs[0, 0])
        for key, data in vision_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'train_loss' in history:
                ax.plot(history['epoch'], history['train_loss'], 
                       label=key, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Training Loss', fontweight='bold')
        ax.set_title('Vision: Training Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Vision: Validation Accuracy
    if vision_results:
        ax = fig.add_subplot(gs[0, 1])
        for key, data in vision_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'val_accuracy' in history:
                acc = [a * 100 for a in history['val_accuracy']]
                ax.plot(history['epoch'], acc, 
                       label=key, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontweight='bold')
        ax.set_title('Vision: Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Vision: Validation Loss
    if vision_results:
        ax = fig.add_subplot(gs[0, 2])
        for key, data in vision_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'val_loss' in history:
                ax.plot(history['epoch'], history['val_loss'], 
                       label=key, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Loss', fontweight='bold')
        ax.set_title('Vision: Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Vision: Learning Rate
    if vision_results:
        ax = fig.add_subplot(gs[1, 0])
        for key, data in vision_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'lr' in history:
                ax.plot(history['epoch'], history['lr'], 
                       label=key, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Learning Rate', fontweight='bold')
        ax.set_title('Vision: Learning Rate', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Vision: Gradient Norms
    if vision_results:
        ax = fig.add_subplot(gs[1, 1])
        for key, data in vision_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'grad_norm' in history:
                grad_norms = [g for g in history['grad_norm'] if g > 0]
                epochs = history['epoch'][:len(grad_norms)]
                if grad_norms:
                    ax.plot(epochs, grad_norms, 
                           label=key, linewidth=2, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Gradient L2 Norm', fontweight='bold')
        ax.set_title('Vision: Gradient Norms (log)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # Status Summary (text)
    if vision_results:
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        
        status_text = f"EXPERIMENT STATUS\n{datetime.now().strftime('%H:%M:%S')}\n\n"
        status_text += "VISION (3 GPUs):\n"
        for key, data in vision_results.items():
            history = data.get('history', {})
            if 'val_accuracy' in history and history['val_accuracy']:
                best_acc = max(history['val_accuracy']) * 100
                current_epoch = len(history.get('epoch', []))
                status_text += f"  {key[:20]}\n"
                status_text += f"    Epoch: {current_epoch}\n"
                status_text += f"    Best: {best_acc:.2f}%\n"
        
        if llm_results:
            status_text += "\nLLM (GPU 2):\n"
            for key, data in llm_results.items():
                history = data.get('history', {})
                if 'val_ppl' in history and history['val_ppl']:
                    best_ppl = min(history['val_ppl'])
                    current_epoch = len(history.get('epoch', []))
                    status_text += f"  {key[:20]}\n"
                    status_text += f"    Epoch: {current_epoch}\n"
                    status_text += f"    Best PPL: {best_ppl:.2f}\n"
        
        ax.text(0.1, 0.9, status_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # LLM: Perplexity
    if llm_results and has_both:
        ax = fig.add_subplot(gs[2, 0])
        for key, data in llm_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'val_ppl' in history:
                ax.plot(history['epoch'], history['val_ppl'], 
                       label=key, linewidth=2.5, marker='o', markersize=10, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Perplexity', fontweight='bold')
        ax.set_title('LLM: Validation Perplexity', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # LLM: Loss
    if llm_results and has_both:
        ax = fig.add_subplot(gs[2, 1])
        for key, data in llm_results.items():
            history = data.get('history', {})
            if 'epoch' in history and 'val_loss' in history:
                ax.plot(history['epoch'], history['val_loss'], 
                       label=key, linewidth=2.5, marker='s', markersize=10, alpha=0.8)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Validation Loss', fontweight='bold')
        ax.set_title('LLM: Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Progress bars
    if vision_results or llm_results:
        ax = fig.add_subplot(gs[2 if has_both else 1, 2])
        ax.axis('off')
        
        progress_text = "PROGRESS\n\n"
        
        vision_complete = len([k for k in vision_results.keys()])
        llm_complete = len([k for k in llm_results.keys()])
        total_complete = vision_complete + llm_complete
        
        progress_text += f"Vision: {vision_complete}/4 experiments\n"
        progress_text += "█" * vision_complete + "░" * (4 - vision_complete) + "\n\n"
        
        progress_text += f"LLM: {llm_complete}/2 experiments\n"
        progress_text += "█" * llm_complete + "░" * (2 - llm_complete) + "\n\n"
        
        progress_text += f"Total: {total_complete}/6 experiments\n"
        progress_text += "█" * total_complete + "░" * (6 - total_complete) + "\n"
        
        ax.text(0.1, 0.9, progress_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('Real-Time Experiment Monitor: Base-2 vs Standard Softmax', 
                 fontsize=16, fontweight='bold')
    
    output_path = os.path.join(output_dir, 'LIVE_DASHBOARD.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def watch_mode(results_dir, interval=30):
    """Continuously update dashboard"""
    print(f"\n{'='*80}")
    print(f"LIVE EXPERIMENT MONITOR")
    print(f"{'='*80}")
    print(f"Monitoring: {results_dir}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            results = load_json_results(results_dir)
            
            total_experiments = len(results['vision']) + len(results['llm'])
            
            if total_experiments > 0:
                output_file = create_live_dashboard(results, results_dir)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Update #{iteration}: {total_experiments}/6 experiments | "
                      f"Dashboard: {output_file}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Update #{iteration}: No results yet, waiting...")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nStopping live monitor...")
        print(f"Final dashboard saved to: {results_dir}/LIVE_DASHBOARD.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Experiment Monitor")
    parser.add_argument("--results_dir", type=str, 
                       default="/home/spco/base-2-bitnet/results")
    parser.add_argument("--watch", action="store_true", 
                       help="Continuously update (watch mode)")
    parser.add_argument("--interval", type=int, default=30, 
                       help="Update interval in seconds")
    
    args = parser.parse_args()
    
    if args.watch:
        watch_mode(args.results_dir, args.interval)
    else:
        # One-time generation
        print(f"\nGenerating live dashboard from: {args.results_dir}")
        results = load_json_results(args.results_dir)
        
        total = len(results['vision']) + len(results['llm'])
        print(f"Found {total} experiments")
        print(f"  Vision: {len(results['vision'])}")
        print(f"  LLM: {len(results['llm'])}")
        
        if total > 0:
            output_file = create_live_dashboard(results, args.results_dir)
            print(f"\n✓ Dashboard created: {output_file}")
        else:
            print("\n⚠ No results found yet")


if __name__ == "__main__":
    main()

