#!/usr/bin/env python3
"""
Generate comprehensive comparison visualizations for Base-2 Softmax vs Standard Softmax experiments.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Ensure project root is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_results(results_dir: str) -> Dict:
    """Load all experiment results from JSON files."""
    results = {}
    results_path = Path(results_dir)
    
    for json_file in results_path.glob("results_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                key = f"{data['model']}_{data['dataset']}_{data['softmax']}"
                results[key] = data
                print(f"Loaded: {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results


def plot_training_loss_comparison(results: Dict, output_path: str):
    """Plot training loss comparison for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ResNet-18 CIFAR-10
    ax = axes[0]
    for softmax_type in ['standard', 'base2']:
        key = f"resnet18_CIFAR-10_{softmax_type}"
        if key in results:
            history = results[key]['history']
            label = f"Softmax (base-e)" if softmax_type == 'standard' else f"Softmax (base-2)"
            ax.plot(history['epoch'], history['train_loss'], label=label, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title('Training Loss: ResNet-18 on CIFAR-10', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # ViT-Small CIFAR-10
    ax = axes[1]
    for softmax_type in ['standard', 'base2']:
        key = f"vit-s_CIFAR-10_{softmax_type}"
        if key in results:
            history = results[key]['history']
            label = f"Softmax (base-e)" if softmax_type == 'standard' else f"Softmax (base-2)"
            ax.plot(history['epoch'], history['train_loss'], label=label, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title('Training Loss: ViT-Small on CIFAR-10', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_validation_accuracy_comparison(results: Dict, output_path: str):
    """Plot validation accuracy comparison for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ResNet-18 CIFAR-10
    ax = axes[0]
    for softmax_type in ['standard', 'base2']:
        key = f"resnet18_CIFAR-10_{softmax_type}"
        if key in results:
            history = results[key]['history']
            acc = [a * 100 for a in history['val_accuracy']]  # Convert to percentage
            label = f"Softmax (base-e)" if softmax_type == 'standard' else f"Softmax (base-2)"
            ax.plot(history['epoch'], acc, label=label, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Validation Accuracy: ResNet-18 on CIFAR-10', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # ViT-Small CIFAR-10
    ax = axes[1]
    for softmax_type in ['standard', 'base2']:
        key = f"vit-s_CIFAR-10_{softmax_type}"
        if key in results:
            history = results[key]['history']
            acc = [a * 100 for a in history['val_accuracy']]  # Convert to percentage
            label = f"Softmax (base-e)" if softmax_type == 'standard' else f"Softmax (base-2)"
            ax.plot(history['epoch'], acc, label=label, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14)
    ax.set_title('Validation Accuracy: ViT-Small on CIFAR-10', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gradient_norms_comparison(results: Dict, output_path: str):
    """Plot gradient norms comparison for all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ResNet-18 CIFAR-10
    ax = axes[0]
    for softmax_type in ['standard', 'base2']:
        key = f"resnet18_CIFAR-10_{softmax_type}"
        if key in results and 'grad_norm' in results[key]['history']:
            history = results[key]['history']
            grad_norms = [g for g in history['grad_norm'] if g > 0]
            epochs = [e for e, g in zip(history['epoch'], history['grad_norm']) if g > 0]
            if grad_norms:
                label = f"Softmax (base-e)" if softmax_type == 'standard' else f"Softmax (base-2)"
                ax.plot(epochs, grad_norms, label=label, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Gradient L2 Norm', fontsize=14)
    ax.set_title('Gradient Norms: ResNet-18 on CIFAR-10', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # ViT-Small CIFAR-10
    ax = axes[1]
    for softmax_type in ['standard', 'base2']:
        key = f"vit-s_CIFAR-10_{softmax_type}"
        if key in results and 'grad_norm' in results[key]['history']:
            history = results[key]['history']
            grad_norms = [g for g in history['grad_norm'] if g > 0]
            epochs = [e for e, g in zip(history['epoch'], history['grad_norm']) if g > 0]
            if grad_norms:
                label = f"Softmax (base-e)" if softmax_type == 'standard' else f"Softmax (base-2)"
                ax.plot(epochs, grad_norms, label=label, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Gradient L2 Norm', fontsize=14)
    ax.set_title('Gradient Norms: ViT-Small on CIFAR-10', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_final_performance_comparison(results: Dict, output_path: str):
    """Plot final performance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = []
    standard_accs = []
    base2_accs = []
    
    for model in ['resnet18', 'vit-s']:
        model_name = 'ResNet-18' if model == 'resnet18' else 'ViT-Small'
        models.append(f"{model_name}\n(CIFAR-10)")
        
        # Standard softmax
        key_std = f"{model}_CIFAR-10_standard"
        if key_std in results:
            standard_accs.append(results[key_std]['best_val_acc'] * 100)
        else:
            standard_accs.append(0)
        
        # Base-2 softmax
        key_b2 = f"{model}_CIFAR-10_base2"
        if key_b2 in results:
            base2_accs.append(results[key_b2]['best_val_acc'] * 100)
        else:
            base2_accs.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, standard_accs, width, label='Softmax (base-e)', 
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, base2_accs, width, label='Softmax (base-2)', 
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Peak Validation Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.legend(fontsize=13, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, max(max(standard_accs), max(base2_accs)) * 1.15])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_metrics(results: Dict, output_path: str):
    """Plot combined training and validation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = [('resnet18', 'ResNet-18'), ('vit-s', 'ViT-Small')]
    
    for idx, (model_key, model_name) in enumerate(models):
        # Training Loss
        ax = axes[idx, 0]
        for softmax_type in ['standard', 'base2']:
            key = f"{model_key}_CIFAR-10_{softmax_type}"
            if key in results:
                history = results[key]['history']
                label = f"base-e" if softmax_type == 'standard' else f"base-2"
                ax.plot(history['epoch'], history['train_loss'], label=label, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'{model_name}: Training Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Validation Accuracy
        ax = axes[idx, 1]
        for softmax_type in ['standard', 'base2']:
            key = f"{model_key}_CIFAR-10_{softmax_type}"
            if key in results:
                history = results[key]['history']
                acc = [a * 100 for a in history['val_accuracy']]
                label = f"base-e" if softmax_type == 'standard' else f"base-2"
                ax.plot(history['epoch'], acc, label=label, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax.set_title(f'{model_name}: Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Base-2 Softmax vs Standard Softmax in Ultra-Low Bit Quantization', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_text_summary(results: Dict, output_path: str):
    """Generate text summary of all experiments."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Base-2 Softmax vs Standard Softmax Comparison Summary\n")
        f.write("Ultra-Low Bit Quantization (BitNet 1.58-bit)\n")
        f.write("="*80 + "\n\n")
        
        for model in ['resnet18', 'vit-s']:
            model_name = 'ResNet-18' if model == 'resnet18' else 'ViT-Small'
            f.write(f"\n{'-'*80}\n")
            f.write(f"{model_name} on CIFAR-10\n")
            f.write(f"{'-'*80}\n")
            
            for softmax_type in ['standard', 'base2']:
                key = f"{model}_CIFAR-10_{softmax_type}"
                if key in results:
                    res = results[key]
                    softmax_name = "Standard Softmax (base-e)" if softmax_type == 'standard' else "Base-2 Softmax"
                    f.write(f"\n{softmax_name}:\n")
                    f.write(f"  Best Validation Accuracy: {res['best_val_acc']*100:.2f}%\n")
                    f.write(f"  Final Validation Accuracy: {res['final_val_acc']*100:.2f}%\n")
                    f.write(f"  Final Training Loss: {res['final_train_loss']:.4f}\n")
                    
                    # Calculate average gradient norm if available
                    if 'grad_norm' in res['history']:
                        grad_norms = [g for g in res['history']['grad_norm'] if g > 0]
                        if grad_norms:
                            avg_grad = sum(grad_norms) / len(grad_norms)
                            f.write(f"  Average Gradient Norm: {avg_grad:.4f}\n")
            
            # Comparison
            key_std = f"{model}_CIFAR-10_standard"
            key_b2 = f"{model}_CIFAR-10_base2"
            if key_std in results and key_b2 in results:
                acc_std = results[key_std]['best_val_acc'] * 100
                acc_b2 = results[key_b2]['best_val_acc'] * 100
                diff = acc_b2 - acc_std
                f.write(f"\nAccuracy Difference (Base-2 - Standard): {diff:+.2f}%\n")
        
        f.write(f"\n{'='*80}\n")
        f.write("Summary Statistics\n")
        f.write(f"{'='*80}\n\n")
        
        # Overall statistics
        all_std_accs = []
        all_b2_accs = []
        for model in ['resnet18', 'vit-s']:
            key_std = f"{model}_CIFAR-10_standard"
            key_b2 = f"{model}_CIFAR-10_base2"
            if key_std in results:
                all_std_accs.append(results[key_std]['best_val_acc'] * 100)
            if key_b2 in results:
                all_b2_accs.append(results[key_b2]['best_val_acc'] * 100)
        
        if all_std_accs and all_b2_accs:
            f.write(f"Average Accuracy (Standard): {np.mean(all_std_accs):.2f}%\n")
            f.write(f"Average Accuracy (Base-2): {np.mean(all_b2_accs):.2f}%\n")
            f.write(f"Average Difference: {np.mean(all_b2_accs) - np.mean(all_std_accs):+.2f}%\n")
        
        f.write(f"\n{'='*80}\n")
    
    print(f"Saved: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive comparison visualizations")
    parser.add_argument("--results_dir", type=str, default="/home/spco/base-2-bitnet/results")
    parser.add_argument("--output_prefix", type=str, default="comprehensive_comparison")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Comprehensive Comparison Report Generator")
    print("="*80 + "\n")
    
    # Load results
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found! Please run experiments first.")
        return
    
    print(f"\nFound {len(results)} experiment results\n")
    
    # Generate all plots
    output_dir = Path(args.results_dir)
    
    print("\nGenerating visualizations...\n")
    
    plot_training_loss_comparison(
        results, 
        str(output_dir / f"{args.output_prefix}_training_loss.png")
    )
    
    plot_validation_accuracy_comparison(
        results, 
        str(output_dir / f"{args.output_prefix}_validation_accuracy.png")
    )
    
    plot_gradient_norms_comparison(
        results, 
        str(output_dir / f"{args.output_prefix}_gradient_norms.png")
    )
    
    plot_final_performance_comparison(
        results, 
        str(output_dir / f"{args.output_prefix}_final_performance.png")
    )
    
    plot_combined_metrics(
        results, 
        str(output_dir / f"{args.output_prefix}_combined_metrics.png")
    )
    
    generate_text_summary(
        results, 
        str(output_dir / f"{args.output_prefix}_summary.txt")
    )
    
    print("\n" + "="*80)
    print("✓ All visualizations generated successfully!")
    print(f"Output directory: {args.results_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

