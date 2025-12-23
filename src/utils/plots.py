import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List


def set_english_style():
    # Ensure English labels and a clean style
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.size": 12,
    })
    sns.set_theme(style="whitegrid", palette="deep")


def plot_training_curves(history, out_path: str):
    """Plot training and validation curves."""
    set_english_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if len(history.get("epoch", [])) > 0:
        axes[0].plot(history["epoch"], history.get("train_loss", []), marker='o', label="Train Loss", linewidth=2)
        axes[0].plot(history["epoch"], history.get("val_loss", []), marker='s', label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if len(history.get("epoch", [])) > 0:
        axes[1].plot(history["epoch"], history.get("val_accuracy", []), marker='^', label="Val Accuracy", linewidth=2, color='green')
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_comparison(results_list: List[Dict], out_path: str, metric: str = "accuracy"):
    """Plot comparison of multiple experiments."""
    set_english_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(results_list))
    
    for i, result in enumerate(results_list):
        history = result.get("history", {})
        label = f"{result.get('model', 'model')} - {result.get('softmax', 'std')}"
        
        if metric == "accuracy":
            data = history.get("val_accuracy", [])
            ylabel = "Validation Accuracy"
        elif metric == "loss":
            data = history.get("val_loss", [])
            ylabel = "Validation Loss"
        else:
            data = history.get(metric, [])
            ylabel = metric.capitalize()
        
        epochs = history.get("epoch", list(range(1, len(data) + 1)))
        ax.plot(epochs, data, marker='o', label=label, linewidth=2, color=colors[i])
    
    ax.set_title(f"{ylabel} Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_gradient_norms(results_list: List[Dict], out_path: str):
    """Plot gradient norms comparison."""
    set_english_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = sns.color_palette("husl", len(results_list))
    
    for i, result in enumerate(results_list):
        history = result.get("history", {})
        grad_norms = history.get("grad_norm", [])
        if not grad_norms or all(g == 0 for g in grad_norms):
            continue
        
        label = f"{result.get('model', 'model')} - Softmax (base-{result.get('softmax', 'e')})"
        epochs = history.get("epoch", list(range(1, len(grad_norms) + 1)))
        ax.plot(epochs, grad_norms, marker='o', label=label, linewidth=2, color=colors[i], alpha=0.8)
    
    ax.set_title("Gradient L2 Norm During Training", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient L2 Norm")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_comparison_bar(results_list: List[Dict], out_path: str):
    """Create bar chart comparing final performance."""
    set_english_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    accuracies = []
    colors = []
    
    color_map = {"standard": "#3498db", "base2": "#e74c3c"}
    
    for result in results_list:
        model = result.get("model", "model")
        dataset = result.get("dataset", "dataset")
        softmax = result.get("softmax", "std")
        best_acc = result.get("best_val_acc", 0.0) * 100  # Convert to percentage
        
        label = f"{model}\n{dataset}\n({softmax})"
        labels.append(label)
        accuracies.append(best_acc)
        colors.append(color_map.get(softmax, "#95a5a6"))
    
    bars = ax.bar(range(len(labels)), accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_title("Peak Validation Accuracy Comparison", fontsize=14, fontweight='bold')
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(accuracies) * 1.1)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["standard"], label='Softmax (base-e)', alpha=0.8),
        Patch(facecolor=color_map["base2"], label='Softmax (base-2)', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_comparison_report(results_dir: str, output_path: str):
    """Generate a comprehensive comparison report from all results."""
    results_files = [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]
    
    if not results_files:
        print(f"No results found in {results_dir}")
        return
    
    results_list = []
    for rf in results_files:
        with open(os.path.join(results_dir, rf), 'r') as f:
            results_list.append(json.load(f))
    
    # Generate plots
    base_name = os.path.splitext(output_path)[0]
    
    # Accuracy comparison
    plot_comparison(results_list, f"{base_name}_accuracy_comparison.png", metric="accuracy")
    
    # Loss comparison
    plot_comparison(results_list, f"{base_name}_loss_comparison.png", metric="loss")
    
    # Gradient norms (if available)
    plot_gradient_norms(results_list, f"{base_name}_gradient_norms.png")
    
    # Final bar chart
    plot_final_comparison_bar(results_list, f"{base_name}_final_comparison.png")
    
    print(f"Comparison report generated at {base_name}_*.png")


