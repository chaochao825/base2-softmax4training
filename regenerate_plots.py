import json
import os
import matplotlib.pyplot as plt
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = "/home/spco/base-2-bitnet"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.plots import plot_training_curves

def generate_comparison_plot(output_dir: str, model_name: str):
    """Generate a plot comparing standard vs base2 softmax."""
    std_path = os.path.join(output_dir, f"history_{model_name}_standard.json")
    base2_path = os.path.join(output_dir, f"history_{model_name}_base2.json")
    
    if not (os.path.exists(std_path) and os.path.exists(base2_path)):
        print(f"Missing history files in {output_dir} for {model_name}")
        return

    with open(std_path, 'r') as f:
        std_history = json.load(f)
    with open(base2_path, 'r') as f:
        base2_history = json.load(f)

    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Global font size settings
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
    print(f"Successfully regenerated comparison plot: {comp_path}")

# 1. GPT-2 Comparisons
generate_comparison_plot("/home/spco/base-2-bitnet/results_v3", "gpt2")
generate_comparison_plot("/home/spco/base-2-bitnet/results_v3/gpt2_xl", "gpt2")

# 2. Vision curves
results_v3 = "/home/spco/base-2-bitnet/results_v3"
for f in os.listdir(results_v3):
    if f.startswith("results_") and f.endswith(".json"):
        with open(os.path.join(results_v3, f), 'r') as jf:
            res = json.load(jf)
            history = res.get("history")
            if history:
                # Reconstruct output filename
                model = res.get("model")
                softmax = res.get("softmax")
                amp = res.get("amp_dtype")
                temp = str(res.get("temperature", 1.0)).replace(".", "")
                ds_name = res.get("dataset").replace("-", "").lower()
                out_png = f"curves_{ds_name}_{model}_{softmax}_{amp}_temp{temp}.png"
                out_path = os.path.join(results_v3, out_png)
                plot_training_curves(history, out_path)
                print(f"Regenerated vision curve: {out_path}")


