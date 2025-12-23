# Base-2 Softmax with BitNet 1.58-bit Quantization

## Project Overview

This project provides a comprehensive experimental framework for comparing **Standard Softmax (base-e)** and **Base-2 Softmax** in ultra-low-bit quantization scenarios (BitNet 1.58-bit). The framework supports multiple models and datasets across different domains:

- **Vision Models**: ResNet-18, ViT (Small/Base) on CIFAR-10/100, ImageNet-100/1K
- **Language Models**: TinyLlama on TinyStories dataset
- **Quantization**: BitNet 1.58-bit (ternary weights: {-1, 0, 1})

Large checkpoints, logs, and plots have been removed to keep the repository light for GitHub. Minimal example metrics are kept under `docs/sample_results/` for quick reference.

### Research Questions

1. **Stability**: Does Base-2 Softmax improve training stability in ultra-low-bit quantization?
2. **Performance**: How do the two Softmax variants compare in final model accuracy?
3. **Gradients**: Does Base-2 produce smoother gradients due to slower growth of 2^x vs e^x?
4. **Hardware**: Potential efficiency gains from computing 2^x vs e^x

---

## Setup

### Environment

```bash
# Create conda environment
conda create -y -n base-2-bitnet python=3.10
conda activate base-2-bitnet

# Install dependencies
pip install -r requirements.txt
```

### Quick Verification

```bash
# Verify GPU availability
nvidia-smi

# Run quick 5-epoch test
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --batch_size 128 \
    --epochs 5 \
    --softmax base2
```

---

## Running Experiments

### Automated Batch Experiments

Run all vision experiments (ResNet + ViT, both softmax variants):

```bash
bash scripts/run_quick_experiments.sh
```

This will sequentially run:
1. ResNet-18 CIFAR-10 - Standard Softmax
2. ResNet-18 CIFAR-10 - Base-2 Softmax  
3. ViT-Small CIFAR-10 - Standard Softmax
4. ViT-Small CIFAR-10 - Base-2 Softmax

Results are saved to `results/` and logs to `logs/`.

### Individual Experiments

#### Vision Models

```bash
# ResNet-18 on CIFAR-10
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --batch_size 128 \
    --epochs 50 \
    --lr 1e-3 \
    --softmax base2 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --track_grads \
    --save_checkpoints

# ViT-Small on CIFAR-10
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model vit-s \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --softmax base2 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --track_grads
```

#### Language Models

```bash
# TinyLlama on TinyStories
python scripts/train_llm.py \
    --model_name TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --dataset roneneldan/TinyStories \
    --softmax base2 \
    --batch_size 4 \
    --epochs 1 \
    --num_train_samples 5000
```

---

## Analysis & Visualization

### Generate Comparison Reports

After experiments complete:

```bash
# Generate all comparison plots
python scripts/generate_report.py

# Detailed statistical analysis
python scripts/analyze_results.py
```

This generates:
- `results/comparison_report_accuracy_comparison.png` - Accuracy curves
- `results/comparison_report_loss_comparison.png` - Loss curves
- `results/comparison_report_gradient_norms.png` - Gradient analysis
- `results/comparison_report_final_comparison.png` - Final performance bar chart
- `results/summary.txt` - Text summary with key findings
- `results/experiment_summary.csv` - All results in CSV format

---

## Project Structure

```
base-2-bitnet/
├── src/
│   ├── data/              # Dataset loaders (CIFAR, ImageNet)
│   ├── models/            # Model definitions (BitNet ResNet, ViT, LLaMA)
│   ├── ops/               # Base-2 Softmax implementation
│   ├── quant/             # BitNet 1.58-bit quantization layers
│   └── utils/             # Plotting and utilities
├── docs/
│   └── sample_results/    # Lightweight example metrics kept in Git
├── scripts/
│   ├── train_enhanced.py         # Vision model training
│   ├── train_llm.py              # LLM training
│   ├── run_quick_experiments.sh  # Batch runner
│   ├── generate_report.py        # Report generator
│   └── analyze_results.py        # Statistical analysis
├── configs/               # Configuration files (optional)
├── results/              # Experimental results and plots
├── logs/                 # Training logs
├── EXPERIMENTS_GUIDE.md  # Detailed experimental guide
└── README.md             # This file
```

---

## Outputs & Hygiene

- Experiment outputs are written to `results/`, `logs/`, `wandb/`, `new_result/`, `runs/`, and other `outputs/` or `checkpoints/` directories. These paths are gitignored to prevent pushing large artifacts.
- Sample metrics are stored in `docs/sample_results/`:
  - `llm_gpt2_base2_full.json` / `llm_gpt2_standard_full.json`: 3-epoch GPT-2 runs comparing base-2 vs standard softmax.
  - `llm_gpt2_base2_short.json` / `llm_gpt2_standard_short.json`: Shorter GPT-2 sanity checks.
  - `qwen25_wikitext2_eval.json`: Wikitext-2 evaluation across quantization settings for Qwen2.5-3B.
- Checkpoints (`*.pth`, `*.pt`, `*.safetensors`, `*.bin`) and intermediate tensors (`*.npy`, `*.npz`, `*.pkl`) remain ignored by default. Regenerate them locally using the commands above.


## Advanced Usage

### Multi-GPU Training

```bash
torchrun --nproc_per_node=3 scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model vit-s \
    --batch_size 256 \
    --softmax base2
```

### WandB Logging

```bash
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --softmax base2 \
    --wandb
```

### Resume from Checkpoint

```bash
python scripts/train_enhanced.py \
    --dataset CIFAR-10 \
    --model resnet18 \
    --softmax base2 \
    --resume results/best_resnet18_CIFAR-10_base2.pth
```









