#!/usr/bin/env python3
"""Generate comprehensive comparison report from experimental results."""

import os
import sys
import json
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plots import generate_comparison_report


def main():
    results_dir = PROJECT_ROOT / "results"
    output_path = results_dir / "comparison_report.png"
    
    print(f"Scanning results directory: {results_dir}")
    print(f"Output path: {output_path}")
    
    # Generate all comparison plots
    generate_comparison_report(str(results_dir), str(output_path))
    
    # Also generate a text summary
    results_files = sorted([f for f in results_dir.glob("results_*.json")])
    
    if not results_files:
        print("No results found!")
        return
    
    print(f"\nFound {len(results_files)} result files")
    
    summary_path = results_dir / "summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Base-2 Softmax vs Standard Softmax - Experimental Results Summary\n")
        f.write("="*80 + "\n\n")
        
        # Group by dataset and model
        by_experiment = {}
        for rf in results_files:
            with open(rf, 'r') as fin:
                data = json.load(fin)
                key = f"{data['dataset']}_{data['model']}"
                if key not in by_experiment:
                    by_experiment[key] = []
                by_experiment[key].append(data)
        
        for exp_key, results in sorted(by_experiment.items()):
            dataset = results[0]['dataset']
            model = results[0]['model']
            
            f.write(f"\n{'-'*80}\n")
            f.write(f"Experiment: {model} on {dataset}\n")
            f.write(f"{'-'*80}\n\n")
            
            for res in sorted(results, key=lambda x: x['softmax']):
                softmax_type = res['softmax']
                best_acc = res.get('best_val_acc', 0) * 100
                final_acc = res.get('final_val_acc', 0) * 100
                final_loss = res.get('final_train_loss', 0)
                
                f.write(f"  Softmax Type: {softmax_type.upper()}\n")
                f.write(f"    - Best Val Accuracy:  {best_acc:.2f}%\n")
                f.write(f"    - Final Val Accuracy: {final_acc:.2f}%\n")
                f.write(f"    - Final Train Loss:   {final_loss:.4f}\n")
                f.write(f"    - Temperature:        {res.get('temperature', 1.0)}\n")
                f.write("\n")
            
            # Compare if we have both softmax types
            if len(results) == 2:
                std_res = next((r for r in results if r['softmax'] == 'standard'), None)
                b2_res = next((r for r in results if r['softmax'] == 'base2'), None)
                
                if std_res and b2_res:
                    std_acc = std_res.get('best_val_acc', 0) * 100
                    b2_acc = b2_res.get('best_val_acc', 0) * 100
                    diff = b2_acc - std_acc
                    
                    f.write(f"  Comparison:\n")
                    f.write(f"    - Base-2 vs Standard: {diff:+.2f}% ")
                    if diff > 0:
                        f.write("(Base-2 BETTER ✓)\n")
                    elif diff < 0:
                        f.write("(Standard BETTER)\n")
                    else:
                        f.write("(EQUAL)\n")
                    f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Key Findings:\n")
        f.write("="*80 + "\n\n")
        
        # Calculate overall statistics
        all_improvements = []
        for exp_key, results in by_experiment.items():
            if len(results) == 2:
                std_res = next((r for r in results if r['softmax'] == 'standard'), None)
                b2_res = next((r for r in results if r['softmax'] == 'base2'), None)
                if std_res and b2_res:
                    diff = (b2_res.get('best_val_acc', 0) - std_res.get('best_val_acc', 0)) * 100
                    all_improvements.append((exp_key, diff))
        
        if all_improvements:
            avg_improvement = sum(d for _, d in all_improvements) / len(all_improvements)
            better_count = sum(1 for _, d in all_improvements if d > 0)
            
            f.write(f"1. Average improvement with Base-2 Softmax: {avg_improvement:+.2f}%\n")
            f.write(f"2. Base-2 performed better in {better_count}/{len(all_improvements)} experiments\n")
            f.write(f"3. Largest improvement: {max(all_improvements, key=lambda x: x[1])}\n")
            f.write(f"4. Largest degradation: {min(all_improvements, key=lambda x: x[1])}\n")
        
        f.write("\n")
    
    print(f"\n✓ Summary report saved to: {summary_path}")
    
    # Print to console as well
    with open(summary_path, 'r') as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()


