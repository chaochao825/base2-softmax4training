#!/usr/bin/env python3
"""Detailed analysis script for experimental results."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_results(results_dir="results"):
    """Load all experimental results."""
    results_dir = Path(results_dir)
    results_files = list(results_dir.glob("results_*.json"))
    
    print(f"Found {len(results_files)} result files")
    
    all_results = []
    for rf in results_files:
        with open(rf, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    
    return all_results


def create_summary_dataframe(all_results):
    """Create summary DataFrame from results."""
    summary_data = []
    for res in all_results:
        summary_data.append({
            'Dataset': res['dataset'],
            'Model': res['model'],
            'Softmax': res['softmax'],
            'Temperature': res.get('temperature', 1.0),
            'Best Val Acc': res.get('best_val_acc', 0) * 100,
            'Final Val Acc': res.get('final_val_acc', 0) * 100,
            'Final Train Loss': res.get('final_train_loss', 0),
        })
    
    df = pd.DataFrame(summary_data)
    return df.sort_values(['Dataset', 'Model', 'Softmax'])


def pairwise_comparison(df_summary):
    """Generate pairwise comparison."""
    grouped = df_summary.groupby(['Dataset', 'Model'])
    
    comparison_data = []
    for (dataset, model), group in grouped:
        if len(group) == 2:
            std_row = group[group['Softmax'] == 'standard'].iloc[0]
            b2_row = group[group['Softmax'] == 'base2'].iloc[0]
            
            diff = b2_row['Best Val Acc'] - std_row['Best Val Acc']
            
            comparison_data.append({
                'Experiment': f"{model} on {dataset}",
                'Standard Acc': f"{std_row['Best Val Acc']:.2f}%",
                'Base-2 Acc': f"{b2_row['Best Val Acc']:.2f}%",
                'Difference': f"{diff:+.2f}%",
                'Winner': 'Base-2' if diff > 0.1 else ('Standard' if diff < -0.1 else 'Tie')
            })
    
    return pd.DataFrame(comparison_data)


def statistical_analysis(df_summary):
    """Perform statistical analysis."""
    std_group = df_summary[df_summary['Softmax'] == 'standard'].sort_values(['Dataset', 'Model'])
    base2_group = df_summary[df_summary['Softmax'] == 'base2'].sort_values(['Dataset', 'Model'])
    
    if len(std_group) != len(base2_group):
        print("Warning: Unequal number of experiments for standard and base2")
        return None
    
    std_accs = std_group['Best Val Acc'].values
    base2_accs = base2_group['Best Val Acc'].values
    
    differences = base2_accs - std_accs
    
    # T-test
    t_stat, p_value = stats.ttest_rel(base2_accs, std_accs)
    
    results = {
        'mean_diff': differences.mean(),
        'std_diff': differences.std(),
        'min_diff': differences.min(),
        'max_diff': differences.max(),
        'median_diff': np.median(differences),
        'wins_base2': sum(differences > 0),
        'wins_standard': sum(differences < 0),
        'ties': sum(differences == 0),
        't_statistic': t_stat,
        'p_value': p_value,
    }
    
    return results


def main():
    results_dir = Path("results")
    
    print("\n" + "="*80)
    print("LOADING EXPERIMENTAL RESULTS")
    print("="*80 + "\n")
    
    all_results = load_results(results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    df_summary = create_summary_dataframe(all_results)
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80 + "\n")
    
    df_comparison = pairwise_comparison(df_summary)
    
    print("\n" + "="*80)
    print("PAIRWISE COMPARISON: Base-2 vs Standard")
    print("="*80)
    print(df_comparison.to_string(index=False))
    print("="*80 + "\n")
    
    stats_results = statistical_analysis(df_summary)
    
    if stats_results:
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)
        print(f"Mean difference (Base-2 - Standard): {stats_results['mean_diff']:.3f}%")
        print(f"Std deviation of difference: {stats_results['std_diff']:.3f}%")
        print(f"Min difference: {stats_results['min_diff']:.3f}%")
        print(f"Max difference: {stats_results['max_diff']:.3f}%")
        print(f"Median difference: {stats_results['median_diff']:.3f}%")
        print(f"\nBase-2 wins: {stats_results['wins_base2']}")
        print(f"Standard wins: {stats_results['wins_standard']}")
        print(f"Ties: {stats_results['ties']}")
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {stats_results['t_statistic']:.4f}")
        print(f"  p-value: {stats_results['p_value']:.4f}")
        if stats_results['p_value'] < 0.05:
            print(f"  Result: Statistically significant difference (p < 0.05)")
        else:
            print(f"  Result: No statistically significant difference (p >= 0.05)")
        print("="*80 + "\n")
    
    # Save summaries
    df_summary.to_csv(results_dir / "experiment_summary.csv", index=False)
    df_comparison.to_csv(results_dir / "pairwise_comparison.csv", index=False)
    
    print("✓ Summary exported to results/experiment_summary.csv")
    print("✓ Pairwise comparison exported to results/pairwise_comparison.csv\n")


if __name__ == "__main__":
    main()


