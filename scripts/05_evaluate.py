import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

true_props = pd.read_csv(os.path.join(DATA_DIR, 'true_proportions.csv'), index_col=0)
benchmarks = pd.read_csv(os.path.join(RESULTS_DIR, 'benchmarks.csv'), index_col=0)

methods = ['nnls', 'svr', 'robust_regression']
method_labels = ['NNLS', 'SVR', 'Robust Regression']

predictions = {}
for m, label in zip(methods, method_labels):
    predictions[label] = pd.read_csv(os.path.join(RESULTS_DIR, f'predictions_{m}.csv'), index_col=0)

metrics = {}
for label, pred_df in predictions.items():
    true_flat = true_props.values.flatten()
    pred_flat = pred_df.values.flatten()

    pearson_r, _ = pearsonr(true_flat, pred_flat)
    spearman_r, _ = spearmanr(true_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(true_flat, pred_flat))
    mae = np.mean(np.abs(true_flat - pred_flat))

    per_type_pearson = {}
    for ct in true_props.columns:
        r, _ = pearsonr(true_props[ct].values, pred_df[ct].values)
        per_type_pearson[ct] = round(r, 4)

    metrics[label] = {
        'Pearson r': round(pearson_r, 4),
        'Spearman r': round(spearman_r, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'per_type_pearson': per_type_pearson,
    }

print("=== Overall Accuracy Metrics ===")
summary = pd.DataFrame({m: {k: v for k, v in vals.items() if k != 'per_type_pearson'}
                         for m, vals in metrics.items()}).T
print(summary.to_string())
summary.to_csv(os.path.join(RESULTS_DIR, 'accuracy_metrics.csv'))

print("\n=== Per Cell-Type Pearson Correlation ===")
per_type_df = pd.DataFrame({m: vals['per_type_pearson'] for m, vals in metrics.items()})
print(per_type_df.to_string())
per_type_df.to_csv(os.path.join(RESULTS_DIR, 'per_celltype_correlation.csv'))

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (label, pred_df) in enumerate(predictions.items()):
    ax = axes[idx]
    true_flat = true_props.values.flatten()
    pred_flat = pred_df.values.flatten()
    ax.scatter(true_flat, pred_flat, alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
    ax.set_xlabel('True Proportion')
    ax.set_ylabel('Predicted Proportion')
    ax.set_title(f'{label}\n(r={metrics[label]["Pearson r"]:.3f}, RMSE={metrics[label]["RMSE"]:.3f})')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'correlation_plots.png'), bbox_inches='tight')
plt.close()
print("\nSaved: correlation_plots.png")

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(method_labels))
width = 0.2
ax.bar(x - width, [metrics[m]['Pearson r'] for m in method_labels], width, label='Pearson r')
ax.bar(x, [metrics[m]['Spearman r'] for m in method_labels], width, label='Spearman r')
ax.bar(x + width, [1 - metrics[m]['RMSE'] for m in method_labels], width, label='1 - RMSE')
ax.set_xlabel('Method')
ax.set_ylabel('Score')
ax.set_title('Accuracy Metrics by Method')
ax.set_xticks(x)
ax.set_xticklabels(method_labels)
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_comparison.png'), bbox_inches='tight')
plt.close()
print("Saved: accuracy_comparison.png")

fig, ax = plt.subplots(figsize=(10, 5))
per_type_df.plot(kind='bar', ax=ax)
ax.set_xlabel('Cell Type')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Per Cell-Type Correlation by Method')
ax.legend(title='Method')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_celltype_comparison.png'), bbox_inches='tight')
plt.close()
print("Saved: per_celltype_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(method_labels, benchmarks['runtime_seconds'], color=['#2196F3', '#FF9800', '#4CAF50'])
axes[0].set_ylabel('Runtime (seconds)')
axes[0].set_title('Computational Runtime')
axes[1].bar(method_labels, benchmarks['peak_memory_mb'], color=['#2196F3', '#FF9800', '#4CAF50'])
axes[1].set_ylabel('Peak Memory (MB)')
axes[1].set_title('Memory Usage')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'runtime_memory_benchmarks.png'), bbox_inches='tight')
plt.close()
print("Saved: runtime_memory_benchmarks.png")

print("\nEvaluation complete! All results saved to results/")
