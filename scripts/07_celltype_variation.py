import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn.svm import NuSVR
from sklearn.linear_model import HuberRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

expr = pd.read_csv(os.path.join(DATA_DIR, 'expression_filtered.csv'), index_col=0)
meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata_aligned.csv'), index_col=0)
signature = pd.read_csv(os.path.join(DATA_DIR, 'signature_matrix.csv'), index_col=0)

all_cell_types = signature.columns.tolist()
n_mixtures = 50
noise_scale = 0.1

type_samples = {}
for ct in all_cell_types:
    samples = meta[meta['major_type'] == ct].index
    samples = [s for s in samples if s in expr.columns]
    if samples:
        type_samples[ct] = samples


def run_nnls(mixtures_mat, S):
    results = []
    for i in range(mixtures_mat.shape[1]):
        coeffs, _ = nnls(S, mixtures_mat[:, i])
        coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
        results.append(coeffs)
    return np.array(results)


def run_svr(mixtures_mat, S):
    results = []
    for i in range(mixtures_mat.shape[1]):
        svr = NuSVR(kernel='linear', nu=0.5, C=1.0)
        svr.fit(S, mixtures_mat[:, i])
        coeffs = np.maximum(svr.coef_.flatten(), 0)
        coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
        results.append(coeffs)
    return np.array(results)


def run_robust(mixtures_mat, S):
    results = []
    for i in range(mixtures_mat.shape[1]):
        model = HuberRegressor(fit_intercept=False, max_iter=1000)
        model.fit(S, mixtures_mat[:, i])
        coeffs = np.maximum(model.coef_, 0)
        coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
        results.append(coeffs)
    return np.array(results)


methods = {'NNLS': run_nnls, 'SVR': run_svr, 'Robust Regression': run_robust}
n_type_configs = [3, 4, 5, 6, 7, 8]

all_results = []

for n_types in n_type_configs:
    np.random.seed(42)
    selected_types = all_cell_types[:n_types]
    selected_types_available = [ct for ct in selected_types if ct in type_samples]

    if len(selected_types_available) < n_types:
        continue

    S_sub = signature[selected_types_available].values
    true_proportions = np.array([np.random.dirichlet(np.ones(len(selected_types_available))) for _ in range(n_mixtures)])

    mixtures = np.zeros((len(expr.index), n_mixtures))
    for i in range(n_mixtures):
        mix = np.zeros(len(expr.index))
        for j, ct in enumerate(selected_types_available):
            sample = np.random.choice(type_samples[ct])
            profile = expr[sample].values
            noise = np.random.normal(0, noise_scale * np.std(profile), len(profile))
            mix += true_proportions[i, j] * (profile + noise)
        mixtures[:, i] = np.maximum(mix, 0)

    print(f"\nCell types: {n_types} ({', '.join(selected_types_available)})")

    for name, method in methods.items():
        preds = method(mixtures, S_sub)
        r, _ = pearsonr(true_proportions.flatten(), preds.flatten())
        rmse = np.sqrt(mean_squared_error(true_proportions.flatten(), preds.flatten()))
        all_results.append({
            'n_cell_types': n_types,
            'method': name,
            'pearson_r': round(r, 4),
            'rmse': round(rmse, 4),
        })
        print(f"  {name}: r={r:.4f}, RMSE={rmse:.4f}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'celltype_variation.csv'), index=False)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, metric in enumerate(['pearson_r', 'rmse']):
    ax = axes[idx]
    for method in methods:
        subset = results_df[results_df['method'] == method]
        ax.plot(subset['n_cell_types'], subset[metric], marker='o', label=method, linewidth=2)
    ax.set_xlabel('Number of Cell Types')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} vs Number of Cell Types')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'celltype_variation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: celltype_variation.png")
print("Cell type variation analysis complete!")
