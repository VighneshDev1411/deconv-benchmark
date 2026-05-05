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

cell_types = signature.columns.tolist()
n_mixtures = 50
noise_scale = 0.1
np.random.seed(42)

type_samples = {}
for ct in cell_types:
    samples = meta[meta['major_type'] == ct].index
    samples = [s for s in samples if s in expr.columns]
    if samples:
        type_samples[ct] = samples


def select_top_variance(signature, n_genes):
    gene_var = signature.var(axis=1)
    return gene_var.nlargest(n_genes).index


def select_top_differential(signature, n_genes):
    scores = pd.Series(index=signature.index, dtype=float)
    for gene in signature.index:
        vals = signature.loc[gene]
        max_val = vals.max()
        second_max = vals.nlargest(2).iloc[1]
        scores[gene] = max_val - second_max
    return scores.nlargest(n_genes).index


def select_top_foldchange(signature, n_genes):
    scores = pd.Series(index=signature.index, dtype=float)
    for gene in signature.index:
        vals = signature.loc[gene]
        mean_val = vals.mean()
        if mean_val > 0:
            scores[gene] = vals.max() / (mean_val + 1e-10)
        else:
            scores[gene] = 0
    return scores.nlargest(n_genes).index


strategies = {
    'Top Variance': select_top_variance,
    'Top Differential': select_top_differential,
    'Top Fold Change': select_top_foldchange,
}


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
gene_counts = [100, 250, 500, 1000, 2000, 5000]

all_results = []

for strategy_name, strategy_fn in strategies.items():
    print(f"\n=== Strategy: {strategy_name} ===")
    for n_genes in gene_counts:
        np.random.seed(42)
        selected_genes = strategy_fn(signature, min(n_genes, len(signature)))
        sig_sub = signature.loc[selected_genes]
        expr_sub = expr.loc[selected_genes]

        true_proportions = np.array([np.random.dirichlet(np.ones(len(cell_types))) for _ in range(n_mixtures)])
        mixtures = np.zeros((len(selected_genes), n_mixtures))
        for i in range(n_mixtures):
            mix = np.zeros(len(selected_genes))
            for j, ct in enumerate(cell_types):
                if ct in type_samples:
                    sample = np.random.choice(type_samples[ct])
                    profile = expr_sub[sample].values
                    noise = np.random.normal(0, noise_scale * np.std(profile), len(profile))
                    mix += true_proportions[i, j] * (profile + noise)
            mixtures[:, i] = np.maximum(mix, 0)

        S_sub = sig_sub.values
        print(f"  {n_genes} genes:")

        for name, method in methods.items():
            preds = method(mixtures, S_sub)
            r, _ = pearsonr(true_proportions.flatten(), preds.flatten())
            rmse = np.sqrt(mean_squared_error(true_proportions.flatten(), preds.flatten()))
            all_results.append({
                'strategy': strategy_name,
                'n_genes': n_genes,
                'method': name,
                'pearson_r': round(r, 4),
                'rmse': round(rmse, 4),
            })
            print(f"    {name}: r={r:.4f}, RMSE={rmse:.4f}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'marker_gene_analysis.csv'), index=False)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (strategy_name, _) in enumerate(strategies.items()):
    ax = axes[idx]
    subset = results_df[results_df['strategy'] == strategy_name]
    for method in methods:
        method_data = subset[subset['method'] == method]
        ax.plot(method_data['n_genes'], method_data['pearson_r'], marker='o', label=method, linewidth=2)
    ax.set_xlabel('Number of Marker Genes')
    ax.set_ylabel('Pearson r')
    ax.set_title(f'{strategy_name}')
    ax.legend()
    ax.set_ylim(0.5, 1.0)

plt.suptitle('Effect of Marker Gene Selection Strategy on Deconvolution Accuracy', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'marker_gene_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
pivot = results_df[results_df['n_genes'] == 500].pivot(index='method', columns='strategy', values='pearson_r')
pivot.plot(kind='bar', ax=ax)
ax.set_ylabel('Pearson r')
ax.set_title('Strategy Comparison at 500 Marker Genes')
ax.legend(title='Strategy')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'strategy_comparison_500genes.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nSaved: marker_gene_analysis.png, strategy_comparison_500genes.png")
print("Marker gene analysis complete!")
