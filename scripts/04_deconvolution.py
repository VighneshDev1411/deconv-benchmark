import pandas as pd
import numpy as np
from scipy.optimize import nnls
from sklearn.svm import SVR, NuSVR
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import time
import os
import tracemalloc

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading data...")
mixtures = pd.read_csv(os.path.join(DATA_DIR, 'pseudobulk_mixtures.csv'), index_col=0)
signature = pd.read_csv(os.path.join(DATA_DIR, 'signature_matrix.csv'), index_col=0)
true_props = pd.read_csv(os.path.join(DATA_DIR, 'true_proportions.csv'), index_col=0)

common_genes = mixtures.index.intersection(signature.index)
mixtures = mixtures.loc[common_genes]
signature = signature.loc[common_genes]

S = signature.values
cell_types = signature.columns.tolist()
sample_names = mixtures.columns.tolist()


def run_nnls_deconv(mixtures_mat, S):
    results = []
    for i in range(mixtures_mat.shape[1]):
        b = mixtures_mat[:, i]
        coeffs, _ = nnls(S, b)
        coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
        results.append(coeffs)
    return np.array(results)


def run_svr_deconv(mixtures_mat, S):
    results = []
    for i in range(mixtures_mat.shape[1]):
        b = mixtures_mat[:, i]
        svr = NuSVR(kernel='linear', nu=0.5, C=1.0)
        svr.fit(S, b)
        coeffs = np.maximum(svr.coef_.flatten(), 0)
        coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
        results.append(coeffs)
    return np.array(results)


def run_robust_regression_deconv(mixtures_mat, S):
    results = []
    for i in range(mixtures_mat.shape[1]):
        b = mixtures_mat[:, i]
        model = HuberRegressor(fit_intercept=False, max_iter=1000)
        model.fit(S, b)
        coeffs = np.maximum(model.coef_, 0)
        coeffs = coeffs / coeffs.sum() if coeffs.sum() > 0 else coeffs
        results.append(coeffs)
    return np.array(results)


methods = {
    'NNLS': run_nnls_deconv,
    'SVR': run_svr_deconv,
    'Robust Regression': run_robust_regression_deconv,
}

benchmarks = {}
all_predictions = {}

for name, method in methods.items():
    print(f"\nRunning {name}...")

    tracemalloc.start()
    start_time = time.time()
    preds = method(mixtures.values, S)
    elapsed = time.time() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pred_df = pd.DataFrame(preds, columns=cell_types, index=sample_names)
    pred_df.to_csv(os.path.join(RESULTS_DIR, f'predictions_{name.lower().replace(" ", "_")}.csv'))

    benchmarks[name] = {
        'runtime_seconds': round(elapsed, 3),
        'peak_memory_mb': round(peak_memory / 1024 / 1024, 2),
    }
    all_predictions[name] = pred_df

    print(f"  Runtime: {elapsed:.3f}s | Peak memory: {peak_memory/1024/1024:.2f} MB")

bench_df = pd.DataFrame(benchmarks).T
bench_df.to_csv(os.path.join(RESULTS_DIR, 'benchmarks.csv'))

print("\n--- Computational Benchmarks ---")
print(bench_df.to_string())
print("\nDeconvolution complete!")
