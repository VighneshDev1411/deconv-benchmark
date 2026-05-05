import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("Loading preprocessed data...")
expr = pd.read_csv(os.path.join(DATA_DIR, 'expression_filtered.csv'), index_col=0)
meta = pd.read_csv(os.path.join(DATA_DIR, 'metadata_aligned.csv'), index_col=0)
signature = pd.read_csv(os.path.join(DATA_DIR, 'signature_matrix.csv'), index_col=0)

cell_types = signature.columns.tolist()
n_types = len(cell_types)

np.random.seed(42)
n_mixtures = 50

true_proportions = []
for i in range(n_mixtures):
    props = np.random.dirichlet(np.ones(n_types))
    true_proportions.append(props)

true_proportions = np.array(true_proportions)
true_df = pd.DataFrame(true_proportions, columns=cell_types,
                        index=[f'mixture_{i}' for i in range(n_mixtures)])

print(f"Generating {n_mixtures} pseudo-bulk mixtures from {n_types} cell types...")

type_samples = {}
for ct in cell_types:
    samples = meta[meta['major_type'] == ct].index
    samples = [s for s in samples if s in expr.columns]
    if samples:
        type_samples[ct] = samples

mixtures = pd.DataFrame(index=expr.index)
for i in range(n_mixtures):
    mix = np.zeros(len(expr.index))
    for j, ct in enumerate(cell_types):
        if ct in type_samples:
            sample = np.random.choice(type_samples[ct])
            profile = expr[sample].values
            noise = np.random.normal(0, 0.1 * np.std(profile), len(profile))
            mix += true_proportions[i, j] * (profile + noise)
    mix = np.maximum(mix, 0)
    mixtures[f'mixture_{i}'] = mix

print(f"Pseudo-bulk mixture matrix shape: {mixtures.shape}")
print(f"True proportions shape: {true_df.shape}")
print(f"\nSample true proportions (mixture_0):")
print(true_df.iloc[0].round(3))

mixtures.to_csv(os.path.join(DATA_DIR, 'pseudobulk_mixtures.csv'))
true_df.to_csv(os.path.join(DATA_DIR, 'true_proportions.csv'))

print("\nPseudo-bulk generation complete!")
