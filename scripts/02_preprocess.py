import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

print("Loading data...")
expr = pd.read_csv(os.path.join(DATA_DIR, 'expression_matrix.csv'), index_col=0)
meta = pd.read_csv(os.path.join(DATA_DIR, 'sample_metadata.csv'), index_col=0)

print(f"Raw expression matrix: {expr.shape}")

sample_map = {}
for _, row in meta.iterrows():
    title = row['title']
    matches = [c for c in expr.columns if c.startswith(title.rsplit('_rep', 1)[0]) or c == title]
    if matches:
        sample_map[matches[0]] = row.name

common_cols = [c for c in expr.columns if any(c.startswith(row['title'].rsplit('_rep', 1)[0]) for _, row in meta.iterrows())]

expr = expr[common_cols] if common_cols else expr

expr = expr[expr.sum(axis=1) > 0]
print(f"After removing zero-expression genes: {expr.shape}")

expr = np.log2(expr + 1)

gene_var = expr.var(axis=1)
top_genes = gene_var.nlargest(5000).index
expr_filtered = expr.loc[top_genes]
print(f"After selecting top 5000 variable genes: {expr_filtered.shape}")

col_to_celltype = {}
for col in expr_filtered.columns:
    for _, row in meta.iterrows():
        title_base = row['title'].rsplit('_rep', 1)[0]
        if col.startswith(title_base):
            col_to_celltype[col] = row['cell_type']
            break

meta_aligned = pd.DataFrame({
    'sample': list(col_to_celltype.keys()),
    'cell_type': list(col_to_celltype.values())
}).set_index('sample')

print(f"Matched {len(meta_aligned)} samples to cell types")

major_types = {
    'Naive CD8 T cells': 'T cells',
    'Central memory CD8 T cell': 'T cells',
    'Effector memory CD8 T cells': 'T cells',
    'Terminal effector CD8 T cells': 'T cells',
    'MAIT cells': 'T cells',
    'Vd2 gd T cells': 'T cells',
    'Non-Vd2 gd T cells': 'T cells',
    'Follicular helper T cells': 'T cells',
    'T regulatory cells': 'T cells',
    'Th1 cells': 'T cells',
    'Th1/Th17 cells': 'T cells',
    'Th17 cells': 'T cells',
    'Th2 cells': 'T cells',
    'Naive CD4 T cells': 'T cells',
    'Terminal effector CD4 T cells': 'T cells',
    'Naive B cells': 'B cells',
    'Non-switched memory B cells': 'B cells',
    'Exhausted B cells': 'B cells',
    'Switched memory B cells': 'B cells',
    'Plasmablasts': 'B cells',
    'Classical monocytes': 'Monocytes',
    'Intermediate monocytes': 'Monocytes',
    'Non classical monocytes': 'Monocytes',
    'Natural killer cells': 'NK cells',
    'Plasmacytoid dendritic cells': 'Dendritic cells',
    'Myeloid dendritic cells': 'Dendritic cells',
    'Low-density neutrophils': 'Neutrophils',
    'Low-density basophils': 'Basophils',
    'Progenitor cells': 'Progenitor cells',
    'PBMCs': 'PBMCs',
}

meta_aligned['major_type'] = meta_aligned['cell_type'].map(major_types)
meta_aligned = meta_aligned.dropna(subset=['major_type'])
meta_aligned = meta_aligned[meta_aligned['major_type'] != 'PBMCs']

expr_filtered = expr_filtered[meta_aligned.index]
print(f"After grouping into major types and removing PBMCs: {expr_filtered.shape}")
print(f"Major cell types: {meta_aligned['major_type'].unique()}")

signature = expr_filtered.T.copy()
signature['cell_type'] = meta_aligned['major_type']
signature = signature.groupby('cell_type').mean().T
print(f"Signature matrix shape: {signature.shape}")

expr_filtered.to_csv(os.path.join(DATA_DIR, 'expression_filtered.csv'))
meta_aligned.to_csv(os.path.join(DATA_DIR, 'metadata_aligned.csv'))
signature.to_csv(os.path.join(DATA_DIR, 'signature_matrix.csv'))

print("Preprocessing complete!")
