import GEOparse
import pandas as pd
import os
import urllib.request
import gzip
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

print("Downloading GSE107011...")
gse = GEOparse.get_GEO(geo="GSE107011", destdir=DATA_DIR, silent=True)

suppl_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE107nnn/GSE107011/suppl/GSE107011_Processed_data_TPM.txt.gz"
gz_path = os.path.join(DATA_DIR, "GSE107011_Processed_data_TPM.txt.gz")
txt_path = os.path.join(DATA_DIR, "GSE107011_Processed_data_TPM.txt")

if not os.path.exists(txt_path):
    print("Downloading supplementary expression file...")
    urllib.request.urlretrieve(suppl_url, gz_path)
    with gzip.open(gz_path, 'rb') as f_in:
        with open(txt_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

print("Reading expression matrix...")
expr_matrix = pd.read_csv(txt_path, sep='\t', index_col=0)
print(f"Expression matrix shape: {expr_matrix.shape}")
print(f"First few column names: {list(expr_matrix.columns[:5])}")

metadata = {}
for gsm_name, gsm in gse.gsms.items():
    chars = gsm.metadata.get('characteristics_ch1', [])
    cell_type = None
    for c in chars:
        if 'cell type' in c.lower():
            cell_type = c.split(':')[-1].strip()
            break
    metadata[gsm_name] = {
        'cell_type': cell_type,
        'title': gsm.metadata.get('title', [''])[0]
    }

meta_df = pd.DataFrame(metadata).T
meta_df.index.name = 'sample_id'

print(f"Number of samples: {len(meta_df)}")
print(f"Cell types found ({meta_df['cell_type'].nunique()}): {meta_df['cell_type'].unique()}")

expr_matrix.to_csv(os.path.join(DATA_DIR, 'expression_matrix.csv'))
meta_df.to_csv(os.path.join(DATA_DIR, 'sample_metadata.csv'))

print(f"\nData saved to {DATA_DIR}")
print("Done!")
