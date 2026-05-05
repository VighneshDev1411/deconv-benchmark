# Benchmarking Regression-Based Cell Type Deconvolution Methods for Bulk RNA-seq Data

**Course:** CS 502 — University of Illinois Chicago  
**Author:** Vignesh Pathak

---

## Overview

This project benchmarks three regression-based algorithms for reference-based cell type deconvolution of bulk RNA-seq data:

| Method | Description |
|---|---|
| **NNLS** | Non-Negative Least Squares — convex optimization with non-negativity constraint |
| **SVR** | Support Vector Regression with linear kernel (nu-SVR) |
| **Robust Regression** | Huber-loss regression, down-weights outlier genes |

Evaluation is performed on synthetic pseudo-bulk mixtures derived from **8 immune cell types** using the GEO dataset [GSE107011](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE107011).

---

## Key Findings

- All three methods achieve strong accuracy (Pearson *r* > 0.94) with differences < 0.003 across methods
- **NNLS is recommended** for production pipelines — highest accuracy, most noise-stable, and ~3,000× faster than SVR
- Marker gene selection strategy and cell type transcriptomic distinctness matter more than the choice of regression algorithm
- Top Differential Expression outperforms variance- and fold-change-based selection at low gene counts (< 500 genes)

---

## Repository Structure

```
├── scripts/
│   ├── 01_download_data.py          # Download GSE107011 from GEO
│   ├── 02_preprocess.py             # Filter, log2-transform, select top 5000 genes
│   ├── 03_generate_pseudobulk.py    # Generate 50 pseudo-bulk mixtures (Dirichlet)
│   ├── 04_deconvolution.py          # Run NNLS, SVR, Robust Regression
│   ├── 05_evaluate.py               # Compute Pearson r, RMSE, MAE, Spearman rho
│   ├── 06_noise_robustness.py       # Sweep noise level 0–100%
│   ├── 07_celltype_variation.py     # Sweep number of cell types 3–8
│   └── 08_marker_gene_analysis.py   # Compare gene selection strategies
│
├── results/
│   ├── correlation_plots.png
│   ├── per_celltype_comparison.png
│   ├── noise_robustness.png
│   ├── celltype_variation.png
│   ├── marker_gene_analysis.png
│   ├── accuracy_metrics.csv
│   ├── benchmarks.csv
│   └── ...
│
├── final_report_cs502.tex             # OUP Bioinformatics-style LaTeX report
└── final_report_cs502.pdf             # Compiled PDF
```

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib seaborn
```

Python 3.10+

---

## Reproducing the Results

Run scripts in order from the `scripts/` directory:

```bash
python scripts/01_download_data.py
python scripts/02_preprocess.py
python scripts/03_generate_pseudobulk.py
python scripts/04_deconvolution.py
python scripts/05_evaluate.py
python scripts/06_noise_robustness.py
python scripts/07_celltype_variation.py
python scripts/08_marker_gene_analysis.py
```

> **Note:** The raw GEO data (~111 MB) is not included in this repository. Run `01_download_data.py` to fetch it automatically.

---

## Results Summary

| Method | Pearson *r* | RMSE | Runtime (50 samples) |
|---|---|---|---|
| NNLS | **0.9436** | **0.0367** | 0.04 s |
| SVR | 0.9419 | 0.0371 | 118.8 s |
| Robust Regression | 0.9413 | 0.0374 | 1.47 s |

---

## Dataset

**GSE107011** — Human peripheral blood immune cell expression profiles  
127 sorted samples · 30 fine-grained subtypes · grouped into 8 major cell types

---

## Report

The full report is available as [`final_report_cs502.pdf`](final_report_cs502.pdf), formatted according to the OUP *Bioinformatics* journal style.
