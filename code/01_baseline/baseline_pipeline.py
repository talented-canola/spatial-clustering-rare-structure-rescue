"""Baseline (non-spatial) clustering on MOSTA E11.5, driven through spateo's own
functions for the spateo-specific steps (PCA / neighbor graph / Leiden), and
scanpy for the generic single-cell steps (HVG / scale) that spateo itself wraps.

Pipeline: load -> QC -> HVG -> scale -> PCA(spateo) -> KNN(spateo) -> Leiden(spateo)
"""
import time
import numpy as np
import scanpy as sc
import spateo_loader as sl  # surgical import of spateo.tools.{utils,leiden,find_neighbors}

t0 = time.time()

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
OUT = r"F:/BGI/task3/baseline_results"

import os
os.makedirs(OUT, exist_ok=True)

sc.settings.verbosity = 1

print("[1/7] loading ...", flush=True)
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"  # spateo decorator requirement
print(f"      raw: {adata.shape[0]} bins x {adata.shape[1]} genes", flush=True)

print("[2/7] light QC ...", flush=True)
n0 = adata.shape
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
print(f"      {n0} -> {adata.shape}", flush=True)

print("[3/7] HVG (top 3000) ...", flush=True)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
n_hvg = int(adata.var["highly_variable"].sum())
print(f"      {n_hvg} HVGs", flush=True)

print("[4/7] scale ...", flush=True)
sc.pp.scale(adata, max_value=10)

print("[5/7] PCA via spateo pca_spateo (50 comps) ...", flush=True)
adata = sl.utils.pca_spateo(adata, n_pca_components=50, pca_key="X_pca")
print(f"      X_pca shape: {adata.obsm['X_pca'].shape}", flush=True)

print("[6/7] KNN graph via spateo find_neighbors.neighbors (basis=pca, k=30) ...", flush=True)
_, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=50)
print(f"      expression_connectivities: {adata.obsp['expression_connectivities'].shape}", flush=True)

print("[7/7] Leiden (reproduces spateo calculate_leiden_partition: leidenalg RBConfiguration, seed=888, n_iter=-1) ...", flush=True)
import fast_leiden as fl
adj = adata.obsp["expression_connectivities"]
for res in [0.5, 1.0, 1.5, 2.0]:
    t1 = time.time()
    labels = fl.fast_leiden(adj, res)
    key = f"baseline_leiden_r{res}"
    adata.obs[key] = labels.astype(str)
    print(f"      resolution={res}: {len(np.unique(labels))} clusters  ({time.time()-t1:.1f}s)", flush=True)

# Slim results object for viz/eval (drop count layer + Module/Regulon columns to keep it small)
print("saving slim results ...", flush=True)
keep_obs = [c for c in adata.obs.columns
            if c == "annotation" or c.startswith("baseline_leiden")]
slim = sc.AnnData(X=adata[:, adata.var["highly_variable"]].X.copy(),
                   obs=adata.obs[keep_obs].copy())
slim.obsm["spatial"] = adata.obsm["spatial"].copy()
slim.obsm["X_pca"] = adata.obsm["X_pca"].copy()
slim.obsp["expression_connectivities"] = adata.obsp["expression_connectivities"].copy()
slim.uns["annotation_colors"] = np.array(adata.uns.get("annotation_colors", []), dtype=object)
slim.var_names = adata.var_names[adata.var["highly_variable"]].copy()
slim.write_h5ad(os.path.join(OUT, "baseline_slim.h5ad"))

print(f"\nDONE in {time.time()-t0:.1f}s. Saved to {OUT}", flush=True)
print("cluster counts:", {c: adata.obs[c].nunique() for c in keep_obs if c != "annotation"}, flush=True)
