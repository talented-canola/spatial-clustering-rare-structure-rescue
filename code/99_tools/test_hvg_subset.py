"""Diagnostic: does HVG-subset (the fix) materially change baseline_v2 at res=1.0?"""
import time
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
t0 = time.time()

adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
n_hvg = int(adata.var.highly_variable.sum())
print(f"genes before subset: {adata.n_vars}, HVG marked: {n_hvg}")

# THE FIX: subset to HVG
adata = adata[:, adata.var.highly_variable].copy()
print(f"genes after subset: {adata.n_vars}")

sc.pp.scale(adata, max_value=10)
ann = adata.obs["annotation"].astype(str).values

_, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=30)
conn = adata.obsp["expression_connectivities"].copy()
conn.data[conn.data > 0] = 1

print("\n=== HVG-subset baseline_v2 (arpack-30, expression-only) ===")
for res in [0.5, 1.0, 1.5, 2.0]:
    lab = fl.fast_leiden(conn, res)
    ari = adjusted_rand_score(ann, lab.astype(str))
    nmi = normalized_mutual_info_score(ann, lab.astype(str))
    print(f"res={res}: n_clusters={len(np.unique(lab)):3d}  ARI={ari:.4f}  NMI={nmi:.4f}")

print(f"\n(ref) baseline_v2 WITHOUT subset, res=1.0: ARI=0.4234 NMI=0.6987 n_clusters=28")
print(f"DONE {time.time()-t0:.0f}s")
