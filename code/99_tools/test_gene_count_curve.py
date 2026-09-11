"""Sweep gene count (3000/5000/8000/all) to see ARI/NMI trend vs HVG subset size.
Same scale for all: sc.pp.scale(zero_center=True, max_value=10)."""
import time
import scanpy as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
t0 = time.time()

base = sc.read_h5ad(DATA)
base.uns["__type"] = "UMI"
sc.pp.filter_genes(base, min_cells=3)
sc.pp.filter_cells(base, min_genes=50)
print(f"QC'd: {base.shape}")

ann = base.obs["annotation"].astype(str).values

print("\n=== gene-count sweep (arpack-30, expression-only, scale zero_center=True max_value=10) ===")
print(f"{'n_genes':>8} {'res':>5} {'n_clusters':>10} {'ARI':>8} {'NMI':>8}")
rows = []
for n_top in [3000, 5000, 8000, None]:
    label = "all" if n_top is None else str(n_top)
    if n_top is None:
        a = base.copy()
    else:
        sc.pp.highly_variable_genes(base, n_top_genes=n_top, flavor="seurat")
        a = base[:, base.var.highly_variable].copy()
    sc.pp.scale(a, max_value=10, zero_center=True)  # identical scale across all
    _, a = sl.find_neighbors.neighbors(a, n_neighbors=30, basis="pca", n_pca_components=30)
    conn = a.obsp["expression_connectivities"].copy()
    conn.data[conn.data > 0] = 1
    for res in [0.5, 1.0, 1.5, 2.0]:
        lab = fl.fast_leiden(conn, res)
        ari = adjusted_rand_score(ann, lab.astype(str))
        nmi = normalized_mutual_info_score(ann, lab.astype(str))
        rows.append((label, res, len(np.unique(lab)), round(ari, 4), round(nmi, 4)))
        print(f"{label:>8} {res:>5} {len(np.unique(lab)):>10} {ari:>8.4f} {nmi:>8.4f}")
    print()

print(f"DONE {time.time()-t0:.0f}s")
