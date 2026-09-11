"""Corrected baseline (arpack PCA) + ablation verify + per-tissue overlap compare vs old."""
import time, os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/baseline"
os.makedirs(OUT, exist_ok=True)
RES = [0.5, 1.0, 1.5, 2.0]
TARGET = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity"]

t0 = time.time()
print("preprocess ...", flush=True)
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
sc.pp.scale(adata, max_value=10)
ann = adata.obs["annotation"].astype(str).values
ann_cats = list(adata.obs["annotation"].cat.categories)
xy = adata.obsm["spatial"]

def leiden_all(conn):
    return {res: fl.fast_leiden(conn, res) for res in RES}

# corrected baseline: arpack PCA via neighbors (n_pca=30), expression-only
print("corrected baseline n_pca=30 ...", flush=True)
_, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=30)
conn30 = adata.obsp["expression_connectivities"].copy()
conn30.data[conn30.data > 0] = 1
corr30 = leiden_all(conn30)
X_pca30 = adata.obsm["X_pca"].copy()

# corrected baseline n_pca=50 (diagnostic: isolate solver vs n_pca)
print("corrected baseline n_pca=50 ...", flush=True)
if "X_pca" in adata.obsm:
    del adata.obsm["X_pca"]
_, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=50)
conn50 = adata.obsp["expression_connectivities"].copy()
conn50.data[conn50.data > 0] = 1
corr50 = leiden_all(conn50)

# ---- ablation verify: spatial_adj's expression term must equal corrected baseline ----
print("ablation verify (spatial_adj expression term vs corrected baseline) ...", flush=True)
if "X_pca" in adata.obsm:
    del adata.obsm["X_pca"]
_ = sl.utils.spatial_adj(adata, spatial_key="spatial", pca_key="pca", e_neigh=30, s_neigh=6, n_pca_components=30)
expr_of_spatial_adj = adata.obsp["expression_connectivities"]
print(f"    expression conn identical to corrected30: {(expr_of_spatial_adj != conn30).nnz == 0}")
ab = leiden_all(expr_of_spatial_adj)
for res in RES:
    ari = adjusted_rand_score(ab[res].astype(str), corr30[res].astype(str))
    print(f"    res={res}: ARI(ablation, corrected30) = {ari:.4f}")

# ---- per-tissue hungarian overlap ----
def overlap(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    n = max(n_ann, n_cl)
    Cpad = np.zeros((n, n)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    ann2cl = {ann_cats[i]: ct.columns[ci[i]] for i in range(n_ann)}
    out = {}
    for a in ann_cats:
        c = ann2cl[a]
        out[a] = round(float(ct.loc[a, c] / ct.loc[a].sum()), 3)
    return out, ct, ann2cl

ov30 = {res: overlap(corr30[res])[0] for res in RES}
ov50 = {res: overlap(corr50[res])[0] for res in RES}
ovold = {}
for res in RES:
    m = pd.read_csv(os.path.join(OUT, f"matching_baseline_leiden_r{res}.csv"))
    ovold[res] = {r.annotation: r.overlap_fraction for _, r in m.iterrows()}

# ---- comparison table for 4 target tissues ----
print("\n=== 4 problem tissues: old(randomized-50) vs corrected(arpack-30/50) ===\n", flush=True)
rows = []
for res in RES:
    for a in TARGET:
        rows.append({
            "res": res, "tissue": a,
            "old_rnd50": ovold[res][a],
            "corr_arp30": ov30[res][a],
            "corr_arp50": ov50[res][a],
            "diff(arp30-old)": round(ov30[res][a] - ovold[res][a], 3),
            "diff(arp50-old)": round(ov50[res][a] - ovold[res][a], 3),
        })
cmp = pd.DataFrame(rows)
print(cmp.to_string(index=False), flush=True)

# also overall ARI/NMI for corrected30 vs annotation (context)
print("\n=== corrected baseline (arpack-30) overall ARI/NMI vs annotation ===", flush=True)
for res in RES:
    ari = adjusted_rand_score(ann, corr30[res].astype(str))
    nmi = normalized_mutual_info_score(ann, corr30[res].astype(str))
    print(f"    res={res}: n_clusters={len(np.unique(corr30[res]))}  ARI={ari:.4f}  NMI={nmi:.4f}", flush=True)

# ---- save corrected baseline (arpack-30) as the new sole reference ----
slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for res in RES:
    slim.obs[f"baseline_corrected_r{res}"] = corr30[res].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["X_pca"] = X_pca30.copy()
slim.write_h5ad(os.path.join(OUT, "baseline_corrected_slim.h5ad"))
# save full matching tables for corrected30
for res in RES:
    _, ct, ann2cl = overlap(corr30[res])
    mt = pd.DataFrame([{"annotation": a, "baseline_cluster": ann2cl[a],
                        "overlap_fraction": ov30[res][a]} for a in ann_cats])
    mt.to_csv(os.path.join(OUT, f"matching_corrected_r{res}.csv"), index=False)
    ct.to_csv(os.path.join(OUT, f"confusion_corrected_r{res}.csv"))

print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
