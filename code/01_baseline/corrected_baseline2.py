"""Corrected baseline (arpack) + per-tissue overlap vs old (randomized). Fixed overlap fn."""
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

def overlap(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    n = max(n_ann, n_cl)
    Cpad = np.zeros((n, n)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    out, ann2cl = {}, {}
    for i in range(n_ann):
        a = ann_cats[i]
        j = ci[i]
        if j < n_cl:
            c = clusters[j]
            out[a] = round(float(ct.loc[a, c] / ct.loc[a].sum()), 3)
            ann2cl[a] = c
        else:
            out[a] = 0.0
            ann2cl[a] = None
    return out, ct, ann2cl

def run_baseline(adata, n_pca):
    if "X_pca" in adata.obsm:
        del adata.obsm["X_pca"]
    _, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=n_pca)
    conn = adata.obsp["expression_connectivities"].copy()
    conn.data[conn.data > 0] = 1
    labels = {res: fl.fast_leiden(conn, res) for res in RES}
    return labels, adata.obsm["X_pca"].copy(), adata

print("corrected baseline n_pca=30 ...", flush=True)
corr30, X_pca30, adata = run_baseline(adata, 30)
print("corrected baseline n_pca=50 ...", flush=True)
corr50, _, adata = run_baseline(adata, 50)

ov30 = {res: overlap(corr30[res])[0] for res in RES}
ov50 = {res: overlap(corr50[res])[0] for res in RES}

ovold = {}
for res in RES:
    m = pd.read_csv(os.path.join(OUT, f"matching_baseline_leiden_r{res}.csv"))
    ovold[res] = {r.annotation: float(r.overlap_fraction) for _, r in m.iterrows()}

print("\n=== 4 problem tissues: old(randomized-50) vs corrected(arpack-30/50) ===\n", flush=True)
rows = []
for res in RES:
    for a in TARGET:
        rows.append({"res": res, "tissue": a,
                     "old_rnd50": ovold[res][a],
                     "corr_arp30": ov30[res][a],
                     "corr_arp50": ov50[res][a],
                     "d30-old": round(ov30[res][a] - ovold[res][a], 3),
                     "d50-old": round(ov50[res][a] - ovold[res][a], 3)})
cmp = pd.DataFrame(rows)
print(cmp.to_string(index=False), flush=True)
cmp.to_csv(os.path.join(OUT, "corrected_vs_old_target_tissues.csv"), index=False)

print("\n=== corrected baseline (arpack-30) overall ARI/NMI + cluster counts ===", flush=True)
for res in RES:
    ari = adjusted_rand_score(ann, corr30[res].astype(str))
    nmi = normalized_mutual_info_score(ann, corr30[res].astype(str))
    print(f"    res={res}: n_clusters={len(np.unique(corr30[res]))}  ARI={ari:.4f}  NMI={nmi:.4f}", flush=True)

# save corrected baseline (arpack-30) as sole reference
slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for res in RES:
    slim.obs[f"baseline_corrected_r{res}"] = corr30[res].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["X_pca"] = X_pca30.copy()
slim.write_h5ad(os.path.join(OUT, "baseline_corrected_slim.h5ad"))
for res in RES:
    _, ct, ann2cl = overlap(corr30[res])
    mt = pd.DataFrame([{"annotation": a, "baseline_cluster": ann2cl[a],
                        "overlap_fraction": ov30[res][a]} for a in ann_cats])
    mt.to_csv(os.path.join(OUT, f"matching_corrected_r{res}.csv"), index=False)
    ct.to_csv(os.path.join(OUT, f"confusion_corrected_r{res}.csv"))

print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
