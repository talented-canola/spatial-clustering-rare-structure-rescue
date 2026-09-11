"""Bandwidth sensitivity: vary sigma (0.75/0.5/0.25 x median-neighbor-dist) in incl_self
smoothing, res=1.0 only. Report self-weight fraction + metrics + 7-tissue overlap."""
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
RES = r"F:/BGI/task3/spateo-release-main/results"
TISSUES = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord", "Dorsal root ganglion", "GI tract"]

t0 = time.time()
print("load + QC ...", flush=True)
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
ann = adata.obs["annotation"].astype(str).values
ann_cats = list(adata.obs["annotation"].cat.categories)
xy = adata.obsm["spatial"]
n = xy.shape[0]

s = 8
nn = NearestNeighbors(n_neighbors=s + 1).fit(xy)
dist, idx = nn.kneighbors(xy)
median_dist = float(np.median(dist[:, 1:]))
print(f"median neighbor distance = {median_dist:.3f}", flush=True)

def spatial_coherence(labels):
    li = labels.astype(int)
    nn2 = NearestNeighbors(n_neighbors=9).fit(xy)
    _, nidx = nn2.kneighbors(xy); nidx = nidx[:, 1:]
    return float((li[nidx] == li[:, None]).mean())

def overlap7(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    N = max(n_ann, n_cl)
    Cpad = np.zeros((N, N)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    out = {}
    for i in range(n_ann):
        a = ann_cats[i]; j = ci[i]
        out[a] = round(float(ct.loc[a, clusters[j]] / ct.loc[a].sum()), 3) if j < n_cl else 0.0
    return out

rows = []
for factor in [0.75, 0.5, 0.25]:
    sigma = factor * median_dist
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    w = w / w.sum(axis=1, keepdims=True)
    self_frac = float(w[:, 0].mean())
    W = csr_matrix((w.ravel(), (np.repeat(np.arange(n), s + 1), idx.ravel())), shape=(n, n))
    X_smooth = W @ adata.X

    a = sc.AnnData(X=X_smooth, obs=adata.obs, var=adata.var)
    a.uns["__type"] = "UMI"
    a.obsm["spatial"] = xy.copy()
    sc.pp.scale(a, max_value=10)
    _, a = sl.find_neighbors.neighbors(a, n_neighbors=30, basis="pca", n_pca_components=30)
    conn = a.obsp["expression_connectivities"].copy()
    conn.data[conn.data > 0] = 1
    lab = fl.fast_leiden(conn, 1.0)

    ari = adjusted_rand_score(ann, lab.astype(str))
    nmi = normalized_mutual_info_score(ann, lab.astype(str))
    coh = spatial_coherence(lab)
    ov = overlap7(lab)
    row = {"sigma": round(sigma, 3), "factor": factor, "self_weight_frac": round(self_frac, 3),
           "n_clusters": len(np.unique(lab)), "ARI": round(ari, 4), "NMI": round(nmi, 4),
           "coherence": round(coh, 4)}
    for t in TISSUES:
        row[t] = ov[t]
    rows.append(row)
    print(f"sigma={sigma:.3f} (x{factor}) self_frac={self_frac:.3f} clusters={len(np.unique(lab))} ARI={ari:.4f} NMI={nmi:.4f}", flush=True)

# reference rows: smooth_s8_incl_self (sigma=1.414) and baseline_v2
incl = sc.read_h5ad(f"{RES}/smooth_s8_incl_self/data/smooth_s8_incl_self_slim.h5ad")
ov_incl = overlap7(incl.obs["smooth_s8_incl_self_r1.0"].astype(str).values)
row_incl = {"sigma": round(median_dist, 3), "factor": 1.0, "self_weight_frac": round(float(np.exp(0)/ (np.exp(0)+np.sum(np.exp(-(dist[:,1:]**2)/(2*median_dist**2)),axis=1).mean())), 3),
            "n_clusters": 32, "ARI": 0.2772, "NMI": 0.5797, "coherence": 0.9156}
for t in TISSUES:
    row_incl[t] = ov_incl[t]

base = sc.read_h5ad(f"{RES}/baseline_v2_arpack/data/baseline_corrected_slim.h5ad")
ov_base = overlap7(base.obs["baseline_corrected_r1.0"].astype(str).values)
row_base = {"sigma": np.nan, "factor": np.nan, "self_weight_frac": 1.0,
            "n_clusters": 28, "ARI": 0.4234, "NMI": 0.6987, "coherence": 0.8405}
for t in TISSUES:
    row_base[t] = ov_base[t]

df = pd.DataFrame(rows + [row_incl, row_base])
order = ["sigma", "factor", "self_weight_frac", "n_clusters", "ARI", "NMI", "coherence"] + TISSUES
df = df[order]
print("\n=== summary ===")
print(df.to_string(index=False))
df.to_csv(f"{RES}/analysis/smooth_sigma_sweep.csv", index=False)
print(f"\nDONE {time.time()-t0:.0f}s, saved analysis/smooth_sigma_sweep.csv")
