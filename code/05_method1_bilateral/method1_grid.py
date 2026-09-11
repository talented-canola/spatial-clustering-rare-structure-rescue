"""method1: bilateral-kernel weighted SCC grid search + 3-way ablation.
Loads X_pca (arpack-30) from baseline_corrected_slim (no re-PCA, memory-light)."""
import os, time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import fast_leiden as fl

BASE = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/analysis"
RES = [0.5, 1.0, 1.5, 2.0]

t0 = time.time()
base = sc.read_h5ad(BASE)
ann = base.obs["annotation"].astype(str).values
xy = base.obsm["spatial"]
X_pca = base.obsm["X_pca"]
n = xy.shape[0]
print(f"loaded: n={n}, X_pca={X_pca.shape}", flush=True)

nn_c = NearestNeighbors(n_neighbors=9).fit(xy)
_, nidx = nn_c.kneighbors(xy); nidx = nidx[:, 1:]

def spatial_coherence(labels):
    li = labels.astype(int)
    return float((li[nidx] == li[:, None]).mean())

def eval_labels(lab):
    return (adjusted_rand_score(ann, lab.astype(str)),
            normalized_mutual_info_score(ann, lab.astype(str)),
            spatial_coherence(lab), len(np.unique(lab)))

rows = []
for s in [4, 8, 12]:
    nn = NearestNeighbors(n_neighbors=s + 1).fit(xy)
    dist, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]; spat_dist = dist[:, 1:]           # (n, s)
    sigma_s = float(np.median(spat_dist))
    expr_dist = np.linalg.norm(X_pca[idx] - X_pca[:, None, :], axis=2)  # (n, s)
    sigma_e_modes = {
        "q25": np.percentile(expr_dist, 25),
        "q50": np.percentile(expr_dist, 50),
        "q75": np.percentile(expr_dist, 75),
        "adaptive": np.median(expr_dist, axis=1),
    }
    rows_n = np.repeat(np.arange(n), s)
    cols_n = idx.ravel()
    for mode, se in sigma_e_modes.items():
        se_arr = se[:, None] if mode == "adaptive" else se
        w = np.exp(-(spat_dist ** 2) / (2 * sigma_s ** 2)) * np.exp(-(expr_dist ** 2) / (2 * se_arr ** 2))
        adj = csr_matrix((w.ravel(), (rows_n, cols_n)), shape=(n, n))
        for res in RES:
            lab = fl.fast_leiden_weighted(adj, res)
            ari, nmi, coh, nc = eval_labels(lab)
            rows.append({"scheme": "method1", "s_neigh": s, "sigma_s": round(sigma_s, 3),
                         "sigma_e": mode, "res": res, "n_clusters": nc,
                         "ARI": round(ari, 4), "NMI": round(nmi, 4), "coherence": round(coh, 4)})
    print(f"s={s} done ({time.time()-t0:.0f}s)", flush=True)

# 3-way ablation at s=8, sigma_e=q50
s = 8
nn = NearestNeighbors(n_neighbors=s + 1).fit(xy)
dist, idx = nn.kneighbors(xy)
idx = idx[:, 1:]; spat_dist = dist[:, 1:]
sigma_s = float(np.median(spat_dist))
expr_dist = np.linalg.norm(X_pca[idx] - X_pca[:, None, :], axis=2)
sigma_e = float(np.percentile(expr_dist, 50))
rows_n = np.repeat(np.arange(n), s); cols_n = idx.ravel()
w_spatial = np.exp(-(spat_dist ** 2) / (2 * sigma_s ** 2))
w_expr = np.exp(-(expr_dist ** 2) / (2 * sigma_e ** 2))
for name, w in [("ablate_spatial_only", w_spatial), ("ablate_expr_only", w_expr), ("ablate_bilateral", w_spatial * w_expr)]:
    adj = csr_matrix((w.ravel(), (rows_n, cols_n)), shape=(n, n))
    for res in RES:
        lab = fl.fast_leiden_weighted(adj, res)
        ari, nmi, coh, nc = eval_labels(lab)
        rows.append({"scheme": name, "s_neigh": s, "sigma_s": round(sigma_s, 3),
                     "sigma_e": "q50", "res": res, "n_clusters": nc,
                     "ARI": round(ari, 4), "NMI": round(nmi, 4), "coherence": round(coh, 4)})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "method1_grid_search.csv"), index=False)
print("\n=== grid summary (res=1.0) ===")
print(df[df.res == 1.0].sort_values("ARI", ascending=False).to_string(index=False))
print(f"\nDONE {time.time()-t0:.0f}s")
