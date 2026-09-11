"""INDEPENDENT recompute for W_s=0.5 only.

Cross-check of the 5-ratio scan agent. Graph construction is written FRESH here
(own KNN builder, does NOT import m2_scan_lib), so this path is independent of the
scan agent's graph-building code. Only fast_leiden is imported for clustering
(required by the task). Metrics are computed with this script's own code.
"""
import sys
import hashlib
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, "F:/tmp")
import fast_leiden  # only for the weighted Leiden step

SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"


def build(X, K):
    """Own spateo-faithful KNN builder (fresh, mirrors find_neighbors.neighbors):
    ball_tree, k neighbors incl. self; self-distance (0) dropped -> k-1 non-self
    neighbors per node; binary adjacency."""
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(algorithm='ball_tree', n_neighbors=K, metric='euclidean').fit(X)
    d, idx = nn.kneighbors(X)
    r = np.repeat(np.arange(n), K)
    c = idx.ravel()
    data = (d.ravel() > 0).astype(float)
    M = sp.csr_matrix((data, (r, c)), shape=(n, n))
    M.eliminate_zeros()
    return M


def own_spatial_coherence(labels, xy, k=8):
    """Own k=8 spatial-neighbor purity (fraction of a cell's 8 nearest xy neighbors
    sharing its cluster label; self excluded)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(xy)
    _, idx = nn.kneighbors(xy)
    idx = idx[:, 1:]  # drop self
    li = np.asarray(labels, dtype=int)
    return float((li[idx] == li[:, None]).mean())


def own_hungarian_overlaps(ann_series, labels, cats):
    """Own full 19x-ncrosstab Hungarian-aligned per-class overlap.
    ann_series keeps .cat.categories order."""
    ann = ann_series.astype(str).values
    lab = np.asarray(labels).astype(str)
    ct = pd.crosstab(pd.Series(ann), pd.Series(lab))
    ct = ct.reindex(index=[str(c) for c in cats], fill_value=0)
    na, nc = ct.shape
    N = max(na, nc)
    C = np.zeros((N, N))
    C[:na, :nc] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-C)
    cl = list(ct.columns)
    out = {}
    for i in range(na):
        a = cats[i]
        j = ci[i]
        out[a] = round(float(ct.loc[a, cl[j]] / ct.loc[a].sum()), 4) if j < nc else 0.0
    return out


# 1. load AnnData
import scanpy as sc
a = sc.read_h5ad(SLIM)
n = a.n_obs
print(f"n_obs={n}, X_pca shape={a.obsm['X_pca'].shape}, spatial shape={a.obsm['spatial'].shape}")

# 2. build BOTH binary graphs with OWN code
expr = build(a.obsm["X_pca"], 30)
spat = build(a.obsm["spatial"], 8)
union = (expr + spat).tocsr()

print(f"expr(30) nnz   = {expr.nnz}   (expect 793962)")
print(f"spat(8) nnz    = {spat.nnz}   (expect 191646)")
print(f"union nnz      = {union.nnz}  (expect 915422)")
ok = (expr.nnz == 793962) and (spat.nnz == 191646) and (union.nnz == 915422)
print(f"SANITY_OK = {ok}")
if not ok:
    # stop and report the discrepancy rather than proceeding
    print("STOP: graph nnz mismatch with verified SCC union. Not proceeding.")
    sys.exit(1)

# 3. weighted union W_s=0.5, W_e=1
adj = expr * 1.0 + spat * 0.5
adj = adj.tocsr()
nw = adj.nnz
w1 = int(np.sum(adj.data == 1.0))
w15 = int(np.sum(adj.data == 1.5))
print(f"adj nnz={nw}, weight=1.0 edges={w1}, weight=1.5 edges={w15}")

# 4. weighted Leiden
lab = fast_leiden.fast_leiden_weighted(adj, 1.0)

# 5. metrics
ann = a.obs["annotation"].astype(str)
n_clusters = len(np.unique(lab))
ARI = adjusted_rand_score(ann, lab.astype(str))
NMI = normalized_mutual_info_score(ann, lab.astype(str))
sub = np.load("F:/tmp/m2_subsample.npy")
Silhouette = silhouette_score(a.obsm["X_pca"][sub], lab.astype(int)[sub])
coherence = own_spatial_coherence(lab, a.obsm["spatial"])

# hungarian overlaps, keep the same 8 classes as the scan agent
ov_all = own_hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
keep = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
overlaps = {k: ov_all[k] for k in keep}

# label sha256
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

# 6. single JSON on the LAST line
out = {
    "Ws": 0.5,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Silhouette, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": overlaps,
}
print(json.dumps(out, ensure_ascii=False))
