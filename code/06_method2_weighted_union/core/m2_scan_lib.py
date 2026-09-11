"""Shared helpers for method2 weighted-union scan.

Graph construction MIRRORS the verified SCC `spatial_adj` pipeline exactly:
  - expr KNN graph: k=30 on arpack-30 X_pca, binary (rows -> k nearest, self dropped)
  - spatial KNN graph: k=8 on xy, binary
  - union nnz must equal the VERIFIED scc_s8 union nnz = 915412  (sanity anchor)
Weighted union: adj = We*expr_binary + Ws*spatial_binary  (W_e=1 throughout the scan).
Clustering: fast_leiden.fast_leiden_weighted (leidenalg RBConfigurationVertexPartition,
resolution_parameter, seed=888, n_iterations=-1, edge weights).
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

EXPR_K, SPAT_K = 30, 8   # spateo n_neighbors params for expression / spatial KNN
UNION_NNZ_REF = 915422   # gold-verified scc_s8 union nnz (from ACTUAL spateo spatial_adj)


def knn_binary(X, k):
    """FAITHFUL replication of spateo find_neighbors.neighbors():
    NearestNeighbors(algorithm='ball_tree', n_neighbors=k).kneighbors(X) returns k
    entries INCLUDING self; distance-0 entries (self) keep data 0 then are eliminated
    -> k-1 non-self neighbors per node. Gold-checked: expr(30)=793962, spatial(8)=191646,
    union=915422 — exactly matches spateo's obsp. Do NOT switch to k+1-and-drop-self."""
    n = X.shape[0]
    nn = NearestNeighbors(algorithm="ball_tree", n_neighbors=k, metric="euclidean").fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    data = (dist.ravel() > 0).astype(float)
    M = csr_matrix((data, (rows, cols)), shape=(n, n))
    M.eliminate_zeros()
    return M


def build_graphs(slim_path):
    import scanpy as sc
    a = sc.read_h5ad(slim_path)
    expr = knn_binary(a.obsm["X_pca"], EXPR_K)
    spat = knn_binary(a.obsm["spatial"], SPAT_K)
    return expr, spat, a


def weighted_union(expr, spat, Ws, We=1.0):
    """adj(i,j) = We*expr_binary + Ws*spatial_binary. scipy coalesces dual edges by summing."""
    adj = expr * float(We) + spat * float(Ws)
    return adj.tocsr()


def hungarian_overlaps(ann_series, labels, cats):
    """Full 19-class Hungarian-aligned overlap. ann_series keeps .cat.categories order."""
    ann = ann_series.astype(str).values
    lab = np.asarray(labels).astype(str)
    ct = pd.crosstab(pd.Series(ann), pd.Series(lab))
    ct = ct.reindex(index=[str(c) for c in cats], fill_value=0)
    na, nc = ct.shape
    N = max(na, nc)
    C = np.zeros((N, N)); C[:na, :nc] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-C)
    cl = list(ct.columns)
    out = {}
    for i in range(na):
        a = cats[i]; j = ci[i]
        out[a] = round(float(ct.loc[a, cl[j]] / ct.loc[a].sum()), 4) if j < nc else 0.0
    return out


def spatial_coherence(labels, xy):
    n = xy.shape[0]
    nn = NearestNeighbors(n_neighbors=9).fit(xy)
    _, idx = nn.kneighbors(xy); idx = idx[:, 1:]
    li = np.asarray(labels, dtype=int)
    return float((li[idx] == li[:, None]).mean())


def subsample_idx(n, k=4000, seed=0):
    rng = np.random.RandomState(seed)
    return rng.choice(n, size=k, replace=False)
