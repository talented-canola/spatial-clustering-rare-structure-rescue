"""Gold-standard check: run the ACTUAL spateo spatial_adj(e=30,s=8) and compare its
obsp graph nnz against a plain-sklearn replication. This settles what graph SCC really built."""
import sys
sys.path.insert(0, "F:/tmp")
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

import spateo_loader  # registers spateo leaf modules + matplotlib shim

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"

# --- actual spateo spatial_adj, run directly on the slim (27378-cell filtered adata,
#     same X_pca/spatial the SCC pipeline used) ---
slim = sc.read_h5ad(SLIM)
slim.uns["__type"] = "UMI"  # satisfy SKM.check_adata_is_type for the gold-standard nnz check
adj_real = spateo_loader.utils.spatial_adj(slim, e_neigh=30, s_neigh=8)
conn_e = slim.obsp["expression_connectivities"]
conn_s = slim.obsp["spatial_connectivities"]
print(f"SPATEO actual: expr_conn nnz={conn_e.nnz}  spatial_conn nnz={conn_s.nnz}  union adj nnz={adj_real.nnz}")
print(f"  expr per-cell={conn_e.nnz/slim.n_obs:.3f}  spatial per-cell={conn_s.nnz/slim.n_obs:.3f}")

# --- plain sklearn replication of spateo neighbors() ---
def spateo_connectivities(X, K, algo="ball_tree"):
    n = X.shape[0]
    nn = NearestNeighbors(algorithm=algo, n_neighbors=K, metric="euclidean").fit(X)
    dist, idx = nn.kneighbors(X)          # K incl self
    rows = np.repeat(np.arange(n), K)
    cols = idx.ravel()
    data = (dist.ravel() > 0).astype(float)   # spateo: data>0 -> 1, dist 0 (self) -> 0 then eliminated
    M = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    M.eliminate_zeros()
    return M

expr_r = spateo_connectivities(slim.obsm["X_pca"], 30)
spat_r = spateo_connectivities(slim.obsm["spatial"], 8)
adj_r = expr_r + spat_r
adj_r.data[adj_r.data > 0] = 1
print(f"SKLEARN repl.: expr_conn nnz={expr_r.nnz}  spatial_conn nnz={spat_r.nnz}  union adj nnz={adj_r.nnz}")
print(f"  expr matches spateo  = {expr_r.nnz == conn_e.nnz}")
print(f"  union matches spateo = {adj_r.nnz == adj_real.nnz}")

# also check MY previous 30/8 construction vs spateo
def knn30_8(X_e, X_s):
    n = X_e.shape[0]
    nne = NearestNeighbors(n_neighbors=31).fit(X_e); _, idxe = nne.kneighbors(X_e); idxe = idxe[:, 1:]
    nns = NearestNeighbors(n_neighbors=9).fit(X_s); _, idxs = nns.kneighbors(X_s); idxs = idxs[:, 1:]
    re_ = np.repeat(np.arange(n), 30); cs_ = idxe.ravel()
    rs_ = np.repeat(np.arange(n), 8); cs = idxs.ravel()
    E = sp.csr_matrix((np.ones(len(re_)), (re_, cs_)), shape=(n, n))
    S = sp.csr_matrix((np.ones(len(rs_)), (rs_, cs)), shape=(n, n))
    return E, S, (E + S)
E, S, U = knn30_8(slim.obsm["X_pca"], slim.obsm["spatial"])
U.data[U.data > 0] = 1
print(f"OLD 30/8 build: expr nnz={E.nnz} spatial nnz={S.nnz} union nnz={U.nnz}")
print(f"  old union vs spateo = {U.nnz == adj_real.nnz}")
