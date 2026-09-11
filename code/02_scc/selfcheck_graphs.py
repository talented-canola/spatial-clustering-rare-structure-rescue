"""Self-check: verify scc_default(s=6) and scc_s8(s=8) used DIFFERENT union adjacency
matrices (i.e., the confusion/spatial outputs were NOT fed a wrongly-shared graph)."""
import scanpy as sc
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

B2 = sc.read_h5ad(r"F:/BGI/task3/spateo-release-main/results/baseline/baseline_corrected_slim.h5ad")
X_pca = B2.obsm["X_pca"]   # arpack-30 (shared, deterministic)
xy = B2.obsm["spatial"]
n = xy.shape[0]

def knn_conn(data, n_neighbors):
    """Replicates find_neighbors.neighbors(): NearestNeighbors(n_neighbors) incl. self,
    then drop self (distance 0) -> (n_neighbors-1) real neighbors, binary."""
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    _, idx = nn.kneighbors(data)
    idx = idx[:, 1:]                      # drop self (col 0)
    k = n_neighbors - 1
    rows = np.repeat(np.arange(n), k)
    cols = idx.flatten()
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

expr = knn_conn(X_pca, 30)   # e_neigh=30 -> 29 real neighbors
spat6 = knn_conn(xy, 6)      # s_neigh=6 -> 5 real neighbors
spat8 = knn_conn(xy, 8)      # s_neigh=8 -> 7 real neighbors

union6 = expr + spat6
union6.data[union6.data > 0] = 1
union8 = expr + spat8
union8.data[union8.data > 0] = 1

print("expression_conn nnz:", expr.nnz)
print("spatial_conn(s=6) nnz:", spat6.nnz)
print("spatial_conn(s=8) nnz:", spat8.nnz)
print("---")
print("union_6 (scc_default) nnz:", union6.nnz)
print("union_8 (scc_s8)       nnz:", union8.nnz)
print("same object (is)?", union6 is union8)
print("id(union6) == id(union8)?", id(union6) == id(union8))
print("matrices exactly equal?", (union6 != union8).nnz == 0)
print("nnz difference:", abs(union6.nnz - union8.nnz))
