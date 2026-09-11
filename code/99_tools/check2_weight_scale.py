"""Check 2: does the bilateral-kernel weight MAGNITUDE cause the 48-cluster explosion?
Test uniform rescaling invariance (RBConfiguration modularity should be scale-invariant)."""
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
import fast_leiden as fl

BASE = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
base = sc.read_h5ad(BASE)
ann = base.obs["annotation"].astype(str).values
xy = base.obsm["spatial"]
X_pca = base.obsm["X_pca"]
n = xy.shape[0]

# bilateral over expr-KNN (30), s=8 spatial weight, sigma_e=q50
nn = NearestNeighbors(n_neighbors=31).fit(X_pca)
edist, eidx = nn.kneighbors(X_pca)
eidx = eidx[:, 1:]; edist = edist[:, 1:]
sdist = np.linalg.norm(xy[eidx] - xy[:, None, :], axis=2)
sigma_e = float(np.median(edist)); sigma_s = float(np.median(sdist))
w = np.exp(-(edist**2)/(2*sigma_e**2)) * np.exp(-(sdist**2)/(2*sigma_s**2))
print(f"mean bilateral weight = {w.mean():.4f}  (min {w.min():.4f}, max {w.max():.4f})")
print(f"SCC union graph weight = 1.0 (binary)  -> ratio SCC/bilateral = {1.0/w.mean():.2f}x")

rows_n = np.repeat(np.arange(n), 30); cols_n = eidx.ravel()
print("\nrescaling invariance test (res=1.0):")
for scale in [1.0, 10.0, 100.0, 1000.0]:
    adj = csr_matrix(((w * scale).ravel(), (rows_n, cols_n)), shape=(n, n))
    lab = fl.fast_leiden_weighted(adj, 1.0)
    ari = adjusted_rand_score(ann, lab.astype(str))
    print(f"  scale={scale:>6.0f}: n_clusters={len(np.unique(lab)):3d}  ARI={ari:.4f}")
