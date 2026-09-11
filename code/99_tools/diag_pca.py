"""Diagnose why ablation != baseline: confirm it's randomized (pca_spateo) vs
arpack (neighbors) PCA difference."""
import time
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
BASELINE = r"F:/BGI/task3/spateo-release-main/results/baseline/baseline_slim.h5ad"

t0 = time.time()
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
sc.pp.scale(adata, max_value=10)
X = adata.X  # scaled, dense

base = sc.read_h5ad(BASELINE, backed="r")
base_lab = base.obs["baseline_leiden_r1.0"].astype(str).values

# (A) randomized PCA (pca_spateo's exact path)
pca_rand = PCA(n_components=50, random_state=1).fit(X)   # svd_solver='auto' -> randomized
Xr = pca_rand.transform(X)

# (B) arpack PCA (neighbors' exact path)
pca_arp = PCA(n_components=50, svd_solver="arpack", random_state=0).fit(X)
Xa = pca_arp.transform(X)

# subspace agreement: max |corr| between each arpack PC and all randomized PCs
C = np.abs(np.corrcoef(Xa.T, Xr.T)[:50, 50:])   # 50 arpack x 50 randomized
mean_best = np.mean(C.max(axis=1))
# build KNN graph directly on embedding to mimic baseline's neighbors step
from sklearn.neighbors import NearestNeighbors
def knn_graph(Xemb, k=30):
    nn = NearestNeighbors(n_neighbors=k+1).fit(Xemb)
    d, idx = nn.kneighbors(Xemb)
    idx = idx[:, 1:]; d = d[:, 1:]
    import scipy.sparse as sp
    n = Xemb.shape[0]
    r = np.repeat(np.arange(n), k); c = idx.flatten()
    conn = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    return conn

for name, Xemb in [("randomized-50", Xr), ("arpack-50", Xa)]:
    conn = knn_graph(Xemb)
    lab = fl.fast_leiden(conn, 1.0)
    ari = adjusted_rand_score(lab.astype(str), base_lab)
    print(f"{name} ARI vs baseline r1.0 = {ari:.4f}", flush=True)

print(f"mean best |corr| arpack-vs-randomized PC = {mean_best:.6f}")
print(f"min best |corr| = {C.max(axis=1).min():.6f}")
print(f"DONE {time.time()-t0:.0f}s")
