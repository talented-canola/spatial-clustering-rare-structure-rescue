"""smooth_s8_incl_self: Gaussian-kernel spatial smoothing (s=8) WITH self included.
Self at distance 0 -> weight exp(0)=1 (max). Same pipeline otherwise."""
import os, time
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/smooth_s8_incl_self"
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
RES = [0.5, 1.0, 1.5, 2.0]

t0 = time.time()
print("load + QC ...", flush=True)
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
print(f"QC'd: {adata.shape}", flush=True)

ann = adata.obs["annotation"].astype(str).values
xy = adata.obsm["spatial"]
n = xy.shape[0]

# ---- Gaussian smoothing WITH self (self + s neighbors = s+1 points) ----
s = 8
print(f"Gaussian smoothing WITH self (s={s}) ...", flush=True)
nn = NearestNeighbors(n_neighbors=s + 1).fit(xy)
dist, idx = nn.kneighbors(xy)      # (n, s+1), col 0 = self (dist 0)
sigma = float(np.median(dist[:, 1:]))   # neighbors only, same sigma as excl-self version
w = np.exp(-(dist ** 2) / (2 * sigma ** 2))   # self (d=0) -> w=1 (max)
w = w / w.sum(axis=1, keepdims=True)
W = csr_matrix((w.ravel(), (np.repeat(np.arange(n), s + 1), idx.ravel())), shape=(n, n))
X_smooth = W @ adata.X
print(f"sigma={sigma:.3f}, X_smooth nnz={X_smooth.nnz} (orig {adata.X.nnz})", flush=True)

adata.X = X_smooth
sc.pp.scale(adata, max_value=10)

print("PCA + neighbors + Leiden ...", flush=True)
_, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=30)
conn = adata.obsp["expression_connectivities"].copy()
conn.data[conn.data > 0] = 1

labels = {}
for res in RES:
    labels[res] = fl.fast_leiden(conn, res)
    print(f"  res={res}: {len(np.unique(labels[res]))} clusters", flush=True)

slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for res in RES:
    slim.obs[f"smooth_s8_incl_self_r{res}"] = labels[res].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["X_pca"] = adata.obsm["X_pca"].copy()
slim.obsp["expression_connectivities"] = conn.copy()
slim.write_h5ad(os.path.join(OUT, "data", "smooth_s8_incl_self_slim.h5ad"))
print(f"DONE {time.time()-t0:.0f}s")
