"""smooth_s8: Gaussian-kernel spatial smoothing (s=8) then standard clustering.
Pipeline identical to baseline_v2 (full-gene PCA arpack-30 -> neighbors e_neigh=30 -> Leiden),
except the log1p expression is first smoothed over 8 spatial neighbors."""
import os, time
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/smooth_s8"
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

# ---- Gaussian-kernel spatial smoothing over s=8 neighbors (exclude self) ----
s = 8
print(f"Gaussian smoothing (s={s}) ...", flush=True)
nn = NearestNeighbors(n_neighbors=s + 1).fit(xy)
dist, idx = nn.kneighbors(xy)
idx = idx[:, 1:]          # drop self
dist = dist[:, 1:]        # (n, s)
sigma = float(np.median(dist))
w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
w = w / w.sum(axis=1, keepdims=True)
W = csr_matrix((w.ravel(), (np.repeat(np.arange(n), s), idx.ravel())), shape=(n, n))
X_smooth = W @ adata.X   # sparse @ sparse -> sparse, log1p smoothed
print(f"sigma={sigma:.3f}, X_smooth nnz={X_smooth.nnz} (orig {adata.X.nnz})", flush=True)

# replace X with smoothed, then standard pipeline (identical to baseline_v2)
adata.X = X_smooth
sc.pp.scale(adata, max_value=10)

print("PCA (arpack-30, full gene) + neighbors (e_neigh=30) ...", flush=True)
_, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=30)
conn = adata.obsp["expression_connectivities"].copy()
conn.data[conn.data > 0] = 1

print("Leiden ...", flush=True)
labels = {}
for res in RES:
    labels[res] = fl.fast_leiden(conn, res)
    print(f"  res={res}: {len(np.unique(labels[res]))} clusters", flush=True)

# save slim
slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for res in RES:
    slim.obs[f"smooth_s8_r{res}"] = labels[res].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["X_pca"] = adata.obsm["X_pca"].copy()
slim.obsp["expression_connectivities"] = conn.copy()
slim.write_h5ad(os.path.join(OUT, "data", "smooth_s8_slim.h5ad"))

print(f"\nDONE {time.time()-t0:.0f}s, saved {OUT}/data/smooth_s8_slim.h5ad")
