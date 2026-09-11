"""GraphST reference: train on raw counts (Stereo), cluster emb via KNN+Leiden at 4 resolutions."""
import os, time
import numpy as np
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/graphst_reference"
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
RES = [0.5, 1.0, 1.5, 2.0]

t0 = time.time()
print("load + QC ...", flush=True)
adata = sc.read_h5ad(DATA)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
# GraphST does its own normalize+log1p; feed raw counts
adata.X = adata.layers["count"].copy()
ann = adata.obs["annotation"].astype(str).values
xy = adata.obsm["spatial"]

from GraphST.GraphST import GraphST
device = torch.device("cpu")
print("train GraphST (Stereo, CPU, 600 epochs) ...", flush=True)
model = GraphST(adata, device=device, datatype="Stereo", epochs=600)
adata_out = model.train()
emb = adata_out.obsm["emb"]  # (n, 64)
print(f"emb shape {emb.shape}, train time {time.time()-t0:.0f}s", flush=True)

# cluster emb via KNN(30) + Leiden, consistent with other methods
nn = NearestNeighbors(n_neighbors=31).fit(emb)
_, idx = nn.kneighbors(emb)
idx = idx[:, 1:]  # drop self -> 30 neighbors
rows = np.repeat(np.arange(emb.shape[0]), 30)
conn = csr_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(emb.shape[0], emb.shape[0]))

labels = {}
for res in RES:
    labels[res] = fl.fast_leiden(conn, res)
    print(f"  res={res}: {len(np.unique(labels[res]))} clusters", flush=True)

slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for res in RES:
    slim.obs[f"graphst_reference_r{res}"] = labels[res].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["emb"] = emb.copy()
slim.obsp["emb_connectivities"] = conn.copy()
slim.write_h5ad(os.path.join(OUT, "data", "graphst_reference_slim.h5ad"))
print(f"DONE {time.time()-t0:.0f}s")
