"""Manual GraphST pipeline: replicate preprocess + sparse spatial graph + train loop,
avoiding GraphST's dense n x n adjacency memory blowup. Monkeypatch AvgReadout for sparse."""
import os, time
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import fast_leiden as fl

from GraphST.model import Encoder_sparse, AvgReadout
from GraphST.preprocess import preprocess_adj_sparse, fix_seed, permutation

# --- monkeypatch AvgReadout to support sparse mask ---
def sparse_readout(self, emb, mask=None):
    if mask.is_sparse:
        vsum = torch.spmm(mask, emb)
        row_sum = torch.spmm(mask, torch.ones(emb.shape[0], 1, device=emb.device)).squeeze(1)
        global_emb = vsum / row_sum.unsqueeze(1)
    else:
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum
    return F.normalize(global_emb, p=2, dim=1)
AvgReadout.forward = sparse_readout

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/graphst_reference"
os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
RES = [0.5, 1.0, 1.5, 2.0]
device = torch.device("cpu")
fix_seed(41)

t0 = time.time()
print("load + QC ...", flush=True)
adata = sc.read_h5ad(DATA)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
ann = adata.obs["annotation"].astype(str).values
xy = adata.obsm["spatial"]
n = adata.n_obs

# --- replicate GraphST.preprocess (needs raw counts) ---
adata.X = adata.layers["count"].copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, zero_center=False, max_value=10)
print(f"preprocess done, HVG={int(adata.var.highly_variable.sum())}", flush=True)

# --- sparse spatial graph (replicate construct_interaction_KNN, n_neighbors=3) ---
k = 3
nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy)
_, indices = nbrs.kneighbors(xy)
x = indices[:, 0].repeat(k); y = indices[:, 1:].flatten()
interaction = csr_matrix((np.ones(len(x)), (x, y)), shape=(n, n))
graph_neigh = interaction
adj = interaction + interaction.T
adj.data[adj.data > 1] = 1
adj_t = preprocess_adj_sparse(adj).to(device)   # torch sparse
gn = (graph_neigh + sp.identity(n, format='csr')).tocoo()
graph_neigh_t = torch.sparse.FloatTensor(
    torch.LongTensor(np.vstack((gn.row, gn.col))), torch.FloatTensor(gn.data), torch.Size(gn.shape)
).to(device)
print("sparse graph built", flush=True)

# --- features + contrastive label ---
feat = adata[:, adata.var['highly_variable']].X.toarray().astype(np.float32)
features = torch.FloatTensor(feat).to(device)
label_CSL = torch.FloatTensor(np.concatenate([np.ones((n, 1)), np.zeros((n, 1))], axis=1)).to(device)
dim_input = feat.shape[1]

model = Encoder_sparse(dim_input, 64, graph_neigh_t).to(device)
loss_CSL = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0)

print(f"train GraphST (600 epochs, CPU, dim_input={dim_input}) ...", flush=True)
for epoch in range(600):
    model.train()
    features_a = torch.FloatTensor(permutation(feat)).to(device)
    hiden_feat, emb, ret, ret_a = model(features, features_a, adj_t)
    loss_sl_1 = loss_CSL(ret, label_CSL)
    loss_sl_2 = loss_CSL(ret_a, label_CSL)
    loss_feat = F.mse_loss(features, emb)
    loss = 10 * loss_feat + 1 * (loss_sl_1 + loss_sl_2)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

with torch.no_grad():
    model.eval()
    emb_rec = model(features, features, adj_t)[1]
    emb_rec = F.normalize(emb_rec, p=2, dim=1).detach().cpu().numpy()
print(f"emb_rec shape {emb_rec.shape}, train time {time.time()-t0:.0f}s", flush=True)

# cluster emb_rec via KNN(30) + Leiden
nn = NearestNeighbors(n_neighbors=31).fit(emb_rec)
_, idx = nn.kneighbors(emb_rec); idx = idx[:, 1:]
rows = np.repeat(np.arange(n), 30)
conn = csr_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(n, n))
labels = {}
for res in RES:
    labels[res] = fl.fast_leiden(conn, res)
    print(f"  res={res}: {len(np.unique(labels[res]))} clusters", flush=True)

slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for res in RES:
    slim.obs[f"graphst_reference_r{res}"] = labels[res].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["emb"] = emb_rec.copy()
slim.obsp["emb_connectivities"] = conn.copy()
slim.write_h5ad(os.path.join(OUT, "data", "graphst_reference_slim.h5ad"))
print(f"DONE {time.time()-t0:.0f}s")
