"""Check 1: GraphST — cluster on hidden z (64-dim) vs reconstruction h (3000-dim).
Re-train 600 epochs, extract both, compare ARI at res=1.0."""
import os, time
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import fast_leiden as fl
from GraphST.model import Encoder_sparse, AvgReadout
from GraphST.preprocess import preprocess_adj_sparse, fix_seed, permutation

def sparse_readout(self, emb, mask=None):
    if mask.is_sparse:
        vsum = torch.spmm(mask, emb)
        row_sum = torch.spmm(mask, torch.ones(emb.shape[0], 1, device=emb.device)).squeeze(1)
        g = vsum / row_sum.unsqueeze(1)
    else:
        vsum = torch.mm(mask, emb); row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        g = vsum / row_sum
    return F.normalize(g, p=2, dim=1)
AvgReadout.forward = sparse_readout

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
device = torch.device("cpu"); fix_seed(41)
t0 = time.time()
adata = sc.read_h5ad(DATA)
sc.pp.filter_genes(adata, min_cells=3); sc.pp.filter_cells(adata, min_genes=50)
ann = adata.obs["annotation"].astype(str).values
xy = adata.obsm["spatial"]; n = adata.n_obs
adata.X = adata.layers["count"].copy()
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata); sc.pp.scale(adata, zero_center=False, max_value=10)

k = 3
nbrs = NearestNeighbors(n_neighbors=k+1).fit(xy)
_, indices = nbrs.kneighbors(xy)
x = indices[:,0].repeat(k); y = indices[:,1:].flatten()
interaction = csr_matrix((np.ones(len(x)), (x, y)), shape=(n, n))
adj = interaction + interaction.T; adj.data[adj.data > 1] = 1
adj_t = preprocess_adj_sparse(adj).to(device)
gn = (interaction + sp.identity(n, format='csr')).tocoo()
graph_neigh_t = torch.sparse.FloatTensor(torch.LongTensor(np.vstack((gn.row, gn.col))), torch.FloatTensor(gn.data), torch.Size(gn.shape)).to(device)

feat = adata[:, adata.var['highly_variable']].X.toarray().astype(np.float32)
features = torch.FloatTensor(feat).to(device)
label_CSL = torch.FloatTensor(np.concatenate([np.ones((n,1)), np.zeros((n,1))], axis=1)).to(device)
model = Encoder_sparse(feat.shape[1], 64, graph_neigh_t).to(device)
loss_CSL = torch.nn.BCEWithLogitsLoss(); opt = torch.optim.Adam(model.parameters(), 0.001, weight_decay=0.0)
for epoch in range(600):
    model.train()
    fa = torch.FloatTensor(permutation(feat)).to(device)
    hf, h, ret, ret_a = model(features, fa, adj_t)
    loss = 10*F.mse_loss(features, h) + 1*(loss_CSL(ret, label_CSL) + loss_CSL(ret_a, label_CSL))
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    model.eval()
    z_hidden, h_recon, _, _ = model(features, features, adj_t)
    z_hidden = z_hidden.detach().cpu().numpy()           # (n, 64) hidden
    h_recon = F.normalize(h_recon, p=2, dim=1).detach().cpu().numpy()  # (n, 3000) recon
print(f"trained {time.time()-t0:.0f}s, z={z_hidden.shape}, h={h_recon.shape}", flush=True)

def cluster_and_score(emb, name):
    nn = NearestNeighbors(n_neighbors=31).fit(emb)
    _, idx = nn.kneighbors(emb); idx = idx[:, 1:]
    rows = np.repeat(np.arange(n), 30)
    conn = csr_matrix((np.ones(len(rows)), (rows, idx.ravel())), shape=(n, n))
    for res in [0.5, 1.0, 1.5, 2.0]:
        lab = fl.fast_leiden(conn, res)
        ari = adjusted_rand_score(ann, lab.astype(str))
        nmi = normalized_mutual_info_score(ann, lab.astype(str))
        print(f"  {name} res={res}: n={len(np.unique(lab)):3d} ARI={ari:.4f} NMI={nmi:.4f}", flush=True)

print("=== hidden z (64-dim) ==="); cluster_and_score(z_hidden, "z_hidden")
print("=== reconstruction h (3000-dim) ==="); cluster_and_score(h_recon, "h_recon")
print(f"DONE {time.time()-t0:.0f}s")
