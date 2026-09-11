"""Re-train GraphST, SAVE z (hidden) and h (reconstruction), produce fig3 (UMAP z vs h)."""
import os, time
import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
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
OUT = r"F:/BGI/task3/spateo-release-main/results/analysis/figures"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cpu"); fix_seed(41)
t0 = time.time()
adata = sc.read_h5ad(DATA)
sc.pp.filter_genes(adata, min_cells=3); sc.pp.filter_cells(adata, min_genes=50)
ann = adata.obs["annotation"].astype(str).values
ann_cats = list(adata.obs["annotation"].cat.categories)
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
    z_hidden = z_hidden.detach().cpu().numpy()
    h_recon = F.normalize(h_recon, p=2, dim=1).detach().cpu().numpy()
print(f"trained {time.time()-t0:.0f}s", flush=True)

# save z and h
slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
slim.obsm["z_hidden"] = z_hidden; slim.obsm["h_recon"] = h_recon; slim.obsm["spatial"] = xy
slim.write_h5ad(r"F:/BGI/task3/spateo-release-main/results/graphst_reference/data/graphst_emb_z_h.h5ad")

# fig 3: UMAP z vs h, colored by annotation
pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, emb, title in [(axes[0], z_hidden, "UMAP of hidden z (64-dim)"), (axes[1], h_recon, "UMAP of reconstruction h (3000-dim)")]:
    um = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(emb)
    for a in ann_cats:
        m = (ann == a)
        ax.scatter(um[m, 0], um[m, 1], s=2, color=ann_colors[a], rasterized=True)
    ax.set_title(title)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_graphst_umap_z_vs_h.png"), dpi=130); plt.close()
print(f"fig3 saved, DONE {time.time()-t0:.0f}s")
