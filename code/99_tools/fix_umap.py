"""Fix UMAP: compute per-scheme UMAP from each scheme's OWN connectivities (not a shared
X_pca embedding), and fix the 'by annotation' left panel to actually color by annotation."""
import os
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

RES = r"F:/BGI/task3/spateo-release-main/results"
B2 = sc.read_h5ad(os.path.join(RES, "baseline", "baseline_corrected_slim.h5ad"))
SCC = sc.read_h5ad(os.path.join(RES, "scc", "scc_slim.h5ad"))
SCC4812 = sc.read_h5ad(os.path.join(RES, "scc", "scc_s4812_slim.h5ad"))

X_pca = B2.obsm["X_pca"]
xy = B2.obsm["spatial"]
ann = B2.obs["annotation"].astype(str).values
ann_cats = list(B2.obs["annotation"].cat.categories)
n = xy.shape[0]
pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}

def knn_conn(data, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    _, idx = nn.kneighbors(data)
    idx = idx[:, 1:]                       # drop self
    k = n_neighbors - 1
    rows = np.repeat(np.arange(n), k)
    cols = idx.flatten()
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

expr = knn_conn(X_pca, 30)
spat6 = knn_conn(xy, 6)
spat8 = knn_conn(xy, 8)
union6 = expr + spat6; union6.data[union6.data > 0] = 1
union8 = expr + spat8; union8.data[union8.data > 0] = 1

def umap_on_conn(conn):
    tmp = sc.AnnData(X=X_pca.copy())
    tmp.obsp["connectivities"] = conn
    tmp.obsp["distances"] = conn.copy()
    tmp.uns["neighbors"] = {"connectivities_key": "connectivities",
                            "distances_key": "distances",
                            "params": {"n_neighbors": 30, "method": "umap"}}
    sc.tl.umap(tmp, random_state=42)
    return tmp.obsm["X_umap"]

schemes = [
    ("baseline_v2", "baseline", B2.obs["baseline_corrected_r1.0"].astype(str).values, expr),
    ("scc_default", "scc", SCC.obs["scc_s6_r1.0"].astype(str).values, union6),
    ("scc_s8", "scc", SCC4812.obs["scc_s8_r1.0"].astype(str).values, union8),
    ("scc_official", "scc", SCC4812.obs["scc_louvain_res0.4_s8"].astype(str).values, union8),
]

umaps = {}
for scheme_id, outdir, lab, conn in schemes:
    um = umap_on_conn(conn)
    umaps[scheme_id] = um
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for a in ann_cats:
        m = (ann == a)
        axes[0].scatter(um[m, 0], um[m, 1], s=2, color=ann_colors[a], rasterized=True)
    axes[0].set_title("UMAP by annotation")
    axes[1].scatter(um[:, 0], um[:, 1], s=2, c=lab.astype(int), cmap=plt.cm.turbo, rasterized=True)
    axes[1].set_title(f"UMAP by {scheme_id}")
    plt.tight_layout()
    p = os.path.join(RES, outdir, f"umap_{scheme_id}.png")
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"saved {p}")

# verification: pairwise mean-abs-diff of embeddings
print("\n=== UMAP embedding pairwise mean |diff| ===")
for a in umaps:
    for b in umaps:
        if a < b:
            d = float(np.abs(umaps[a] - umaps[b]).mean())
            print(f"  {a:14s} vs {b:14s}: {d:.4f}")
