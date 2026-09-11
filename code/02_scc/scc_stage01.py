"""Stage 0 (ablation / machinery check) + Stage 1 (SCC + intrinsic quality)."""
import time, os
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
BASELINE = r"F:/BGI/task3/spateo-release-main/results/baseline/baseline_slim.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/scc"
os.makedirs(OUT, exist_ok=True)

RES = [0.5, 1.0, 1.5, 2.0]
S_NEIGH = [3, 6, 10]

t0 = time.time()
print("[Stage 0] load + preprocess (identical to baseline) ...", flush=True)
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
sc.pp.scale(adata, max_value=10)
print("    preprocessed:", adata.shape, flush=True)

ann = adata.obs["annotation"].astype(str).values
ann_cats = list(adata.obs["annotation"].cat.categories)
xy = adata.obsm["spatial"]
n = adata.n_obs

base = sc.read_h5ad(BASELINE, backed="r")
assert np.array_equal(base.obs.index, adata.obs.index), "cell order mismatch!"
base_labels = {res: base.obs[f"baseline_leiden_r{res}"].astype(str).values for res in RES}

# ---- Stage 0: ablation ----
print("[Stage 0] ablation: expression-only via spatial_adj's neighbors path ...", flush=True)
ablation = {}
for n_pca in [50, 30]:
    if "X_pca" in adata.obsm:
        del adata.obsm["X_pca"]
    _, adata = sl.find_neighbors.neighbors(adata, n_neighbors=30, basis="pca", n_pca_components=n_pca)
    conn = adata.obsp["expression_connectivities"].copy()
    conn.data[conn.data > 0] = 1
    for res in RES:
        ablation[(n_pca, res)] = fl.fast_leiden(conn, res)

print("    ARI(ablation vs baseline) per resolution [0.5,1.0,1.5,2.0]:")
for n_pca in [50, 30]:
    ari = [round(adjusted_rand_score(ablation[(n_pca, res)].astype(str), base_labels[res]), 4) for res in RES]
    print(f"      n_pca={n_pca}: {ari}", flush=True)

# ---- Stage 1: SCC + intrinsic ----
print(f"[Stage 1] SCC via spatial_adj, s_neigh in {S_NEIGH} ...", flush=True)
results = {}
if "X_pca" in adata.obsm:
    del adata.obsm["X_pca"]
for s in S_NEIGH:
    adj = sl.utils.spatial_adj(adata, spatial_key="spatial", pca_key="pca",
                               e_neigh=30, s_neigh=s, n_pca_components=30)
    for res in RES:
        results[(s, res)] = fl.fast_leiden(adj, res)
    print(f"    s_neigh={s}: union adj nnz={adj.nnz}", flush=True)

X_pca = adata.obsm["X_pca"].copy()

# spatial KNN graph (k=8) for intrinsic metrics
knn = NearestNeighbors(n_neighbors=9, metric="euclidean").fit(xy)
_, nidx = knn.kneighbors(xy)
nidx = nidx[:, 1:]
rows = np.repeat(np.arange(n), 8)
cols = nidx.flatten()
adj_sp = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
adj_sp = adj_sp.maximum(adj_sp.T)

def intrinsic_metrics(labels):
    lab = np.asarray(labels, dtype=int)
    uniq = np.unique(lab)
    same = lab[nidx] == lab[:, None]
    coherence = float(same.mean())
    n_comp_list, tiny = [], 0
    for c in uniq:
        m = (lab == c)
        if int(m.sum()) < 20:
            tiny += 1
        sub = adj_sp[m][:, m]
        nc, _ = connected_components(sub, directed=False)
        n_comp_list.append(nc)
    return {
        "n_clusters": len(uniq),
        "spatial_coherence": round(coherence, 4),
        "mean_components_per_cluster": round(float(np.mean(n_comp_list)), 3),
        "max_components": int(np.max(n_comp_list)),
        "n_fragmented_clusters(comp>1)": int(np.sum(np.array(n_comp_list) > 1)),
        "n_tiny_clusters(<20)": tiny,
    }

iq_rows = []
for s in S_NEIGH:
    counts = [len(np.unique(results[(s, res)])) for res in RES]
    monotonic = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))
    for res in RES:
        m = intrinsic_metrics(results[(s, res)])
        m.update({"s_neigh": s, "res": res, "resolution_monotonic": monotonic})
        iq_rows.append(m)

iq = pd.DataFrame(iq_rows)
iq.to_csv(os.path.join(OUT, "stage1_intrinsic_quality.csv"), index=False)
print("\n" + iq.to_string(index=False), flush=True)

# raw spatial figures
for s in S_NEIGH:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for k, res in enumerate(RES):
        lab = results[(s, res)]
        axes[k].scatter(xy[:, 0], xy[:, 1], s=1, c=lab, cmap=plt.cm.turbo, rasterized=True)
        axes[k].set_title(f"s_neigh={s}, res={res}, {len(np.unique(lab))} clusters")
        axes[k].set_aspect("equal"); axes[k].invert_yaxis(); axes[k].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"stage1_raw_spatial_s{s}.png"), dpi=100)
    plt.close()

# save labels + X_pca
slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for s in S_NEIGH:
    for res in RES:
        slim.obs[f"scc_s{s}_r{res}"] = results[(s, res)].astype(str)
slim.obsm["spatial"] = xy.copy()
slim.obsm["X_pca"] = X_pca.copy()
slim.write_h5ad(os.path.join(OUT, "scc_slim.h5ad"))

print(f"\nStage 0+1 DONE in {time.time()-t0:.1f}s", flush=True)
