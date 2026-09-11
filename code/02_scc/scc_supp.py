"""Supplementary SCC runs:
  (1) s_neigh in {4,8,12} (Stereo-seq official grid recs) with Leiden (main-line algorithm);
  (2) official default recipe: Louvain + resolution=0.4 + s_neigh=8 (sensitivity only)."""
import os, time
import numpy as np
import pandas as pd
import igraph
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import spateo_loader as sl
import fast_leiden as fl

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
BASE = r"F:/BGI/task3/spateo-release-main/results/baseline/baseline_corrected_slim.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/scc"
os.makedirs(OUT, exist_ok=True)
RES = [0.5, 1.0, 1.5, 2.0]
TARGET_FRAG = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity"]
TARGET_SWALLOW = ["Spinal cord", "Dorsal root ganglion", "GI tract"]

t0 = time.time()
print("preprocess ...", flush=True)
adata = sc.read_h5ad(DATA)
adata.uns["__type"] = "UMI"
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor="seurat")
sc.pp.scale(adata, max_value=10)
ann = adata.obs["annotation"].astype(str).values
ann_cats = list(adata.obs["annotation"].cat.categories)
xy = adata.obsm["spatial"]

base = sc.read_h5ad(BASE)
assert np.array_equal(base.obs.index, adata.obs.index)

def overlap(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    n = max(n_ann, n_cl)
    Cpad = np.zeros((n, n)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    out = {}
    for i in range(n_ann):
        a = ann_cats[i]; j = ci[i]
        out[a] = round(float(ct.loc[a, clusters[j]] / ct.loc[a].sum()), 3) if j < n_cl else 0.0
    return out

def fast_louvain(adj, resolution=0.4):
    rows, cols = adj.nonzero()
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    part = G.community_multilevel(resolution=resolution)
    return np.array(part.membership, dtype=int)

# baseline overlaps (recompute for consistency)
base_ov = {res: overlap(base.obs[f"baseline_corrected_r{res}"].astype(str).values) for res in RES}

# ---- Task 1: s_neigh in {4,8,12} via spatial_adj, Leiden ----
print("SCC s_neigh in {4,8,12} (Leiden) ...", flush=True)
unions = {}
if "X_pca" in adata.obsm:
    del adata.obsm["X_pca"]
for s in [4, 8, 12]:
    unions[s] = sl.utils.spatial_adj(adata, spatial_key="spatial", pca_key="pca",
                                     e_neigh=30, s_neigh=s, n_pca_components=30)
    print(f"    s_neigh={s}: union nnz={unions[s].nnz}", flush=True)

scc_labels = {}
for s in [4, 8, 12]:
    for res in RES:
        scc_labels[(s, res)] = fl.fast_leiden(unions[s], res)

# ---- Task 2: Louvain res=0.4 s_neigh=8 ----
print("official recipe: Louvain res=0.4 s_neigh=8 ...", flush=True)
louvain_lab = fast_louvain(unions[8], resolution=0.4)

# ---- evaluation ----
print("\n=== Task 1: s_neigh {4,8,12} vs corrected baseline (res=1.0) ===\n", flush=True)
rows = []
for s in [4, 8, 12]:
    for res in RES:
        lab = scc_labels[(s, res)]
        ari = adjusted_rand_score(ann, lab.astype(str))
        nmi = normalized_mutual_info_score(ann, lab.astype(str))
        ov = overlap(lab)
        r = {"s_neigh": s, "res": res, "n_clusters": len(np.unique(lab)),
             "ARI": round(ari, 4), "NMI": round(nmi, 4)}
        for a in TARGET_FRAG + TARGET_SWALLOW:
            r[a] = f"{ov[a]} (base {base_ov[res][a]}, d{ov[a]-base_ov[res][a]:+.2f})"
        rows.append(r)
df1 = pd.DataFrame(rows)
print(df1.to_string(index=False), flush=True)
df1.to_csv(os.path.join(OUT, "scc_s4812_summary.csv"), index=False)

print("\n=== Task 2: official recipe (Louvain res=0.4 s_neigh=8) ===", flush=True)
ari_l = adjusted_rand_score(ann, louvain_lab.astype(str))
nmi_l = normalized_mutual_info_score(ann, louvain_lab.astype(str))
ov_l = overlap(louvain_lab)
print(f"    n_clusters={len(np.unique(louvain_lab))}  ARI={ari_l:.4f}  NMI={nmi_l:.4f}", flush=True)
print("    per-tissue overlap (SCC louvain vs baseline r1.0):", flush=True)
for a in ann_cats:
    b = base_ov[1.0][a]
    print(f"      {a:22s}: {ov_l[a]:.3f} (baseline {b:.3f}, d{ov_l[a]-b:+.3f})", flush=True)

# save labels
slim = sc.AnnData(X=None, obs=adata.obs[["annotation"]].copy())
for s in [4, 8, 12]:
    for res in RES:
        slim.obs[f"scc_s{s}_r{res}"] = scc_labels[(s, res)].astype(str)
slim.obs["scc_louvain_res0.4_s8"] = louvain_lab.astype(str)
slim.obsm["spatial"] = xy.copy()
slim.write_h5ad(os.path.join(OUT, "scc_s4812_slim.h5ad"))

print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
