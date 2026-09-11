"""Produce evidence figures 1 (7-tissue overlap heatmap), 2 (ARI/NMI bar), 4 (degree dist).
Also save method1 bilateral labels to slim h5ad + compute all methods' metrics."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import fast_leiden as fl

RES = r"F:/BGI/task3/spateo-release-main/results"
OUT = os.path.join(RES, "analysis", "figures")
os.makedirs(OUT, exist_ok=True)

# reference for X_pca + spatial + annotation
B2 = sc.read_h5ad(os.path.join(RES, "baseline_v2_arpack", "data", "baseline_corrected_slim.h5ad"))
ann = B2.obs["annotation"].astype(str).values
ann_cats = list(B2.obs["annotation"].cat.categories)
xy = B2.obsm["spatial"]
X_pca = B2.obsm["X_pca"]
n = xy.shape[0]

TISSUES = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord", "Dorsal root ganglion", "GI tract"]

def overlap7(labels):
    ct = pd.crosstab(ann, labels.astype(str))
    na, nc = ct.shape; N = max(na, nc)
    C = np.zeros((N, N)); C[:na, :nc] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-C)
    cl = list(ct.columns)
    return {ann_cats[i]: round(float(ct.loc[ann_cats[i], cl[ci[i]]] / ct.loc[ann_cats[i]].sum()), 3) if ci[i] < nc else 0.0 for i in range(na)}

# ---- compute method1 bilateral labels (res=1.0) ----
def bilateral_labels(nn_kind):
    if nn_kind == "spatial":
        nn = NearestNeighbors(n_neighbors=9).fit(xy)  # s=8
        d, idx = nn.kneighbors(xy); idx = idx[:, 1:]; spat = d[:, 1:]
        expr = np.linalg.norm(X_pca[idx] - X_pca[:, None, :], axis=2)
        k = 8
    else:  # expr-KNN 30
        nn = NearestNeighbors(n_neighbors=31).fit(X_pca)
        d, idx = nn.kneighbors(X_pca); idx = idx[:, 1:]; expr = d[:, 1:]
        spat = np.linalg.norm(xy[idx] - xy[:, None, :], axis=2)
        k = 30
    ss = np.median(spat); se = np.median(expr)
    w = np.exp(-(spat**2)/(2*ss**2)) * np.exp(-(expr**2)/(2*se**2))
    rows = np.repeat(np.arange(n), k); cols = idx.ravel()
    adj = csr_matrix((w.ravel(), (rows, cols)), shape=(n, n))
    return {res: fl.fast_leiden_weighted(adj, res) for res in [0.5, 1.0, 1.5, 2.0]}, adj

m1_spatial, adj_spatial = bilateral_labels("spatial")
m1_expr, adj_expr = bilateral_labels("expr")
# save method1 slim
for name, labels in [("method1_bilateral_spatial_knn", m1_spatial), ("method1_bilateral_expr_knn", m1_expr)]:
    slim = sc.AnnData(X=None, obs=B2.obs[["annotation"]].copy())
    for res in [0.5, 1.0, 1.5, 2.0]:
        slim.obs[f"{name}_r{res}"] = labels[res].astype(str)
    slim.obsm["spatial"] = xy.copy(); slim.obsm["X_pca"] = X_pca.copy()
    slim.write_h5ad(os.path.join(RES, "analysis", f"{name}_slim.h5ad"))

# ---- load all method labels at res=1.0 ----
methods = {}
methods["baseline_v2"] = B2.obs["baseline_corrected_r1.0"].astype(str).values
SCC = sc.read_h5ad(os.path.join(RES, "scc_s4", "data", "scc_s4812_slim.h5ad"))
methods["scc_s4"] = SCC.obs["scc_s4_r1.0"].astype(str).values
methods["scc_s8"] = SCC.obs["scc_s8_r1.0"].astype(str).values
methods["scc_s12"] = SCC.obs["scc_s12_r1.0"].astype(str).values
methods["smooth_s8"] = sc.read_h5ad(os.path.join(RES, "smooth_s8", "data", "smooth_s8_slim.h5ad")).obs["smooth_s8_r1.0"].astype(str).values
methods["smooth_s8_incl_self"] = sc.read_h5ad(os.path.join(RES, "smooth_s8_incl_self", "data", "smooth_s8_incl_self_slim.h5ad")).obs["smooth_s8_incl_self_r1.0"].astype(str).values
methods["graphst_reference"] = sc.read_h5ad(os.path.join(RES, "graphst_reference", "data", "graphst_reference_slim.h5ad")).obs["graphst_reference_r1.0"].astype(str).values
methods["method1_bilateral_spatial_knn"] = m1_spatial[1.0].astype(str)
methods["method1_bilateral_expr_knn"] = m1_expr[1.0].astype(str)

# metrics
metrics = {}
for name, lab in methods.items():
    metrics[name] = (adjusted_rand_score(ann, lab), normalized_mutual_info_score(ann, lab))

# ---- FIG 1: 7-tissue overlap heatmap ----
names = list(methods.keys())
ov_mat = np.zeros((len(TISSUES), len(names)))
for j, name in enumerate(names):
    ov = overlap7(methods[name])
    for i, t in enumerate(TISSUES):
        ov_mat[i, j] = ov[t]
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(ov_mat, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(TISSUES))); ax.set_yticklabels(TISSUES, fontsize=10)
for i in range(len(TISSUES)):
    for j in range(len(names)):
        ax.text(j, i, f"{ov_mat[i,j]:.2f}", ha="center", va="center", fontsize=7)
ax.set_title("7 key tissues overlap (res=1.0) — 0=red, 1=blue")
plt.colorbar(im, ax=ax)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_tissue_overlap_heatmap.png"), dpi=130); plt.close()

# ---- FIG 2: ARI/NMI bar ----
order = sorted(names, key=lambda m: metrics[m][0], reverse=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, title in [(axes[0], 0, "ARI"), (axes[1], 1, "NMI")]:
    vals = [metrics[m][key] for m in order]
    colors = ["#d62728" if m == "scc_s8" else "#1f77b4" for m in order]
    ax.barh(range(len(order)), vals, color=colors)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.set_title(title); ax.set_xlim(0, max(vals) * 1.15)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
plt.suptitle("Overall ARI / NMI across methods (res=1.0), scc_s8 highlighted")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_ari_nmi_bar.png"), dpi=130); plt.close()

# ---- FIG 4: degree distribution (SCC union vs bilateral expr-KNN) ----
# SCC union graph (binary): expr-KNN(30) union spatial-KNN(8)
nn = NearestNeighbors(n_neighbors=31).fit(X_pca)
_, eidx = nn.kneighbors(X_pca); eidx = eidx[:, 1:]
nn2 = NearestNeighbors(n_neighbors=9).fit(xy)
_, sidx = nn2.kneighbors(xy); sidx = sidx[:, 1:]
rows = np.concatenate([np.repeat(np.arange(n), 30), np.repeat(np.arange(n), 8)])
cols = np.concatenate([eidx.ravel(), sidx.ravel()])
scc_union = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
scc_deg = np.asarray(scc_union.sum(axis=1)).ravel()   # binary degree
bilateral_deg = np.asarray(adj_expr.sum(axis=1)).ravel()  # weighted degree (expr-KNN bilateral)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(scc_deg, bins=50, color="#1f77b4"); axes[0].set_title(f"SCC union (binary) degree, mean={scc_deg.mean():.1f}")
axes[1].hist(bilateral_deg, bins=50, color="#d62728"); axes[1].set_title(f"Bilateral kernel weighted degree, mean={bilateral_deg.mean():.1f}")
for ax in axes: ax.set_xlabel("(weighted) degree"); ax.set_ylabel("# nodes")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_degree_distribution.png"), dpi=130); plt.close()

print("ARI/NMI summary (res=1.0):")
for m in order:
    print(f"  {m:32s} ARI={metrics[m][0]:.4f} NMI={metrics[m][1]:.4f}")
print(f"\nfigures saved to {OUT}")
print(f"SCC union mean degree={scc_deg.mean():.2f}, bilateral mean weighted degree={bilateral_deg.mean():.2f}")
