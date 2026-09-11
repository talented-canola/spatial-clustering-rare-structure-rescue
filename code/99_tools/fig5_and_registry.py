"""Fig 5: low-degree nodes of bilateral kernel overlaid on annotation spatial. + registry update."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

RES = r"F:/BGI/task3/spateo-release-main/results"
OUT = os.path.join(RES, "analysis", "figures")

B2 = sc.read_h5ad(os.path.join(RES, "baseline_v2_arpack", "data", "baseline_corrected_slim.h5ad"))
ann = B2.obs["annotation"].astype(str).values
ann_cats = list(B2.obs["annotation"].cat.categories)
xy = B2.obsm["spatial"]
X_pca = B2.obsm["X_pca"]
n = xy.shape[0]
pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}

# bilateral kernel (expr-KNN 30), weighted degree
nn = NearestNeighbors(n_neighbors=31).fit(X_pca)
d, idx = nn.kneighbors(X_pca); idx = idx[:, 1:]; edist = d[:, 1:]
spat = np.linalg.norm(xy[idx] - xy[:, None, :], axis=2)
ss = np.median(spat); se = np.median(edist)
w = np.exp(-(spat**2)/(2*ss**2)) * np.exp(-(edist**2)/(2*se**2))
rows = np.repeat(np.arange(n), 30); cols = idx.ravel()
adj = csr_matrix((w.ravel(), (rows, cols)), shape=(n, n))
wdeg = np.asarray(adj.sum(axis=1)).ravel()
thr = np.percentile(wdeg, 10)
low = wdeg <= thr
print(f"low-degree threshold (10th pct) = {thr:.2f}, {low.sum()} nodes flagged ({low.mean()*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# left: annotation with low-degree nodes overlaid in black
for a in ann_cats:
    m = (ann == a)
    axes[0].scatter(xy[m, 0], xy[m, 1], s=2, color=ann_colors[a], rasterized=True)
axes[0].scatter(xy[low, 0], xy[low, 1], s=6, color="black", label="low-degree nodes", rasterized=True)
axes[0].set_title("Annotation + low-degree nodes (black)"); axes[0].set_aspect("equal"); axes[0].invert_yaxis()
axes[0].legend(markerscale=3, fontsize=8)
# right: degree as color (spatial)
sc_ = axes[1].scatter(xy[:, 0], xy[:, 1], s=2, c=wdeg, cmap="viridis", rasterized=True)
axes[1].set_title("Bilateral weighted degree (spatial)"); axes[1].set_aspect("equal"); axes[1].invert_yaxis()
plt.colorbar(sc_, ax=axes[1])
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig5_low_degree_nodes.png"), dpi=130); plt.close()
print("fig5 saved")

# ---- registry update: add graphst_reference + method1 variants ----
reg = os.path.join(RES, "experiment_registry.csv")
df = pd.read_csv(reg, keep_default_na=False)
new_rows = []
# graphst_reference
graphst = {"0.5": (20, 0.3265, 0.5560, 0.9356), "1.0": (30, 0.3043, 0.5779, 0.9193),
           "1.5": (40, 0.2639, 0.5738, 0.9053), "2.0": (48, 0.2484, 0.5810, 0.8951)}
for rs, (nc, ari, nmi, coh) in graphst.items():
    new_rows.append({"方案ID": "graphst_reference", "聚类算法": "Leiden", "PCA求解器": "N/A(GAE)",
                     "PCA维度": "N/A", "e_neigh": 30, "s_neigh": 3, "resolution": rs,
                     "是否用了spatial_adj": "否", "整体ARI": ari, "整体NMI": nmi,
                     "Silhouette": "未计算", "空间连贯性": coh,
                     "完整19类重叠表路径": "缺失", "空间图路径": "缺失", "混淆矩阵路径": "缺失",
                     "UMAP路径": "analysis/figures/fig3_graphst_umap_z_vs_h.png" if rs == "1.0" else "缺失",
                     "marker基因验证路径": "缺失", "状态": "部分完成"})
# method1 variants
m1 = {"method1_bilateral_spatial_knn": {"0.5": (32, 0.1948, 0.4751, 0.9567), "1.0": (51, 0.1784, 0.4885, 0.9416),
                                          "1.5": (68, 0.1461, 0.4924, 0.9312), "2.0": (76, 0.1369, 0.4967, 0.9271)},
      "method1_bilateral_expr_knn": {"0.5": (35, 0.3146, 0.6150, 0.8974), "1.0": (48, 0.2909, 0.6191, 0.8815),
                                       "1.5": (58, 0.2507, 0.6053, 0.8643), "2.0": (70, 0.2360, 0.6171, 0.8533)}}
for sid, data in m1.items():
    for rs, (nc, ari, nmi, coh) in data.items():
        new_rows.append({"方案ID": sid, "聚类算法": "Leiden(加权)", "PCA求解器": "arpack",
                         "PCA维度": 30, "e_neigh": 30, "s_neigh": 8, "resolution": rs,
                         "是否用了spatial_adj": "否(双边核加权)", "整体ARI": ari, "整体NMI": nmi,
                         "Silhouette": "未计算", "空间连贯性": coh,
                         "完整19类重叠表路径": "缺失", "空间图路径": "缺失", "混淆矩阵路径": "缺失",
                         "UMAP路径": "缺失", "marker基因验证路径": "缺失", "状态": "部分完成"})
df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
df.to_csv(reg, index=False, encoding="utf-8-sig")
print(f"registry updated: {len(df)} rows total")
