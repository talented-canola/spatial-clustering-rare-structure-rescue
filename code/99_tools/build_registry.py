"""Task A (final): rebuild results/experiment_registry.csv with metrics + all existing paths."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

RES_ROOT = r"F:/BGI/task3/spateo-release-main/results"
B1 = sc.read_h5ad(os.path.join(RES_ROOT, "baseline", "baseline_slim.h5ad"))
B2 = sc.read_h5ad(os.path.join(RES_ROOT, "baseline", "baseline_corrected_slim.h5ad"))
SCC = sc.read_h5ad(os.path.join(RES_ROOT, "scc", "scc_slim.h5ad"))
SCC4812 = sc.read_h5ad(os.path.join(RES_ROOT, "scc", "scc_s4812_slim.h5ad"))

ann = B2.obs["annotation"].astype(str).values
xy = B2.obsm["spatial"]
n = xy.shape[0]
knn = NearestNeighbors(n_neighbors=9).fit(xy)
_, nidx = knn.kneighbors(xy); nidx = nidx[:, 1:]
X_arp = B2.obsm["X_pca"]
X_rand = B1.obsm["X_pca"]
rng = np.random.RandomState(0)
sub = rng.choice(n, size=min(6000, n), replace=False)

def metrics(labels, X):
    ari = adjusted_rand_score(ann, labels.astype(str).values)
    nmi = normalized_mutual_info_score(ann, labels.astype(str).values)
    sil = silhouette_score(X[sub], labels.astype(int).values[sub])
    li = labels.astype(int).values
    coh = float((li[nidx] == li[:, None]).mean())
    return round(ari, 4), round(nmi, 4), round(sil, 4), round(coh, 4)

schemes = [
    ("baseline_v1",        "Leiden", "randomized", 50, 30, "N/A", "否", [0.5,1.0,1.5,2.0], B1,     "baseline_leiden_r{res}",    X_rand),
    ("baseline_v2_arpack", "Leiden", "arpack",     30, 30, "N/A", "否", [0.5,1.0,1.5,2.0], B2,     "baseline_corrected_r{res}", X_arp),
    ("scc_default",        "Leiden", "arpack",     30, 30, 6,     "是", [0.5,1.0,1.5,2.0], SCC,    "scc_s6_r{res}",             X_arp),
    ("scc_s4",             "Leiden", "arpack",     30, 30, 4,     "是", [0.5,1.0,1.5,2.0], SCC4812,"scc_s4_r{res}",             X_arp),
    ("scc_s8",             "Leiden", "arpack",     30, 30, 8,     "是", [0.5,1.0,1.5,2.0], SCC4812,"scc_s8_r{res}",             X_arp),
    ("scc_s12",            "Leiden", "arpack",     30, 30, 12,    "是", [0.5,1.0,1.5,2.0], SCC4812,"scc_s12_r{res}",            X_arp),
    ("scc_official",       "Louvain","arpack",     30, 30, 8,     "是", [0.4],                  SCC4812,"scc_louvain_res0.4_s8",     X_arp),
]

def paths(sid, rs):
    p = {"完整19类重叠表路径": "缺失", "空间图路径": "缺失", "混淆矩阵路径": "缺失",
         "UMAP路径": "缺失", "marker基因验证路径": "缺失", "状态": "部分完成"}
    if sid == "baseline_v1":
        p["完整19类重叠表路径"] = f"baseline/matching_baseline_leiden_r{rs}.csv"
        p["空间图路径"] = f"baseline/spatial_aligned_baseline_leiden_r{rs}.png"
        p["混淆矩阵路径"] = f"baseline/confusion_baseline_leiden_r{rs}.png"
        if rs == "1.0":
            p["UMAP路径"] = "baseline/umap_annotation_vs_baseline.png"
    elif sid == "baseline_v2_arpack":
        p["完整19类重叠表路径"] = f"baseline/matching_corrected_r{rs}.csv"
        if rs == "1.0":
            p["空间图路径"] = "baseline/spatial_aligned_baseline_v2.png"
            p["混淆矩阵路径"] = "baseline/confusion_baseline_v2.png"
            p["UMAP路径"] = "baseline/umap_baseline_v2.png"
            p["marker基因验证路径"] = "marker_verification/top_markers_baseline_v2.csv"
            p["状态"] = "已完成"
    elif sid == "scc_default" and rs == "1.0":
        p.update({"完整19类重叠表路径": "scc/matching_scc_default.csv",
                  "空间图路径": "scc/spatial_aligned_scc_default.png",
                  "混淆矩阵路径": "scc/confusion_scc_default.png",
                  "UMAP路径": "scc/umap_scc_default.png",
                  "marker基因验证路径": "marker_verification/top_markers_scc_default.csv",
                  "状态": "已完成"})
    elif sid == "scc_s8" and rs == "1.0":
        p.update({"完整19类重叠表路径": "scc/matching_scc_s8.csv",
                  "空间图路径": "scc/spatial_aligned_scc_s8.png",
                  "混淆矩阵路径": "scc/confusion_scc_s8.png",
                  "UMAP路径": "scc/umap_scc_s8.png"})
    elif sid == "scc_official" and rs == "0.4":
        p.update({"完整19类重叠表路径": "scc/matching_scc_official.csv",
                  "空间图路径": "scc/spatial_aligned_scc_official.png",
                  "混淆矩阵路径": "scc/confusion_scc_official.png",
                  "UMAP路径": "scc/umap_scc_official.png"})
    return p

rows = []
for (sid, algo, solver, dim, e, s, spa, ress, adata, tpl, X) in schemes:
    for res in ress:
        rs = str(res)
        lab = adata.obs[tpl.format(res=rs)]
        ari, nmi, sil, coh = metrics(lab, X)
        row = {"方案ID": sid, "聚类算法": algo, "PCA求解器": solver, "PCA维度": dim,
               "e_neigh": e, "s_neigh": s, "resolution": rs, "是否用了spatial_adj": spa,
               "整体ARI": ari, "整体NMI": nmi, "Silhouette": sil, "空间连贯性": coh}
        row.update(paths(sid, rs))
        rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(RES_ROOT, "experiment_registry.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"registry rebuilt: {len(df)} rows -> {out}")
