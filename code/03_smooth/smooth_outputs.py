"""smooth_s8: produce all figures/tables/metrics + README + registry update."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import seaborn as sns

RES = r"F:/BGI/task3/spateo-release-main/results"
SID = "smooth_s8"
OUT = os.path.join(RES, SID)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)

slim = sc.read_h5ad(os.path.join(OUT, "data", "smooth_s8_slim.h5ad"))
ann = slim.obs["annotation"].astype(str).values
ann_cats = list(slim.obs["annotation"].cat.categories)
xy = slim.obsm["spatial"]
X_pca = slim.obsm["X_pca"]
conn = slim.obsp["expression_connectivities"]
n = xy.shape[0]
RESOLUTIONS = [0.5, 1.0, 1.5, 2.0]

pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}

# spatial KNN (k=8) for coherence
knn = NearestNeighbors(n_neighbors=9).fit(xy)
_, nidx = knn.kneighbors(xy); nidx = nidx[:, 1:]

rng = np.random.RandomState(0)
sub = rng.choice(n, size=min(6000, n), replace=False)

def coherence(labels):
    li = labels.astype(int).values
    return float((li[nidx] == li[:, None]).mean())

def hungarian(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    N = max(n_ann, n_cl)
    Cpad = np.zeros((N, N)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    ov = {}
    ann2cl = {}
    for i in range(n_ann):
        a = ann_cats[i]; j = ci[i]
        if j < n_cl:
            ov[a] = round(float(ct.loc[a, clusters[j]] / ct.loc[a].sum()), 3)
            ann2cl[a] = clusters[j]
        else:
            ov[a] = 0.0; ann2cl[a] = None
    return ct, ann2cl, ov

# UMAP (res=1.0, on smooth_s8's own connectivities)
tmp = sc.AnnData(X=X_pca.copy())
tmp.obsp["connectivities"] = conn
tmp.obsp["distances"] = conn.copy()
tmp.uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                        "params": {"n_neighbors": 30, "method": "umap"}}
sc.tl.umap(tmp, random_state=42)
um = tmp.obsm["X_umap"]

metrics_rows = []
for res in RESOLUTIONS:
    rs = str(res)
    lab = slim.obs[f"{SID}_r{rs}"].astype(str).values
    ari = adjusted_rand_score(ann, lab)
    nmi = normalized_mutual_info_score(ann, lab)
    sil = silhouette_score(X_pca[sub], lab.astype(int)[sub])
    coh = coherence(slim.obs[f"{SID}_r{rs}"])
    metrics_rows.append({"resolution": rs, "n_clusters": len(np.unique(lab)),
                         "ARI": round(ari, 4), "NMI": round(nmi, 4),
                         "Silhouette": round(sil, 4), "空间连贯性": round(coh, 4)})

    # 19-class overlap table + confusion csv
    ct, ann2cl, ov = hungarian(lab)
    pd.DataFrame([{"annotation": a, "smooth_s8_cluster": ann2cl[a], "overlap_fraction": ov[a]}
                  for a in ann_cats]).to_csv(os.path.join(OUT, "tables", f"matching_r{rs}.csv"), index=False)
    ct.to_csv(os.path.join(OUT, "tables", f"confusion_r{rs}.csv"))

    # confusion heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.3, ax=ax, annot_kws={"size": 6})
    ax.set_title(f"Confusion matrix: annotation x {SID} (res={rs})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "figures", f"confusion_r{rs}.png"), dpi=120); plt.close()

    # spatial aligned
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for a in ann_cats:
        m = (ann == a)
        axes[0].scatter(xy[m, 0], xy[m, 1], s=2, color=ann_colors[a], rasterized=True)
    axes[0].set_title("Annotation"); axes[0].set_aspect("equal"); axes[0].invert_yaxis()
    cl2col = {cl: ann_colors[a] for a, cl in ann2cl.items() if cl is not None}
    matched = {c for c in ann2cl.values() if c is not None}
    leftover = [c for c in list(ct.columns) if c not in matched]
    for i, cl in enumerate(leftover):
        cl2col[cl] = pool[19 + i]
    for cl in list(ct.columns):
        m = (lab == cl)
        axes[1].scatter(xy[m, 0], xy[m, 1], s=2, color=cl2col[cl], rasterized=True)
    axes[1].set_title(f"{SID} (res={rs}, re-colored)"); axes[1].set_aspect("equal"); axes[1].invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "figures", f"spatial_aligned_r{rs}.png"), dpi=120); plt.close()

# UMAP figure (res=1.0): annotation vs cluster
lab10 = slim.obs[f"{SID}_r1.0"].astype(str).values
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for a in ann_cats:
    m = (ann == a)
    axes[0].scatter(um[m, 0], um[m, 1], s=2, color=ann_colors[a], rasterized=True)
axes[0].set_title("UMAP by annotation")
axes[1].scatter(um[:, 0], um[:, 1], s=2, c=lab10.astype(int), cmap=plt.cm.turbo, rasterized=True)
axes[1].set_title(f"UMAP by {SID}")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figures", "umap.png"), dpi=120); plt.close()

# metrics table
mdf = pd.DataFrame(metrics_rows)
mdf.to_csv(os.path.join(OUT, "tables", "overall_metrics.csv"), index=False)
print("=== smooth_s8 overall metrics ===")
print(mdf.to_string(index=False))

# README
lines = [f"# 方案：{SID}", "", "## 参数",
         "- 聚类算法：Leiden", "- PCA求解器：arpack", "- PCA维度：30",
         "- e_neigh：30", "- s_neigh：8（平滑用空间邻居数）",
         "- 是否用了空间约束：是（表达层空间平滑，非图融合 spatial_adj）",
         "- 空间信息利用方式：高斯核（σ=中位邻居距离）对 log1p 表达做 8 空间邻居加权平均，再走 baseline 全基因 PCA(arpack-30)→neighbors(e_neigh=30)→Leiden", "",
         "## 各分辨率关键指标",
         "| resolution | 簇数 | ARI | NMI | Silhouette | 空间连贯性 |",
         "|---|---|---|---|---|---|"]
for _, r in mdf.iterrows():
    lines.append(f"| {r['resolution']} | {r['n_clusters']} | {r['ARI']} | {r['NMI']} | {r['Silhouette']} | {r['空间连贯性']} |")
lines += ["", "## 本方案产出文件清单",
          "- figures/: spatial_aligned_r*.png, confusion_r*.png, umap.png",
          "- tables/: matching_r*.csv, confusion_r*.csv, overall_metrics.csv",
          "- data/: smooth_s8_slim.h5ad", "",
          "## 已知问题/待办",
          "平滑是表达层面的去噪，与 SCC 的图融合是两种哲学；是否也吞并小类见分组织分析。"]
with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"wrote {SID}/README.md")

# update registry
reg_path = os.path.join(RES, "experiment_registry.csv")
df = pd.read_csv(reg_path, keep_default_na=False)
for _, r in mdf.iterrows():
    rs = r["resolution"]
    df = pd.concat([df, pd.DataFrame([{
        "方案ID": SID, "聚类算法": "Leiden", "PCA求解器": "arpack", "PCA维度": 30,
        "e_neigh": 30, "s_neigh": 8, "resolution": rs, "是否用了spatial_adj": "否",
        "整体ARI": r["ARI"], "整体NMI": r["NMI"], "Silhouette": r["Silhouette"],
        "空间连贯性": r["空间连贯性"],
        "完整19类重叠表路径": f"{SID}/tables/matching_r{rs}.csv",
        "空间图路径": f"{SID}/figures/spatial_aligned_r{rs}.png",
        "混淆矩阵路径": f"{SID}/figures/confusion_r{rs}.png",
        "UMAP路径": f"{SID}/figures/umap.png" if rs == "1.0" else "缺失",
        "marker基因验证路径": "缺失", "状态": "已完成" if rs == "1.0" else "部分完成",
    }])], ignore_index=True)
df.to_csv(reg_path, index=False, encoding="utf-8-sig")
print("registry updated")
