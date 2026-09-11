"""Create scheme folders for graphst_reference + method1 variants, migrate results,
generate tables/README, update registry. Same structure as existing schemes."""
import os, shutil
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
RESOL = [0.5, 1.0, 1.5, 2.0]

SCHEMES = {
    "graphst_reference": {
        "slim": os.path.join(RES, "graphst_reference", "data", "graphst_reference_slim.h5ad"),
        "label_tpl": "graphst_reference_r{res}",
        "emb_key": "emb",
        "algo": "Leiden(GraphST嵌入)", "solver": "N/A(GAE)", "dim": "64", "e": 30, "s": 3,
        "spatial": "GraphST 图自编码器(DGI式对比+GAE)，空间KNN(k=3)训练，在其输出表示上聚类",
        "todo": "SOTA未迁移：ARI 0.30 远低于 baseline 0.42/SCC 0.44。注意 GraphST 代码把 3000 维重构(而非 64 维隐层)存为 obsm['emb']，已核对两种表示聚类结果均不佳。",
    },
    "method1_bilateral_spatial_knn": {
        "slim": os.path.join(RES, "analysis", "method1_bilateral_spatial_knn_slim.h5ad"),
        "label_tpl": "method1_bilateral_spatial_knn_r{res}",
        "emb_key": "X_pca",
        "algo": "Leiden(加权)", "solver": "arpack", "dim": "30", "e": 30, "s": 8,
        "spatial": "双边核加权：w=exp(-spatial²/2σs²)·exp(-expr²/2σe²)，在 s=8 空间KNN 上",
        "todo": "负结果：ARI 0.18 远低于 baseline/SCC。乘法式空间权重破坏表达结构(已排除权重量级混淆)。",
    },
    "method1_bilateral_expr_knn": {
        "slim": os.path.join(RES, "analysis", "method1_bilateral_expr_knn_slim.h5ad"),
        "label_tpl": "method1_bilateral_expr_knn_r{res}",
        "emb_key": "X_pca",
        "algo": "Leiden(加权)", "solver": "arpack", "dim": "30", "e": 30, "s": 8,
        "spatial": "双边核加权：w=exp(-spatial²/2σs²)·exp(-expr²/2σe²)，在 30 表达KNN 上",
        "todo": "负结果：ARI 0.29 仍低于 baseline。乘法式空间权重破坏表达结构。",
    },
}

# reference annotation/spatial for coherence
B2 = sc.read_h5ad(os.path.join(RES, "baseline_v2_arpack", "data", "baseline_corrected_slim.h5ad"))
ann_cats = list(B2.obs["annotation"].cat.categories)
xy = B2.obsm["spatial"]
nn_c = NearestNeighbors(n_neighbors=9).fit(xy)
_, nidx = nn_c.kneighbors(xy); nidx = nidx[:, 1:]

def coherence(labels):
    li = labels.astype(int).values
    return float((li[nidx] == li[:, None]).mean())

def hungarian_overlap(ann, labels):
    ct = pd.crosstab(ann, labels.astype(str))
    na, nc = ct.shape; N = max(na, nc)
    C = np.zeros((N, N)); C[:na, :nc] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-C)
    cl = list(ct.columns)
    out, ann2cl = {}, {}
    for i in range(na):
        a = ann_cats[i]; j = ci[i]
        if j < nc:
            out[a] = round(float(ct.loc[a, cl[j]] / ct.loc[a].sum()), 3); ann2cl[a] = cl[j]
        else:
            out[a] = 0.0; ann2cl[a] = None
    return ct, out, ann2cl

for sid, meta in SCHEMES.items():
    slim = sc.read_h5ad(meta["slim"])
    ann = slim.obs["annotation"].astype(str).values
    os.makedirs(os.path.join(RES, sid, "figures"), exist_ok=True)
    os.makedirs(os.path.join(RES, sid, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RES, sid, "data"), exist_ok=True)
    emb = slim.obsm[meta["emb_key"]]
    rng = np.random.RandomState(0)
    sub = rng.choice(slim.n_obs, size=min(4000, slim.n_obs), replace=False)
    metrics_rows = []
    for res in RESOL:
        rs = str(res)
        lab = slim.obs[meta["label_tpl"].format(res=rs)].astype(str).values
        ari = adjusted_rand_score(ann, lab); nmi = normalized_mutual_info_score(ann, lab)
        sil = silhouette_score(emb[sub], lab.astype(int)[sub])
        coh = coherence(slim.obs[meta["label_tpl"].format(res=rs)])
        metrics_rows.append({"resolution": rs, "n_clusters": len(np.unique(lab)),
                             "ARI": round(ari, 4), "NMI": round(nmi, 4),
                             "Silhouette": round(sil, 4), "空间连贯性": round(coh, 4)})
        ct, ov, ann2cl = hungarian_overlap(ann, lab)
        pd.DataFrame([{"annotation": a, "cluster": ann2cl[a], "overlap_fraction": ov[a]} for a in ann_cats]
                     ).to_csv(os.path.join(RES, sid, "tables", f"matching_r{rs}.csv"), index=False)
        ct.to_csv(os.path.join(RES, sid, "tables", f"confusion_r{rs}.csv"))
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.3, ax=ax, annot_kws={"size": 6})
        ax.set_title(f"Confusion: annotation x {sid} (res={rs})")
        plt.tight_layout(); plt.savefig(os.path.join(RES, sid, "figures", f"confusion_r{rs}.png"), dpi=120); plt.close()
    mdf = pd.DataFrame(metrics_rows)
    mdf.to_csv(os.path.join(RES, sid, "tables", "overall_metrics.csv"), index=False)

    # README
    lines = [f"# 方案：{sid}", "", "## 参数",
             f"- 聚类算法：{meta['algo']}", f"- PCA求解器：{meta['solver']}", f"- PCA维度：{meta['dim']}",
             f"- e_neigh：{meta['e']}", f"- s_neigh：{meta['s']}",
             f"- 是否用了空间约束：是" if "graphst" not in sid and "双边" in meta["spatial"] else "- 是否用了空间约束：是",
             f"- 空间信息利用方式：{meta['spatial']}", "",
             "## 各分辨率关键指标",
             "| resolution | 簇数 | ARI | NMI | Silhouette | 空间连贯性 |", "|---|---|---|---|---|---|"]
    for _, r in mdf.iterrows():
        lines.append(f"| {r['resolution']} | {r['n_clusters']} | {r['ARI']} | {r['NMI']} | {r['Silhouette']} | {r['空间连贯性']} |")
    lines += ["", "## 本方案产出文件清单",
              "- figures/: confusion_r*.png", "- tables/: matching_r*.csv, confusion_r*.csv, overall_metrics.csv",
              "- data/: slim h5ad", "", "## 已知问题/待办", meta["todo"], ""]
    with open(os.path.join(RES, sid, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {sid}/ (tables + README)")

# move slim h5ad files into scheme data/
shutil.move(os.path.join(RES, "analysis", "method1_bilateral_spatial_knn_slim.h5ad"),
            os.path.join(RES, "method1_bilateral_spatial_knn", "data", "method1_bilateral_spatial_knn_slim.h5ad"))
shutil.move(os.path.join(RES, "analysis", "method1_bilateral_expr_knn_slim.h5ad"),
            os.path.join(RES, "method1_bilateral_expr_knn", "data", "method1_bilateral_expr_knn_slim.h5ad"))
# move fig3 -> graphst_reference/figures, fig5 -> method1_bilateral_expr_knn/figures
shutil.move(os.path.join(RES, "analysis", "figures", "fig3_graphst_umap_z_vs_h.png"),
            os.path.join(RES, "graphst_reference", "figures", "umap_z_vs_h.png"))
shutil.move(os.path.join(RES, "analysis", "figures", "fig5_low_degree_nodes.png"),
            os.path.join(RES, "method1_bilateral_expr_knn", "figures", "low_degree_nodes.png"))
print("moved slim h5ad + fig3 + fig5")

# update registry paths
reg = os.path.join(RES, "experiment_registry.csv")
df = pd.read_csv(reg, keep_default_na=False)
for sid in SCHEMES:
    for rs in [str(x) for x in RESOL]:
        m = (df["方案ID"] == sid) & (df["resolution"] == rs)
        df.loc[m, "完整19类重叠表路径"] = f"{sid}/tables/matching_r{rs}.csv"
        df.loc[m, "混淆矩阵路径"] = f"{sid}/figures/confusion_r{rs}.png"
        df.loc[m, "Silhouette"] = "见 overall_metrics.csv"
df.loc[(df["方案ID"] == "graphst_reference") & (df["resolution"] == "1.0"), "UMAP路径"] = "graphst_reference/figures/umap_z_vs_h.png"
df.to_csv(reg, index=False, encoding="utf-8-sig")
print(f"registry updated: {len(df)} rows")
