"""Generate per-scheme README.md and update experiment_registry.csv paths."""
import os
import pandas as pd

RES = r"F:/BGI/task3/spateo-release-main/results"
os.chdir(RES)

df = pd.read_csv("experiment_registry.csv", keep_default_na=False)

# scheme metadata for README (spatial method + known issues) — hardcoded facts, not fabricated numbers
META = {
    "baseline_v1": ("无（仅表达空间 KNN 图；PCA 用 randomized SVD）",
                    "已被 baseline_v2_arpack 取代（randomized PCA 降维不准）；仅作历史对照，marker 验证未做。"),
    "baseline_v2_arpack": ("无（仅表达空间 KNN 图；PCA 用 arpack）",
                           "修正版 baseline，作为唯一基准。res=0.5 因 18 簇<19 组织为粒度效应、不可比（见 analysis/）。"),
    "scc_default": ("spatial_adj：表达 KNN 图 ∪ 空间 KNN 图（s_neigh=6）二值并集",
                    "官方默认 s_neigh=6。小类吞噬风险高（Spinal cord/DRG/GI tract）。"),
    "scc_s4": ("spatial_adj：表达 KNN 图 ∪ 空间 KNN 图（s_neigh=4）二值并集",
               "无 figures/tables（仅指标）。labels 在 data/scc_s4812_slim.h5ad（与 s8/s12/official 共享）。"),
    "scc_s8": ("spatial_adj：表达 KNN 图 ∪ 空间 KNN 图（s_neigh=8）二值并集",
               "Stereo-seq 官方推荐 s_neigh=8。labels 在 scc_s4/data/scc_s4812_slim.h5ad。"),
    "scc_s12": ("spatial_adj：表达 KNN 图 ∪ 空间 KNN 图（s_neigh=12）二值并集",
                "无产出文件（仅指标）。labels 在 scc_s4/data/scc_s4812_slim.h5ad。"),
    "scc_official": ("spatial_adj（s_neigh=8）并集图上跑 Louvain（resolution=0.4）",
                     "官方教程配方，仅 14 簇、切得较粗。labels 在 scc_s4/data/scc_s4812_slim.h5ad。"),
}

def list_dir(d):
    p = os.path.join(RES, d)
    if not os.path.isdir(p):
        return []
    out = []
    for root, _, files in os.walk(p):
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), p))
    return sorted(out)

# generate READMEs
for sid, (spatial_method, todo) in META.items():
    sub = df[df["方案ID"] == sid]
    if sub.empty:
        continue
    r0 = sub.iloc[0]
    lines = [f"# 方案：{sid}", "", "## 参数",
             f"- 聚类算法：{r0['聚类算法']}",
             f"- PCA求解器：{r0['PCA求解器']}",
             f"- PCA维度：{r0['PCA维度']}",
             f"- e_neigh：{r0['e_neigh']}",
             f"- s_neigh：{r0['s_neigh']}",
             f"- 是否用了空间约束：{'是' if r0['是否用了spatial_adj']=='是' else '否'}",
             f"- 空间信息利用方式：{spatial_method}", "",
             "## 各分辨率关键指标",
             "| resolution | ARI | NMI | Silhouette | 空间连贯性 |",
             "|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        lines.append(f"| {r['resolution']} | {r['整体ARI']} | {r['整体NMI']} | {r['Silhouette']} | {r['空间连贯性']} |")
    lines += ["", "## 本方案产出文件清单"]
    for subdir in ["figures", "tables", "data"]:
        files = list_dir(os.path.join(sid, subdir))
        if files:
            lines.append(f"- {subdir}/:")
            for f in files:
                lines.append(f"  - {f}")
    lines += ["", "## 已知问题/待办", todo, ""]
    os.makedirs(sid, exist_ok=True)
    with open(os.path.join(sid, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {sid}/README.md")

# update registry path columns to new structure
def new_paths(sid, res):
    p = {"完整19类重叠表路径": "缺失", "空间图路径": "缺失", "混淆矩阵路径": "缺失",
         "UMAP路径": "缺失", "marker基因验证路径": "缺失"}
    tbl = lambda f: f"{sid}/tables/{f}"
    fig = lambda f: f"{sid}/figures/{f}"
    if sid == "baseline_v1":
        p["完整19类重叠表路径"] = tbl(f"matching_r{res}.csv")
        p["空间图路径"] = fig(f"spatial_aligned_r{res}.png")
        p["混淆矩阵路径"] = fig(f"confusion_r{res}.png")
        if res == "1.0":
            p["UMAP路径"] = fig("umap.png")
    elif sid == "baseline_v2_arpack":
        p["完整19类重叠表路径"] = tbl(f"matching_r{res}.csv")
        if res == "1.0":
            p["空间图路径"] = fig("spatial_aligned_r1.0.png")
            p["混淆矩阵路径"] = fig("confusion_r1.0.png")
            p["UMAP路径"] = fig("umap.png")
            p["marker基因验证路径"] = tbl("marker_top20.csv")
    elif sid == "scc_default" and res == "1.0":
        p.update({"完整19类重叠表路径": tbl("matching_r1.0.csv"),
                  "空间图路径": fig("spatial_aligned_r1.0.png"),
                  "混淆矩阵路径": fig("confusion_r1.0.png"),
                  "UMAP路径": fig("umap.png"),
                  "marker基因验证路径": tbl("marker_top20.csv")})
    elif sid == "scc_s8" and res == "1.0":
        p.update({"完整19类重叠表路径": tbl("matching_r1.0.csv"),
                  "空间图路径": fig("spatial_aligned_r1.0.png"),
                  "混淆矩阵路径": fig("confusion_r1.0.png"),
                  "UMAP路径": fig("umap.png")})
    elif sid == "scc_official" and res == "0.4":
        p.update({"完整19类重叠表路径": tbl("matching_r0.4.csv"),
                  "空间图路径": fig("spatial_aligned_r0.4.png"),
                  "混淆矩阵路径": fig("confusion_r0.4.png"),
                  "UMAP路径": fig("umap.png")})
    return p

for i, row in df.iterrows():
    np_ = new_paths(row["方案ID"], str(row["resolution"]))
    for col, v in np_.items():
        df.at[i, col] = v

df.to_csv("experiment_registry.csv", index=False, encoding="utf-8-sig")
print("registry paths updated")
