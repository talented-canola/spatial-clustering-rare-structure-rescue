import pandas as pd, os
RES = r"F:/BGI/task3/spateo-release-main/results"
df = pd.read_csv(os.path.join(RES, "experiment_registry.csv"))

updates = {
    ("baseline_v2_arpack", "1.0"): {
        "空间图路径": "baseline/spatial_aligned_baseline_v2.png",
        "混淆矩阵路径": "baseline/confusion_baseline_v2.png",
        "完整19类重叠表路径": "baseline/matching_baseline_v2.csv",
        "UMAP路径": "baseline/umap_baseline_v2.png",
        "marker基因验证路径": "marker_verification/top_markers_baseline_v2.csv",
        "状态": "已完成",
    },
    ("scc_default", "1.0"): {
        "空间图路径": "scc/spatial_aligned_scc_default.png",
        "混淆矩阵路径": "scc/confusion_scc_default.png",
        "完整19类重叠表路径": "scc/matching_scc_default.csv",
        "UMAP路径": "scc/umap_scc_default.png",
        "marker基因验证路径": "marker_verification/top_markers_scc_default.csv",
        "状态": "已完成",
    },
    ("scc_s8", "1.0"): {
        "空间图路径": "scc/spatial_aligned_scc_s8.png",
        "混淆矩阵路径": "scc/confusion_scc_s8.png",
        "完整19类重叠表路径": "scc/matching_scc_s8.csv",
        "UMAP路径": "scc/umap_scc_s8.png",
        "状态": "部分完成",
    },
    ("scc_official", "0.4"): {
        "空间图路径": "scc/spatial_aligned_scc_official.png",
        "混淆矩阵路径": "scc/confusion_scc_official.png",
        "完整19类重叠表路径": "scc/matching_scc_official.csv",
        "UMAP路径": "scc/umap_scc_official.png",
        "状态": "部分完成",
    },
}
for (sid, res), vals in updates.items():
    mask = (df["方案ID"] == sid) & (df["resolution"] == res)
    for col, v in vals.items():
        df.loc[mask, col] = v

df.to_csv(os.path.join(RES, "experiment_registry.csv"), index=False, encoding="utf-8-sig")
print("registry updated,", len(df), "rows")
