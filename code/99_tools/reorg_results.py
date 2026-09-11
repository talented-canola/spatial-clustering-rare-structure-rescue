"""Reorganize results/ into one-folder-per-scheme. Move/rename only, no recompute."""
import os, shutil, csv
import pandas as pd

RES = r"F:/BGI/task3/spateo-release-main/results"
os.chdir(RES)

# (src, dst) — src relative to results/ (or absolute via chdir), dst relative to results/
# "MERGED" markers = identical duplicate dropped.
MOVES = [
    # baseline_v1
    ("baseline/baseline_slim.h5ad", "baseline_v1/data/baseline_slim.h5ad"),
    ("baseline/confusion_baseline_leiden_r0.5.csv", "baseline_v1/tables/confusion_r0.5.csv"),
    ("baseline/confusion_baseline_leiden_r1.0.csv", "baseline_v1/tables/confusion_r1.0.csv"),
    ("baseline/confusion_baseline_leiden_r1.5.csv", "baseline_v1/tables/confusion_r1.5.csv"),
    ("baseline/confusion_baseline_leiden_r2.0.csv", "baseline_v1/tables/confusion_r2.0.csv"),
    ("baseline/confusion_baseline_leiden_r0.5.png", "baseline_v1/figures/confusion_r0.5.png"),
    ("baseline/confusion_baseline_leiden_r1.0.png", "baseline_v1/figures/confusion_r1.0.png"),
    ("baseline/confusion_baseline_leiden_r1.5.png", "baseline_v1/figures/confusion_r1.5.png"),
    ("baseline/confusion_baseline_leiden_r2.0.png", "baseline_v1/figures/confusion_r2.0.png"),
    ("baseline/matching_baseline_leiden_r0.5.csv", "baseline_v1/tables/matching_r0.5.csv"),
    ("baseline/matching_baseline_leiden_r1.0.csv", "baseline_v1/tables/matching_r1.0.csv"),
    ("baseline/matching_baseline_leiden_r1.5.csv", "baseline_v1/tables/matching_r1.5.csv"),
    ("baseline/matching_baseline_leiden_r2.0.csv", "baseline_v1/tables/matching_r2.0.csv"),
    ("baseline/spatial_aligned_baseline_leiden_r0.5.png", "baseline_v1/figures/spatial_aligned_r0.5.png"),
    ("baseline/spatial_aligned_baseline_leiden_r1.0.png", "baseline_v1/figures/spatial_aligned_r1.0.png"),
    ("baseline/spatial_aligned_baseline_leiden_r1.5.png", "baseline_v1/figures/spatial_aligned_r1.5.png"),
    ("baseline/spatial_aligned_baseline_leiden_r2.0.png", "baseline_v1/figures/spatial_aligned_r2.0.png"),
    ("baseline/spatial_annotation_vs_baseline.png", "baseline_v1/figures/spatial_raw.png"),
    ("baseline/spatial_annotation_vs_baseline_view.png", "baseline_v1/figures/spatial_raw_view.png"),
    ("baseline/umap_annotation_vs_baseline.png", "baseline_v1/figures/umap.png"),
    ("baseline/umap_annotation_vs_baseline_view.png", "baseline_v1/figures/umap_view.png"),
    # baseline_v2_arpack
    ("baseline/baseline_corrected_slim.h5ad", "baseline_v2_arpack/data/baseline_corrected_slim.h5ad"),
    ("baseline/confusion_corrected_r0.5.csv", "baseline_v2_arpack/tables/confusion_r0.5.csv"),
    ("baseline/confusion_corrected_r1.0.csv", "baseline_v2_arpack/tables/confusion_r1.0.csv"),
    ("baseline/confusion_corrected_r1.5.csv", "baseline_v2_arpack/tables/confusion_r1.5.csv"),
    ("baseline/confusion_corrected_r2.0.csv", "baseline_v2_arpack/tables/confusion_r2.0.csv"),
    ("baseline/confusion_baseline_v2.png", "baseline_v2_arpack/figures/confusion_r1.0.png"),
    ("baseline/matching_corrected_r0.5.csv", "baseline_v2_arpack/tables/matching_r0.5.csv"),
    ("baseline/matching_corrected_r1.0.csv", "baseline_v2_arpack/tables/matching_r1.0.csv"),
    ("baseline/matching_corrected_r1.5.csv", "baseline_v2_arpack/tables/matching_r1.5.csv"),
    ("baseline/matching_corrected_r2.0.csv", "baseline_v2_arpack/tables/matching_r2.0.csv"),
    ("baseline/spatial_aligned_baseline_v2.png", "baseline_v2_arpack/figures/spatial_aligned_r1.0.png"),
    ("baseline/umap_baseline_v2.png", "baseline_v2_arpack/figures/umap.png"),
    # marker (baseline_v2)
    ("marker_verification/problem_tissue_check_baseline_v2.csv", "baseline_v2_arpack/tables/marker_problem_check.csv"),
    ("marker_verification/top_markers_baseline_v2.csv", "baseline_v2_arpack/tables/marker_top20.csv"),
    # scc_default
    ("scc/scc_slim.h5ad", "scc_default/data/scc_slim.h5ad"),
    ("scc/confusion_scc_default.csv", "scc_default/tables/confusion_r1.0.csv"),
    ("scc/confusion_scc_default.png", "scc_default/figures/confusion_r1.0.png"),
    ("scc/matching_scc_default.csv", "scc_default/tables/matching_r1.0.csv"),
    ("scc/spatial_aligned_scc_default.png", "scc_default/figures/spatial_aligned_r1.0.png"),
    ("scc/umap_scc_default.png", "scc_default/figures/umap.png"),
    ("scc/stage1_raw_spatial_s6.png", "scc_default/figures/raw_spatial_r1.0.png"),
    ("marker_verification/problem_tissue_check_scc_default.csv", "scc_default/tables/marker_problem_check.csv"),
    ("marker_verification/top_markers_scc_default.csv", "scc_default/tables/marker_top20.csv"),
    # scc_s4 (shared h5ad)
    ("scc/scc_s4812_slim.h5ad", "scc_s4/data/scc_s4812_slim.h5ad"),
    # scc_s8
    ("scc/confusion_scc_s8.csv", "scc_s8/tables/confusion_r1.0.csv"),
    ("scc/confusion_scc_s8.png", "scc_s8/figures/confusion_r1.0.png"),
    ("scc/matching_scc_s8.csv", "scc_s8/tables/matching_r1.0.csv"),
    ("scc/spatial_aligned_scc_s8.png", "scc_s8/figures/spatial_aligned_r1.0.png"),
    ("scc/umap_scc_s8.png", "scc_s8/figures/umap.png"),
    # scc_official
    ("scc/confusion_scc_official.csv", "scc_official/tables/confusion_r0.4.csv"),
    ("scc/confusion_scc_official.png", "scc_official/figures/confusion_r0.4.png"),
    ("scc/matching_scc_official.csv", "scc_official/tables/matching_r0.4.csv"),
    ("scc/spatial_aligned_scc_official.png", "scc_official/figures/spatial_aligned_r0.4.png"),
    ("scc/umap_scc_official.png", "scc_official/figures/umap.png"),
    # cross-scheme analysis
    ("baseline/corrected_vs_old_target_tissues.csv", "analysis/corrected_vs_old_target_tissues.csv"),
    ("scc/scc_s4812_summary.csv", "analysis/scc_s4812_summary.csv"),
    ("scc/stage1_intrinsic_quality.csv", "analysis/stage1_intrinsic_quality.csv"),
    ("scc/stage2_eval.csv", "analysis/stage2_eval.csv"),
    ("scc/stage3_vs_baseline.csv", "analysis/stage3_vs_baseline.csv"),
    ("scc/stage1_raw_spatial_s3.png", "analysis/stage1_raw_spatial_s3.png"),
    ("scc/stage1_raw_spatial_s10.png", "analysis/stage1_raw_spatial_s10.png"),
]

# identical duplicates to drop (verified by diff)
MERGED = [
    "baseline/confusion_baseline_v2.csv",   # == confusion_corrected_r1.0.csv
    "baseline/matching_baseline_v2.csv",    # == matching_corrected_r1.0.csv
]

# execute moves
moved_count = 0
for src, dst in MOVES:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(src):
        print(f"WARN missing src: {src}")
        continue
    shutil.move(src, dst)
    moved_count += 1
print(f"moved {moved_count} files")

for src in MERGED:
    if os.path.exists(src):
        os.remove(src)
        print(f"merged (removed identical duplicate): {src}")

# clean up now-empty old dirs
for d in ["marker_verification", "scc", "baseline"]:
    if os.path.isdir(d) and not os.listdir(d):
        os.rmdir(d)
        print(f"removed empty dir: {d}")
print("reorg done")
