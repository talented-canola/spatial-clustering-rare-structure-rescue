"""Color-align baseline clusters to gold-standard annotation via confusion matrix +
Hungarian assignment, so corresponding classes share colors for honest visual comparison."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import seaborn as sns

OUT = r"F:/BGI/task3/spateo-release-main/results/baseline"
adata = sc.read_h5ad(os.path.join(OUT, "baseline_slim.h5ad"))

ann = adata.obs["annotation"].astype(str)
ann_cats = list(adata.obs["annotation"].cat.categories)  # 19 names, fixed order
xy = adata.obsm["spatial"]

# distinct color pool (tab20 + tab20b + tab20c = 60 colors)
pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}

RES = ["baseline_leiden_r0.5", "baseline_leiden_r1.0", "baseline_leiden_r1.5", "baseline_leiden_r2.0"]

def hungarian_match(ct):
    """ct: DataFrame (annotations x clusters). Returns {annotation_name: cluster_label}."""
    C = ct.values.astype(float)
    n_ann, n_cl = C.shape
    n = max(n_ann, n_cl)
    Cpad = np.zeros((n, n))
    Cpad[:n_ann, :n_cl] = C
    row_ind, col_ind = linear_sum_assignment(-Cpad)
    ann2cl = {}
    clusters = list(ct.columns)
    for i in range(n_ann):
        j = col_ind[i]
        ann2cl[ann_cats[i]] = clusters[j]
    return ann2cl

for res in RES:
    ct = pd.crosstab(ann, adata.obs[res])
    ann2cl = hungarian_match(ct)

    # matching table with overlap fractions
    rows = []
    for a in ann_cats:
        cl = ann2cl[a]
        frac = ct.loc[a, cl] / ct.loc[a].sum()
        rows.append((a, cl, round(float(frac), 3)))
    mt = pd.DataFrame(rows, columns=["annotation", "baseline_cluster", "overlap_fraction"])
    mt.to_csv(os.path.join(OUT, f"matching_{res}.csv"), index=False)
    ct.to_csv(os.path.join(OUT, f"confusion_{res}.csv"))

    # color map for baseline clusters: matched -> annotation color, unmatched -> leftover
    matched_clusters = set(ann2cl.values())
    leftover = [c for c in list(ct.columns) if c not in matched_clusters]
    cl2color = {cl: ann_colors[a] for a, cl in ann2cl.items()}
    for i, cl in enumerate(leftover):
        cl2color[cl] = pool[19 + i]

    # ---- aligned spatial figure ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for a in ann_cats:
        m = (ann == a).values
        axes[0].scatter(xy[m, 0], xy[m, 1], s=2, color=ann_colors[a], rasterized=True)
    axes[0].set_title("Gold-standard annotation")
    axes[0].legend(ann_cats, markerscale=4, fontsize=5, loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
    axes[0].set_aspect("equal"); axes[0].invert_yaxis()

    for cl in list(ct.columns):
        m = (adata.obs[res] == cl).values
        axes[1].scatter(xy[m, 0], xy[m, 1], s=2, color=cl2color[cl], rasterized=True)
    axes[1].set_title(f"Baseline ({res.replace('baseline_leiden_','res=')}) — re-colored to match annotation")
    # legend: matched -> "cluster N ≈ annotation", unmatched -> "cluster N"
    labels = []
    for a in ann_cats:
        labels.append(f"{ann2cl[a]} ≈ {a}")
    for cl in leftover:
        labels.append(f"{cl} (unmatched)")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cl2color[cl], markersize=6)
               for cl in ([ann2cl[a] for a in ann_cats] + leftover)]
    axes[1].legend(handles, labels, markerscale=1, fontsize=5, loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
    axes[1].set_aspect("equal"); axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"spatial_aligned_{res}.png"), dpi=120)
    plt.close()

    # ---- confusion matrix heatmap ----
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.3, ax=ax,
                annot_kws={"size": 6})
    ax.set_xlabel("baseline cluster"); ax.set_ylabel("annotation")
    ax.set_title(f"Confusion matrix ({res})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"confusion_{res}.png"), dpi=120)
    plt.close()

# print matching tables for the two coarsest (most interpretable) resolutions
for res in ["baseline_leiden_r0.5", "baseline_leiden_r1.0"]:
    mt = pd.read_csv(os.path.join(OUT, f"matching_{res}.csv"))
    print(f"\n=== {res} === (annotation -> baseline cluster, overlap fraction)")
    for _, r in mt.iterrows():
        print(f"  {r.annotation:22s} -> cluster {r.baseline_cluster:>3s}   overlap {r.overlap_fraction}")

print("\nAll aligned figures + CSVs saved to", OUT)
