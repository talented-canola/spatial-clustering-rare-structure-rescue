"""Task B: spatial(aligned) + confusion + 19-class overlap + UMAP for 4 schemes @ res=1.0."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import umap

RES = r"F:/BGI/task3/spateo-release-main/results"
B2 = sc.read_h5ad(os.path.join(RES, "baseline", "baseline_corrected_slim.h5ad"))
SCC = sc.read_h5ad(os.path.join(RES, "scc", "scc_slim.h5ad"))
SCC4812 = sc.read_h5ad(os.path.join(RES, "scc", "scc_s4812_slim.h5ad"))

ann = B2.obs["annotation"].astype(str).values
ann_cats = list(B2.obs["annotation"].cat.categories)
xy = B2.obsm["spatial"]
X_pca = B2.obsm["X_pca"]  # arpack-30, shared

pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}

# UMAP once
red = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(X_pca)

def hungarian(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    n = max(n_ann, n_cl)
    Cpad = np.zeros((n, n)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    ann2cl = {}
    ov = {}
    for i in range(n_ann):
        a = ann_cats[i]; j = ci[i]
        if j < n_cl:
            ann2cl[a] = clusters[j]
            ov[a] = round(float(ct.loc[a, clusters[j]] / ct.loc[a].sum()), 3)
        else:
            ann2cl[a] = None
            ov[a] = 0.0
    return ct, ann2cl, ov

def cl2color(ann2cl, clusters):
    matched = {c for c in ann2cl.values() if c is not None}
    leftover = [c for c in clusters if c not in matched]
    m = {cl: ann_colors[a] for a, cl in ann2cl.items() if cl is not None}
    for i, cl in enumerate(leftover):
        m[cl] = pool[19 + i]
    return m, leftover

def process(scheme_id, outdir, labels):
    os.makedirs(outdir, exist_ok=True)
    lab = labels.astype(str).values
    ct, ann2cl, ov = hungarian(labels)
    clusters = list(ct.columns)
    c2c, leftover = cl2color(ann2cl, clusters)

    # 19-class overlap CSV
    mt = pd.DataFrame([{"annotation": a, "baseline_cluster": ann2cl[a], "overlap_fraction": ov[a]} for a in ann_cats])
    mt.to_csv(os.path.join(outdir, f"matching_{scheme_id}.csv"), index=False)
    ct.to_csv(os.path.join(outdir, f"confusion_{scheme_id}.csv"))

    # spatial aligned
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for a in ann_cats:
        m = (ann == a)
        axes[0].scatter(xy[m, 0], xy[m, 1], s=2, color=ann_colors[a], rasterized=True)
    axes[0].set_title("Annotation (19 classes)"); axes[0].set_aspect("equal"); axes[0].invert_yaxis()
    for cl in clusters:
        m = (lab == cl)
        axes[1].scatter(xy[m, 0], xy[m, 1], s=2, color=c2c[cl], rasterized=True)
    axes[1].set_title(f"{scheme_id} (re-colored to match annotation)"); axes[1].set_aspect("equal"); axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"spatial_aligned_{scheme_id}.png"), dpi=120)
    plt.close()

    # confusion heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.3, ax=ax, annot_kws={"size": 6})
    ax.set_title(f"Confusion matrix: annotation x {scheme_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"confusion_{scheme_id}.png"), dpi=120)
    plt.close()

    # UMAP: annotation (left) + cluster (right), no color alignment
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for a in ann_cats:
        m = (ann == a)
        axes[0].scatter(red[m, 0], red[m, 1], s=2, rasterized=True)
    axes[0].set_title("UMAP by annotation")
    axes[1].scatter(red[:, 0], red[:, 1], s=2, c=lab.astype(int), cmap=plt.cm.turbo, rasterized=True)
    axes[1].set_title(f"UMAP by {scheme_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"umap_{scheme_id}.png"), dpi=120)
    plt.close()

    return mt

print("baseline_v2 ...", flush=True)
process("baseline_v2", os.path.join(RES, "baseline"), B2.obs["baseline_corrected_r1.0"])
print("scc_default ...", flush=True)
process("scc_default", os.path.join(RES, "scc"), SCC.obs["scc_s6_r1.0"])
print("scc_s8 ...", flush=True)
process("scc_s8", os.path.join(RES, "scc"), SCC4812.obs["scc_s8_r1.0"])
print("scc_official ...", flush=True)
process("scc_official", os.path.join(RES, "scc"), SCC4812.obs["scc_louvain_res0.4_s8"])
print("DONE")
