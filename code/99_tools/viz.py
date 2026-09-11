"""Visualize baseline clustering: spatial layout + UMAP, colored by gold-standard
annotation vs baseline clusters."""
import os
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"F:/BGI/task3/baseline_results"
adata = sc.read_h5ad(os.path.join(OUT, "baseline_slim.h5ad"))

# UMAP on PCA embedding (umap-learn directly, avoids scanpy uns['neighbors'] plumbing)
import umap
adata.obsm["X_umap"] = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.3).fit_transform(adata.obsm["X_pca"])

xy = adata.obsm["spatial"]
res = "baseline_leiden_r1.0"  # representative resolution

# color palettes
ann_colors = {c: f"C{i}" for i, c in enumerate(adata.obs["annotation"].cat.categories)}
n_cl = adata.obs[res].nunique()
cl_colors = {c: f"C{i}" for i, c in enumerate(sorted(adata.obs[res].unique()))}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# (a) gold-standard annotation
order = list(adata.obs["annotation"].cat.categories)
for cat in order:
    m = adata.obs["annotation"] == cat
    axes[0].scatter(xy[m, 0], xy[m, 1], s=2, c=ann_colors[cat], label=cat, rasterized=True)
axes[0].set_title("Gold-standard annotation (19 classes)")
axes[0].legend(markerscale=3, fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
axes[0].set_aspect("equal"); axes[0].invert_yaxis()
# (b) baseline clusters
for cl in sorted(adata.obs[res].unique()):
    m = adata.obs[res] == cl
    axes[1].scatter(xy[m, 0], xy[m, 1], s=2, c=cl_colors[cl], label=f"cl {cl}", rasterized=True)
axes[1].set_title(f"Baseline Leiden (resolution 1.0, {n_cl} clusters)")
axes[1].legend(markerscale=3, fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
axes[1].set_aspect("equal"); axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "spatial_annotation_vs_baseline.png"), dpi=120)
plt.close()

# UMAP panel
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
um = adata.obsm["X_umap"]
for cat in order:
    m = adata.obs["annotation"] == cat
    axes[0].scatter(um[m, 0], um[m, 1], s=2, c=ann_colors[cat], rasterized=True)
axes[0].set_title("UMAP by annotation")
for cl in sorted(adata.obs[res].unique()):
    m = adata.obs[res] == cl
    axes[1].scatter(um[m, 0], um[m, 1], s=2, c=cl_colors[cl], rasterized=True)
axes[1].set_title("UMAP by baseline cluster")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "umap_annotation_vs_baseline.png"), dpi=120)
plt.close()

print("figures saved to", OUT)
print("baseline clusters (r1.0):", n_cl)
