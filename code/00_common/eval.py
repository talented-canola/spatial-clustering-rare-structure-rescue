"""Evaluate baseline clustering: ARI/NMI vs gold-standard annotation,
spatial coherence (fraction of same-label spatial neighbors), and silhouette."""
import os
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

OUT = r"F:/BGI/task3/baseline_results"
adata = sc.read_h5ad(os.path.join(OUT, "baseline_slim.h5ad"))

ann = adata.obs["annotation"].astype(str).values

print("=" * 60)
for res in ["baseline_leiden_r0.5", "baseline_leiden_r1.0", "baseline_leiden_r1.5", "baseline_leiden_r2.0"]:
    if res not in adata.obs.columns:
        continue
    cl = adata.obs[res].values
    ari = adjusted_rand_score(ann, cl)
    nmi = normalized_mutual_info_score(ann, cl)
    n = adata.obs[res].nunique()
    print(f"{res}: n_clusters={n:4d}  ARI={ari:.4f}  NMI={nmi:.4f}")
print("=" * 60)

# Spatial coherence: fraction of same-label spatial neighbors (k=8 grid neighbors)
xy = adata.obsm["spatial"]
nn = NearestNeighbors(n_neighbors=9, metric="euclidean").fit(xy)  # include self
_, idx = nn.kneighbors(xy)
idx = idx[:, 1:]  # drop self

def spatial_coherence(labels):
    lab = np.asarray(labels)
    same = lab[idx] == lab[:, None]
    return float(same.mean())

print("\nSpatial coherence (fraction same-label spatial neighbors, k=8):")
for res in ["baseline_leiden_r0.5", "baseline_leiden_r1.0", "baseline_leiden_r1.5", "baseline_leiden_r2.0"]:
    if res not in adata.obs.columns:
        continue
    print(f"  {res}: {spatial_coherence(adata.obs[res].values):.4f}")
print(f"  gold-standard annotation: {spatial_coherence(ann):.4f}")

# Silhouette on X_pca (subsample to keep memory bounded)
rng = np.random.RandomState(0)
sub = rng.choice(adata.n_obs, size=min(6000, adata.n_obs), replace=False)
X = adata.obsm["X_pca"][sub]
print("\nSilhouette on X_pca (subsample 6000):")
for res in ["baseline_leiden_r0.5", "baseline_leiden_r1.0", "baseline_leiden_r1.5", "baseline_leiden_r2.0"]:
    if res not in adata.obs.columns:
        continue
    s = silhouette_score(X, adata.obs[res].values[sub].astype(int))
    print(f"  {res}: {s:.4f}")
