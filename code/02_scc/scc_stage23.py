"""Stage 2 (standalone SCC eval) + Stage 3 (SCC vs corrected baseline comparison).
Loads saved labels (no re-preprocessing)."""
import os, time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp

SCC = r"F:/BGI/task3/spateo-release-main/results/scc/scc_slim.h5ad"
BASE = r"F:/BGI/task3/spateo-release-main/results/baseline/baseline_corrected_slim.h5ad"
OUT = r"F:/BGI/task3/spateo-release-main/results/scc"
RES = [0.5, 1.0, 1.5, 2.0]
S_NEIGH = [3, 6, 10]

scc = sc.read_h5ad(SCC)
base = sc.read_h5ad(BASE)
assert np.array_equal(scc.obs.index, base.obs.index), "cell order mismatch"

ann = scc.obs["annotation"].astype(str).values
ann_cats = list(scc.obs["annotation"].cat.categories)
xy = scc.obsm["spatial"]

# X_pca consistency (both arpack-30)
print("X_pca identical (SCC vs baseline)?", np.allclose(scc.obsm["X_pca"], base.obsm["X_pca"]), flush=True)
X_pca = scc.obsm["X_pca"]

# spatial KNN (k=8) for coherence
n = xy.shape[0]
knn = NearestNeighbors(n_neighbors=9).fit(xy)
_, nidx = knn.kneighbors(xy)
nidx = nidx[:, 1:]

def coherence(labels):
    lab = np.asarray(labels, dtype=int)
    return float((lab[nidx] == lab[:, None]).mean())

def hungarian_overlap(labels):
    cl = labels.astype(str)
    ct = pd.crosstab(ann, cl)
    n_ann, n_cl = ct.shape
    n = max(n_ann, n_cl)
    Cpad = np.zeros((n, n)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    out = {}
    for i in range(n_ann):
        a = ann_cats[i]; j = ci[i]
        out[a] = round(float(ct.loc[a, clusters[j]] / ct.loc[a].sum()), 3) if j < n_cl else 0.0
    return out

rng = np.random.RandomState(0)
sub = rng.choice(n, size=min(6000, n), replace=False)
sub_lab = ann[sub].astype(str)

stage2_rows, stage3_rows = [], []
for s in S_NEIGH:
    for res in RES:
        key = f"scc_s{s}_r{res}"
        lab = scc.obs[key].astype(str).values
        base_lab = base.obs[f"baseline_corrected_r{res}"].astype(str).values
        ari = adjusted_rand_score(ann, lab)
        nmi = normalized_mutual_info_score(ann, lab)
        sil = silhouette_score(X_pca[sub], lab[sub].astype(int))
        coh = coherence(lab)
        ov_scc = hungarian_overlap(lab)
        ov_base = hungarian_overlap(base_lab)
        stage2_rows.append({"s_neigh": s, "res": res, "n_clusters": len(np.unique(lab)),
                            "ARI": round(ari, 4), "NMI": round(nmi, 4),
                            "silhouette": round(sil, 4), "spatial_coherence": round(coh, 4)})
        for a in ann_cats:
            stage3_rows.append({"s_neigh": s, "res": res, "tissue": a,
                                "scc": ov_scc[a], "baseline": ov_base[a],
                                "delta": round(ov_scc[a] - ov_base[a], 3)})

stage2 = pd.DataFrame(stage2_rows)
stage3 = pd.DataFrame(stage3_rows)
stage2.to_csv(os.path.join(OUT, "stage2_eval.csv"), index=False)
stage3.to_csv(os.path.join(OUT, "stage3_vs_baseline.csv"), index=False)

print("\n=== STAGE 2: standalone SCC eval (no baseline comparison) ===", flush=True)
print(stage2.to_string(index=False), flush=True)

# Stage 3: summary of deltas (res>=1.0), identify improved vs sacrificed
print("\n=== STAGE 3: per-tissue overlap delta (SCC - corrected baseline), res=1.0 ===", flush=True)
for s in S_NEIGH:
    d = stage3[(stage3.s_neigh == s) & (stage3.res == 1.0)].copy().sort_values("delta")
    print(f"\n--- s_neigh={s} (res=1.0) ---", flush=True)
    print(d.to_string(index=False), flush=True)

print(f"\nDONE", flush=True)
