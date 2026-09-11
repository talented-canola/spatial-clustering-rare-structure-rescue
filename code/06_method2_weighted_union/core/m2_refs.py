"""VERIFIER script: compute baseline_v2_arpack and scc_s8 reference rows from STORED labels.

Metrics (exact, no assumptions, from stored slim h5ad labels):
  - baseline labels = obs['baseline_corrected_r1.0']
  - scc_s8   labels = obs['scc_s8_r1.0']  (both from actual spateo runs)
  - ARI / NMI vs 19-class 'annotation'
  - Silhouette on the 4000-node subsample F:/tmp/m2_subsample.npy of X_pca
    (baseline slim has X_pca; scc slim does NOT, so X_pca is taken from the baseline
     slim -- verified annotation & spatial identical between the two files)
  - coherence via m2_scan_lib.spatial_coherence (k=9 spatial-neighbor purity)
  - 19-class Hungarian-aligned overlap via m2_scan_lib.hungarian_overlaps, then keep the 8
"""
import sys
import json
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, "F:/tmp")
import m2_scan_lib as L

BASELINE = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
SCC = r"F:/BGI/task3/spateo-release-main/results/scc_s4/data/scc_s4812_slim.h5ad"
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]

a_b = sc.read_h5ad(BASELINE)
a_s = sc.read_h5ad(SCC)

# shared embedding / spatial (verified identical annotation & spatial order between files)
X_pca = a_b.obsm["X_pca"]
xy = a_b.obsm["spatial"]
sub = np.load("F:/tmp/m2_subsample.npy")
cats = a_b.obs["annotation"].cat.categories

# sanity: scc labels length matches
assert a_s.n_obs == a_b.n_obs == X_pca.shape[0]


def row(name, labels):
    ann = a_b.obs["annotation"]
    lab = np.asarray(labels).astype(str)
    ARI = adjusted_rand_score(ann, lab)
    NMI = normalized_mutual_info_score(ann, lab)
    Sil = silhouette_score(X_pca[sub], np.asarray(labels).astype(int)[sub])
    coh = L.spatial_coherence(np.asarray(labels).astype(int), xy)
    ov_all = L.hungarian_overlaps(ann, labels, cats)
    return {
        "name": name,
        "n_clusters": len(np.unique(lab)),
        "ARI": round(float(ARI), 4),
        "NMI": round(float(NMI), 4),
        "Silhouette": round(float(Sil), 4),
        "coherence": round(float(coh), 4),
        "overlaps": {k: ov_all[k] for k in KEEP},
    }


rows = [
    row("baseline_v2_arpack", a_b.obs["baseline_corrected_r1.0"].values),
    row("scc_s8", a_s.obs["scc_s8_r1.0"].values),
]
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
