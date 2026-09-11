"""method2 weighted-union scan: W_s=0.2 (W_e=1) at resolution 1.0."""
import sys, json, hashlib
sys.path.insert(0, "F:/tmp")
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import m2_scan_lib as lib
import fast_leiden
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")

adj = lib.weighted_union(expr, spat, Ws=0.2)   # We=1
lab = fast_leiden.fast_leiden_weighted(adj, 1.0)  # seed 888, n_iterations=-1

SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
a = sc.read_h5ad(SLIM)
ann = a.obs["annotation"].astype(str)

n_clusters = int(len(np.unique(lab)))
ARI = float(adjusted_rand_score(ann, lab.astype(str)))
NMI = float(normalized_mutual_info_score(ann, lab.astype(str)))

sub = np.load("F:/tmp/m2_subsample.npy")
lab_sub = lab.astype(int)[sub]
if len(np.unique(lab_sub)) >= 2:
    Sil = float(silhouette_score(a.obsm["X_pca"][sub], lab_sub))
else:
    Sil = float("nan")
coherence = lib.spatial_coherence(lab, a.obsm["spatial"])

ov_full = lib.hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
ov = {k: ov_full[k] for k in KEEP if k in ov_full}

label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

res = {
    "Ws": 0.2,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Sil, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": ov,
}
print(json.dumps(res))
