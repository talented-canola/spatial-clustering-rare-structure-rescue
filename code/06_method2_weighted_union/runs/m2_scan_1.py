"""method2 weighted-union scan: W_s=1 (W_e=1) at resolution 1.0."""
import sys
sys.path.insert(0, "F:/tmp")
import hashlib
import json

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import m2_scan_lib as lib
import fast_leiden

# 1. load binary graphs (gold-verified)
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")

# 2. weighted union, Ws=1, We=1
adj = lib.weighted_union(expr, spat, Ws=1)

# 3. weighted leiden at resolution 1.0
lab = fast_leiden.fast_leiden_weighted(adj, 1.0)

# 4. annotation
a = sc.read_h5ad("F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad")
ann = a.obs["annotation"].astype(str)

# 5. metrics
n_clusters = int(len(np.unique(lab)))
ARI = round(adjusted_rand_score(ann, lab.astype(str)), 4)
NMI = round(normalized_mutual_info_score(ann, lab.astype(str)), 4)
sub = np.load("F:/tmp/m2_subsample.npy")
Silhouette = round(silhouette_score(a.obsm["X_pca"][sub], lab.astype(int)[sub]), 4)
coherence = round(lib.spatial_coherence(lab, a.obsm["spatial"]), 4)

# 6. hungarian overlaps, keep 8 classes
ov = lib.hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
keep = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
overlaps = {k: ov[k] for k in keep}

# 7. label hash
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

out = {
    "Ws": 1,
    "n_clusters": n_clusters,
    "ARI": ARI,
    "NMI": NMI,
    "Silhouette": Silhouette,
    "coherence": coherence,
    "label_sha256": label_sha256,
    "overlaps": overlaps,
}
print(json.dumps(out))
