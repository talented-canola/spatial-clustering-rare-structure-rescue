"""method2 weighted-union scan, W_s=0.3 (W_e=1), resolution 1.0.
Reads gold-verified binary graphs, builds weighted union, clusters via
fast_leiden_weighted (seed 888, n_iterations=-1), scores against the 19-class
annotation, and prints ONE JSON on the last line."""
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

WS = 0.3
RES = 1.0
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"

# 1. load gold-verified binary graphs
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")
assert expr.nnz == 793962 and spat.nnz == 191646, (expr.nnz, spat.nnz)

# 2. weighted union (W_e=1, W_s=0.3)
adj = lib.weighted_union(expr, spat, Ws=WS)

# 3. weighted leiden at res 1.0
lab = fast_leiden.fast_leiden_weighted(adj, RES)

# 4. reference annotation
a = sc.read_h5ad(SLIM)
ann = a.obs["annotation"].astype(str)

# 5. metrics
n_clusters = int(len(np.unique(lab)))
ARI = adjusted_rand_score(ann, lab.astype(str))
NMI = normalized_mutual_info_score(ann, lab.astype(str))
sub = np.load("F:/tmp/m2_subsample.npy")
Silhouette = silhouette_score(a.obsm["X_pca"][sub], lab.astype(int)[sub])
coherence = lib.spatial_coherence(lab, a.obsm["spatial"])

# 6. hungarian overlaps on 8 target classes
ov_full = lib.hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
ov = {c: ov_full[c] for c in KEEP}

# 7. label hash
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

result = {
    "Ws": WS,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Silhouette, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": ov,
}
print(json.dumps(result, ensure_ascii=False))
