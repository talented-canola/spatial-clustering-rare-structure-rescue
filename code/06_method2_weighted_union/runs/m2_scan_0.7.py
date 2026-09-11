import sys
sys.path.insert(0, "F:/tmp")
import hashlib
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import m2_scan_lib as lib
import fast_leiden
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# 2. load graphs
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")

# 3. weighted union  Ws=0.7, We=1
adj = lib.weighted_union(expr, spat, Ws=0.7)

# 4. Leiden
lab = fast_leiden.fast_leiden_weighted(adj, 1.0)

# 5. annotation
a = sc.read_h5ad(r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad")
ann = a.obs["annotation"].astype(str)

# 6. metrics
n_clusters = len(np.unique(lab))
ARI = adjusted_rand_score(ann, lab.astype(str))
NMI = normalized_mutual_info_score(ann, lab.astype(str))
sub = np.load("F:/tmp/m2_subsample.npy")
Silhouette = silhouette_score(a.obsm["X_pca"][sub], lab.astype(int)[sub])
coherence = lib.spatial_coherence(lab, a.obsm["spatial"])

# 7. hungarian overlaps, keep 8 classes
ov_all = lib.hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
keep = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
overlaps = {k: ov_all[k] for k in keep}

# 8. label sha256
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

# 9. JSON
out = {
    "Ws": 0.7,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Silhouette, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": overlaps,
}
import json
print(json.dumps(out, ensure_ascii=False))
