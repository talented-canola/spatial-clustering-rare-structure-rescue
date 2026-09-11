"""method2 weighted-union Leiden at W_s=0.2 (W_e=1), random_state=1, res=1.0."""
import sys
sys.path.insert(0, "F:/tmp")
import hashlib
import json

import numpy as np
import scipy.sparse as sp
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import m2_scan_lib as lib


def leiden_w(adj, res, seed):
    import igraph
    import leidenalg

    rows, cols = adj.nonzero()
    data = np.asarray(adj.data).astype(float)
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    G.es["weight"] = data.tolist()
    part = leidenalg.find_partition(
        G,
        leidenalg.RBConfigurationVertexPartition,
        weights=G.es["weight"],
        resolution_parameter=res,
        seed=seed,
        n_iterations=-1,
    )
    return np.array(part.membership, dtype=int)


# 1. load binary graphs (gold-verified)
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")

# 2. weighted union W_s=0.2, W_e=1
adj = (expr * 1.0 + spat * 0.2).tocsr()

# 3. weighted Leiden with OWN seed=1 (fast_leiden hardcodes 888)
lab = leiden_w(adj, 1.0, seed=1)

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
    "Ws": 0.2,
    "seed": 1,
    "n_clusters": n_clusters,
    "ARI": ARI,
    "NMI": NMI,
    "Silhouette": Silhouette,
    "coherence": coherence,
    "label_sha256": label_sha256,
    "overlaps": overlaps,
}
print(json.dumps(out, ensure_ascii=False))
