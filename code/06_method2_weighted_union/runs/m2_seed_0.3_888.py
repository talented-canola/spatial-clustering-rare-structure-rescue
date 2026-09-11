"""method2 weighted-union Leiden at W_s=0.3, random_state=888, res=1.0.

Reads gold-verified binary KNN graphs, builds the weighted union
adj = expr_binary + 0.3*spatial_binary, clusters with OWN seed-parameterized
weighted Leiden (leidenalg RBConfigurationVertexPartition, n_iterations=-1),
scores against the 19-class annotation, prints ONE JSON on the last line.
"""
import sys
sys.path.insert(0, "F:/tmp")
import hashlib
import json
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import igraph
import leidenalg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import m2_scan_lib as lib

WS = 0.3
RES = 1.0
SEED = 888
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"


def leiden_w(adj, res, seed):
    """Weighted Leiden with an explicit seed parameter (fast_leiden.py hardcodes 888)."""
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


# 1. load gold-verified binary graphs
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")
assert expr.nnz == 793962 and spat.nnz == 191646, (expr.nnz, spat.nnz)

# 2. weighted union (W_e=1, W_s=0.3)
adj = (expr * 1.0 + spat * 0.3).tocsr()

# 3. weighted leiden at res 1.0 with own seed
lab = leiden_w(adj, RES, SEED)

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
    "seed": SEED,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Silhouette, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": ov,
}
print(json.dumps(result, ensure_ascii=False))
