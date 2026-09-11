"""method2 weighted-union Leiden, W_s=0.2, random_state=2, res=1.0.
Prints ONE JSON on the last line."""
import sys, json, hashlib
sys.path.insert(0, "F:/tmp")

import scipy.sparse as sp
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

import m2_scan_lib as lib

WS = 0.2
SEED = 2
RES = 1.0
SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]

# 2. load binary graphs
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")

# weighted union
adj = (expr * 1.0 + spat * 0.2).tocsr()

# 3. Weighted Leiden WITH OWN seed parameter (fast_leiden.py hardcodes 888 -- do NOT use it)
import igraph
import leidenalg


def leiden_w(adj, res, seed):
    rows, cols = adj.nonzero()
    data = np.asarray(adj.data).astype(float)
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    G.es["weight"] = data.tolist()
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition,
                                    weights=G.es["weight"], resolution_parameter=res,
                                    seed=seed, n_iterations=-1)
    return np.array(part.membership, dtype=int)


lab = leiden_w(adj, RES, seed=SEED)

# 4. annotation
a = sc.read_h5ad(SLIM)
ann = a.obs["annotation"].astype(str)
cats = a.obs["annotation"].cat.categories

# 5. metrics
n_clusters = int(len(np.unique(lab)))
ARI = float(adjusted_rand_score(ann, lab.astype(str)))
NMI = float(normalized_mutual_info_score(ann, lab.astype(str)))
sub = np.load("F:/tmp/m2_subsample.npy")
try:
    Silhouette = float(silhouette_score(a.obsm["X_pca"][sub], lab.astype(int)[sub]))
except ValueError as e:
    Silhouette = -1.0
    print(f"  [warn] silhouette failed: {e}", file=sys.stderr)
coherence = lib.spatial_coherence(lab, a.obsm["spatial"])

# 6. Hungarian overlaps, keep only the 8 classes
ov = lib.hungarian_overlaps(a.obs["annotation"], lab, cats)
overlaps = {k: ov[k] for k in KEEP if k in ov}

# 7. label_sha256
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

# 8. print ONE JSON
out = {
    "Ws": WS,
    "seed": SEED,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Silhouette, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": overlaps,
}
print(json.dumps(out))
