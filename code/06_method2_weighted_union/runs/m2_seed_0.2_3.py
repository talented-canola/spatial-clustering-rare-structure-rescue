"""method2 weighted-union Leiden at W_s=0.2, random_state=3, res=1.0.

Graph construction uses the gold-verified binary KNN matrices (spateo-faithful):
  - expr KNN (k=30 on arpack-30 X_pca), binary, nnz=793962
  - spatial KNN (k=8 on xy), binary, nnz=191646
Weighted union: adj = 1.0*expr_binary + 0.2*spatial_binary (scipy sums dual edges).
Leiden: OWN seed parameter (fast_leiden.py hardcodes 888, so NOT usable for seed=3)
via leidenalg RBConfigurationVertexPartition, resolution_parameter=1.0, seed=3,
n_iterations=-1, edge weights = adj.data.
"""
import sys
import json
import hashlib

sys.path.insert(0, "F:/tmp")

import numpy as np
import scipy.sparse as sp
import scanpy as sc
import igraph
import leidenalg
import m2_scan_lib as lib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# --- 1. load gold-verified binary graphs ---
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")
assert expr.nnz == 793962, f"expr nnz mismatch: {expr.nnz}"
assert spat.nnz == 191646, f"spat nnz mismatch: {spat.nnz}"

# --- 2. weighted union ---
adj = (expr * 1.0 + spat * 0.2).tocsr()
assert adj.nnz == 915422, f"union nnz mismatch: {adj.nnz}"

# --- 3. weighted Leiden with OWN seed parameter (fast_leiden hardcodes 888) ---
def leiden_w(adj, res, seed):
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


lab = leiden_w(adj, 1.0, seed=3)

# --- 4. load AnnData ---
SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
a = sc.read_h5ad(SLIM)
ann = a.obs["annotation"].astype(str)

# --- 5. metrics ---
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

# --- 6. hungarian overlaps, keep the 8 named classes ---
ov_full = lib.hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
ov = {k: ov_full[k] for k in KEEP if k in ov_full}

# --- 7. label sha256 ---
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

# --- 8. single JSON on the LAST line ---
res = {
    "Ws": 0.2,
    "seed": 3,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Sil, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": ov,
}
print(json.dumps(res, ensure_ascii=False))
