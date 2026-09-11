"""method2 weighted-union Leiden at W_s=0.2, random_state=888, res=1.0.

Weighted union: adj = We*expr_binary + Ws*spatial_binary, We=1.0, Ws=0.2.
Leiden: own seed-parameterized wrapper (fast_leiden.py hardcodes seed=888, so a
custom function is used so the seed is explicit; seed here IS 888).
"""
import sys
import json
import hashlib

sys.path.insert(0, "F:/tmp")
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import m2_scan_lib as lib
import igraph
import leidenalg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# --- step 2: load binary graphs, build weighted union ---
expr = sp.load_npz("F:/tmp/m2_expr_bin.npz")
spat = sp.load_npz("F:/tmp/m2_spatial_bin_s8.npz")
adj = (expr * 1.0 + spat * 0.2).tocsr()

# --- step 3: weighted Leiden WITH own seed parameter ---
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

lab = leiden_w(adj, 1.0, seed=888)

# --- step 4: gold-verified slim h5ad, 19-class annotation ---
SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
a = sc.read_h5ad(SLIM)
ann = a.obs["annotation"].astype(str)

# --- step 5: metrics ---
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

# --- step 6: Hungarian-aligned overlaps, keep 8 classes ---
ov_full = lib.hungarian_overlaps(a.obs["annotation"], lab, a.obs["annotation"].cat.categories)
KEEP = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
ov = {k: ov_full[k] for k in KEEP if k in ov_full}

# --- step 7: label checksum ---
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

# --- step 8: print ONE JSON on the last line ---
res = {
    "Ws": 0.2,
    "seed": 888,
    "n_clusters": n_clusters,
    "ARI": round(ARI, 4),
    "NMI": round(NMI, 4),
    "Silhouette": round(Sil, 4),
    "coherence": round(coherence, 4),
    "label_sha256": label_sha256,
    "overlaps": ov,
}
print(json.dumps(res, ensure_ascii=False))
