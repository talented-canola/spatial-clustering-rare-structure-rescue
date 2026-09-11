import sys
sys.path.insert(0, 'F:/tmp')
import scipy.sparse as sp
import numpy as np
import hashlib
import scanpy as sc
import m2_scan_lib as lib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# 2. Load expr + spatial binary graphs, weighted union adj = expr*1.0 + spat*0.2
expr = sp.load_npz('F:/tmp/m2_expr_bin.npz')
spat = sp.load_npz('F:/tmp/m2_spatial_bin_s8.npz')
adj = (expr * 1.0 + spat * 0.2).tocsr()

# 3. Weighted Leiden with own seed parameter (fast_leiden.py hardcodes 888, so do NOT use it)
import igraph
import leidenalg

def leiden_w(adj, res, seed):
    rows, cols = adj.nonzero()
    data = np.asarray(adj.data).astype(float)
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    G.es["weight"] = data.tolist()
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition,
                                    weights=G.es["weight"],
                                    resolution_parameter=res, seed=seed, n_iterations=-1)
    return np.array(part.membership, dtype=int)

lab = leiden_w(adj, 1.0, seed=4)

# 4. Read baseline slim h5ad
a = sc.read_h5ad('F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = a.obs['annotation'].astype(str)

# 5. Metrics
n_clusters = int(len(np.unique(lab)))
ARI = float(adjusted_rand_score(ann.values, lab))
NMI = float(normalized_mutual_info_score(ann.values, lab))
sub = np.load('F:/tmp/m2_subsample.npy')
Silhouette = float(silhouette_score(a.obsm['X_pca'][sub], lab[sub]))
coherence = lib.spatial_coherence(lab, a.obsm['spatial'])

# 6. Hungarian overlaps; keep 8 classes
cats = a.obs['annotation'].cat.categories
ov_full = lib.hungarian_overlaps(a.obs['annotation'], lab, cats)
keep = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
ov = {k: ov_full[k] for k in keep}

# 7. sha256
label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

# 8. Print ONE JSON on the last line
import json
out = {
    "Ws": 0.2, "seed": 4, "n_clusters": n_clusters,
    "ARI": round(ARI, 4), "NMI": round(NMI, 4),
    "Silhouette": round(Silhouette, 4), "coherence": round(coherence, 4),
    "label_sha256": label_sha256, "overlaps": ov,
}
print(json.dumps(out))
