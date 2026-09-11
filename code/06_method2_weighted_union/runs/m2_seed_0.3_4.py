import sys
sys.path.insert(0, 'F:/tmp')
import scipy.sparse as sp, numpy as np, hashlib
import scanpy as sc
import m2_scan_lib as lib
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

expr = sp.load_npz('F:/tmp/m2_expr_bin.npz')
spat = sp.load_npz('F:/tmp/m2_spatial_bin_s8.npz')
adj = (expr * 1.0 + spat * 0.3).tocsr()

import igraph, leidenalg

def leiden_w(adj, res, seed):
    rows, cols = adj.nonzero(); data = np.asarray(adj.data).astype(float)
    G = igraph.Graph(n=adj.shape[0]); G.add_edges(list(zip(rows.tolist(), cols.tolist()))); G.es["weight"] = data.tolist()
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights=G.es["weight"],
                                    resolution_parameter=res, seed=seed, n_iterations=-1)
    return np.array(part.membership, dtype=int)

lab = leiden_w(adj, 1.0, seed=4)

a = sc.read_h5ad('F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = a.obs['annotation'].astype(str)

n_clusters = len(np.unique(lab))
ARI = adjusted_rand_score(ann, lab)
NMI = normalized_mutual_info_score(ann, lab)

sub = np.load('F:/tmp/m2_subsample.npy')
Silhouette = silhouette_score(a.obsm['X_pca'][sub], lab[sub])

coherence = lib.spatial_coherence(lab, a.obsm['spatial'])

ov = lib.hungarian_overlaps(a.obs['annotation'], lab, a.obs['annotation'].cat.categories)
keep = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
ov8 = {k: ov[k] for k in keep}

label_sha256 = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]

out = {"Ws": 0.3, "seed": 4, "n_clusters": int(n_clusters),
       "ARI": round(float(ARI), 4), "NMI": round(float(NMI), 4),
       "Silhouette": round(float(Silhouette), 4), "coherence": round(float(coherence), 4),
       "label_sha256": label_sha256, "overlaps": ov8}
print("JSON_ONLY:" + " " + __import__("json").dumps(out))
