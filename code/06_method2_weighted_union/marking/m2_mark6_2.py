import sys; sys.path.insert(0,'F:/tmp')
import numpy as np, scipy.sparse as sp, scanpy as sc, hashlib, igraph, leidenalg, json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import m2_scan_lib as lib

def leiden_unw(adj, res, seed):
    rows, cols = adj.nonzero()
    G = igraph.Graph(n=adj.shape[0]); G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed, n_iterations=-1)
    return np.array(part.membership, dtype=int)

expr = sp.load_npz(r'F:/tmp/m2_expr_bin.npz')
a = sc.read_h5ad(r'F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = a.obs['annotation'].astype(str).values
lab = leiden_unw(expr, 1.0, seed=2)
np.save('F:/tmp/m2_mark_r1_seed2.npy', lab)
ari = round(adjusted_rand_score(ann, lab.astype(str)), 4)
nmi = round(normalized_mutual_info_score(ann, lab.astype(str)), 4)
sha = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]
out = {"seed": 2, "n_clusters": int(len(np.unique(lab))), "ARI": ari, "NMI": nmi, "label_sha256": sha}
print(json.dumps(out))
