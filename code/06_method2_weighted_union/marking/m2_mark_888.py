import sys; sys.path.insert(0,'F:/tmp')
import numpy as np, scipy.sparse as sp, scanpy as sc, hashlib, igraph, leidenalg, json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter
import m2_scan_lib as lib

def leiden_unw(adj, res, seed):
    rows, cols = adj.nonzero()
    G = igraph.Graph(n=adj.shape[0]); G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed, n_iterations=-1)
    return np.array(part.membership, dtype=int)

seed = 888
expr = sp.load_npz(r'F:/tmp/m2_expr_bin.npz')
a = sc.read_h5ad(r'F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = a.obs['annotation'].astype(str).values
lab = leiden_unw(expr, 1.0, seed=888)
n = lab.shape[0]
scm = (ann == 'Spinal cord')
uniq = np.unique(lab)
cnts = {}
for c in uniq: cnts[int(c)] = int((lab[scm] == c).sum())
C = max(cnts, key=cnts.get)
proto = sorted(np.where(lab == C)[0].tolist())
recall = round(cnts[C] / int(scm.sum()), 4)
precision = round(cnts[C] / len(proto), 4)
cnt = Counter(lab)
flag = sorted(np.where(np.array([cnt[l] for l in lab]) < 500)[0].tolist())
flag_sizes = sorted([int(s) for s in cnt.values() if s < 500])
ari = round(adjusted_rand_score(ann, lab.astype(str)), 4)
nmi = round(normalized_mutual_info_score(ann, lab.astype(str)), 4)
sha = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]
if seed == 888:
    np.save('F:/tmp/m2_mark_r1_seed888.npy', lab)
    np.save('F:/tmp/m2_flag_bins.npy', np.array(flag, dtype=np.int64))
base_lab = a.obs['baseline_corrected_r1.0'].astype(int).values if 'baseline_corrected_r1.0' in a.obs.columns else None
base_ari = round(adjusted_rand_score(base_lab.astype(str), lab.astype(str)), 4) if base_lab is not None else None
out = {"seed": 888, "n_clusters": int(len(uniq)), "ARI": ari, "NMI": nmi,
       "sc_proto_size": int(len(proto)), "sc_recall": recall, "sc_precision": precision,
       "sc_proto_bins": proto, "flag_bins": flag, "flag_sizes": flag_sizes,
       "label_sha256": sha, "base_ari_vs_r10": base_ari}
print(json.dumps(out))
