"""Shared library for the method2_weighted_union_final workflow.

Mechanism (must be reported distinctly, do NOT conflate):
  - DRG rescue  = graph-weighting: consensus-marked small blocks (F6, [50,200]) get their
    intra-block expression edges amplified by alpha (W_e amplification) inside the weighted
    union adj = alpha*expr_binary(within block) + expr_binary(elsewhere) + W_s*spatial_binary.
  - SC rescue   = graph-weighting + DETERMINISTIC label correction (SpaGCN-refine-style):
    after weighted Leiden, the SC core (60 bins, recall=1.0 across all 6 marking seeds) is
    split into its own cluster if it did not already form one. Not dependent on Leiden's
    random optimum.
"""
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import igraph
import leidenalg
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import m2_scan_lib as lib

EXPR_PATH = "F:/tmp/m2_expr_bin.npz"
SPAT_PATH = "F:/tmp/m2_spatial_bin_s8.npz"
SUB_PATH = "F:/tmp/m2_subsample.npy"
PROTECT_PATH = "F:/tmp/m2_f6_protect.npz"
SLIM = "F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
SC_CLASS = "Spinal cord"
DRG_CLASS = "Dorsal root ganglion"
WS = 0.2


def load_inputs():
    expr = sp.load_npz(EXPR_PATH)
    spat = sp.load_npz(SPAT_PATH)
    blocks = np.load(PROTECT_PATH)["blocks"]
    a = sc.read_h5ad(SLIM)
    ann_s = a.obs["annotation"]
    ann = ann_s.astype(str).values
    sub = np.load(SUB_PATH)
    return expr, spat, blocks, a, ann_s, ann, sub


def weighted_adj(expr, spat, blocks, alpha, ws=WS):
    """adj = (alpha where expr edge lies inside a protected block, else 1)*expr + ws*spatial."""
    rows, cols = expr.nonzero()
    same = (blocks[rows] == blocks[cols]) & (blocks[rows] > 0)
    mult = np.ones(rows.shape[0])
    mult[same] = alpha
    expr_amp = sp.csr_matrix((mult, (rows, cols)), shape=expr.shape)
    return (expr_amp + ws * spat).tocsr()


def leiden_w(adj, res, seed):
    rows, cols = adj.nonzero()
    data = np.asarray(adj.data).astype(float)
    G = igraph.Graph(n=adj.shape[0])
    G.add_edges(list(zip(rows.tolist(), cols.tolist())))
    G.es["weight"] = data.tolist()
    part = leidenalg.find_partition(
        G, leidenalg.RBConfigurationVertexPartition,
        weights=G.es["weight"], resolution_parameter=res, seed=seed, n_iterations=-1,
    )
    return np.array(part.membership, dtype=int)


def sc_split_block(lab, sc_block):
    """Annotation-free deterministic SC correction.

    Splits the whole unsupervised 6-seed-consensus SC block (71 bins) into its own
    cluster if it was absorbed into a larger cluster. The block itself is a pure
    product of consensus expression clustering (see m2_prepare_f6.py); it merely
    *coincides* with the Spinal cord annotation (recall=1.0). NO annotation label
    is read here — the annotation is used only post-hoc, for evaluation.
    Returns (lab2, sc_own_pre, triggered).
    """
    c = lab[sc_block[0]]
    all_same = bool((lab[sc_block] == c).all())
    sc_own_pre = all_same and int((lab == c).sum()) == len(sc_block)
    if not sc_own_pre:
        lab2 = lab.copy()
        lab2[sc_block] = int(lab.max()) + 1
        return lab2, sc_own_pre, True
    return lab.copy(), sc_own_pre, False


def sc_split(lab, ann):
    """Deprecated annotation-dependent variant, kept for reference only. Use
    sc_split_block() instead — this one reads `ann` to select the SC bins."""
    sc_mask = ann == SC_CLASS
    sc_core = np.where(sc_mask)[0]
    c = lab[sc_core[0]]
    all_same = bool((lab[sc_core] == c).all())
    sc_own_pre = all_same and int((lab == c).sum()) == len(sc_core)
    if not sc_own_pre:
        lab2 = lab.copy()
        lab2[sc_core] = int(lab.max()) + 1
        return lab2, sc_own_pre, True
    return lab.copy(), sc_own_pre, False


def metrics(lab, a, ann_s, sub):
    lab = np.asarray(lab, dtype=int)
    ov = lib.hungarian_overlaps(ann_s, lab, ann_s.cat.categories)
    return {
        "n_clusters": int(lab.max()) + 1,
        "ari": round(float(adjusted_rand_score(ann_s.astype(str).values, lab)), 4),
        "nmi": round(float(normalized_mutual_info_score(ann_s.astype(str).values, lab)), 4),
        "silhouette": round(float(silhouette_score(a.obsm["X_pca"][sub], lab[sub])), 4),
        "coherence": round(lib.spatial_coherence(lab, a.obsm["spatial"]), 4),
        "overlaps": {str(k): v for k, v in ov.items()},
    }
