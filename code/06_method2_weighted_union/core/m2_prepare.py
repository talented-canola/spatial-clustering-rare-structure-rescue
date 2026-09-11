"""Prepare method2 scan inputs: build expr(30)/spatial(8) binary graphs, verify union nnz
against verified scc_s8 union (915412), save npz + subsample indices."""
import sys, os
sys.path.insert(0, "F:/tmp")
import numpy as np
import scipy.sparse as sp
import m2_scan_lib as lib

SLIM = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
expr, spat, a = lib.build_graphs(SLIM)
union = (expr + spat).tocsr()
print(f"expr e30 nnz      = {expr.nnz}  (per-cell ~{expr.nnz/a.n_obs:.1f})")
print(f"spatial s8 nnz    = {spat.nnz}  (per-cell ~{spat.nnz/a.n_obs:.1f})")
print(f"union nnz         = {union.nnz}")
print(f"ref scc_s8 union  = {lib.UNION_NNZ_REF}")
print(f"union matches SCC = {abs(union.nnz - lib.UNION_NNZ_REF) / lib.UNION_NNZ_REF < 0.001}")

sp.save_npz("F:/tmp/m2_expr_bin.npz", expr.astype(float))
sp.save_npz("F:/tmp/m2_spatial_bin_s8.npz", spat.astype(float))
sub = lib.subsample_idx(a.n_obs)
np.save("F:/tmp/m2_subsample.npy", sub)
print(f"saved npz graphs + subsample({len(sub)})")

# also quick check: W_s=1.0 weighted-union Leiden reproduces near-SCC (sanity before scan)
import fast_leiden as fl
from sklearn.metrics import adjusted_rand_score
adj = lib.weighted_union(expr, spat, 1.0)
lab = fl.fast_leiden_weighted(adj, 1.0)
ann = a.obs["annotation"].astype(str).values
ari = adjusted_rand_score(ann, lab.astype(str))
print(f"W_s=1.0 res=1.0: n_clusters={len(np.unique(lab))} ARI={ari:.4f}  (scc_s8 ref ARI=0.4437)")
