# -*- coding: utf-8 -*-
"""Recompute Hungarian-aligned annotation-overlap fractions for 9 clustering schemes."""
import json
import os

import scanpy as sc
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

RES = "F:/BGI/task3/spateo-release-main/results"

MANIFEST = [
    ("baseline_v2_arpack",   os.path.join(RES, "baseline_v2_arpack/data/baseline_corrected_slim.h5ad"), "baseline_corrected_r{res}"),
    ("scc_s4",               os.path.join(RES, "scc_s4/data/scc_s4812_slim.h5ad"), "scc_s4_r{res}"),
    ("scc_s8",               os.path.join(RES, "scc_s4/data/scc_s4812_slim.h5ad"), "scc_s8_r{res}"),
    ("scc_s12",              os.path.join(RES, "scc_s4/data/scc_s4812_slim.h5ad"), "scc_s12_r{res}"),
    ("smooth_s8",            os.path.join(RES, "smooth_s8/data/smooth_s8_slim.h5ad"), "smooth_s8_r{res}"),
    ("smooth_s8_incl_self",  os.path.join(RES, "smooth_s8_incl_self/data/smooth_s8_incl_self_slim.h5ad"), "smooth_s8_incl_self_r{res}"),
    ("graphst_reference",    os.path.join(RES, "graphst_reference/data/graphst_reference_slim.h5ad"), "graphst_reference_r{res}"),
    ("method1_bilateral_spatial_knn", os.path.join(RES, "method1_bilateral_spatial_knn/data/method1_bilateral_spatial_knn_slim.h5ad"), "method1_bilateral_spatial_knn_r{res}"),
    ("method1_bilateral_expr_knn",   os.path.join(RES, "method1_bilateral_expr_knn/data/method1_bilateral_expr_knn_slim.h5ad"), "method1_bilateral_expr_knn_r{res}"),
]

RESOLUTIONS = ["0.5", "1.0", "1.5", "2.0"]

TISSUES = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
           "Dorsal root ganglion", "GI tract", "Surface ectoderm", "Liver", "Heart"]

results = []

for scheme, path, label_tpl in MANIFEST:
    if not os.path.exists(path):
        print(f"WARNING: missing path for scheme={scheme}: {path}")
        continue

    adata = sc.read_h5ad(path)

    # Preserve original 19-class category order.
    orig_cats = list(adata.obs["annotation"].cat.categories)
    ann = pd.Categorical(adata.obs["annotation"].astype(str), categories=orig_cats)
    ann_cats = list(ann.categories)

    for res in RESOLUTIONS:
        lab_col = label_tpl.format(res=res)
        if lab_col not in adata.obs.columns:
            print(f"WARNING: missing label column {lab_col} in {path}")
            continue

        lab = adata.obs[lab_col]

        ct = pd.crosstab(ann, lab)
        # Reindex rows to the original 19 categories (crosstab respects category order,
        # but reindex defensively).
        ct = ct.reindex(ann_cats)

        nrows, ncols = ct.shape
        C = ct.values.astype(float)
        N = max(nrows, ncols)
        if N > nrows:
            C = np.pad(C, ((0, N - nrows), (0, 0)))
        if N > ncols:
            C = np.pad(C, ((0, 0), (0, N - ncols)))

        ri, ci = linear_sum_assignment(-C)

        colnames = list(ct.columns)
        overlaps = {}
        for i, a_cls in enumerate(ann_cats):
            row_total = ct.loc[a_cls].sum()
            if ci[i] < ncols:
                val = ct.loc[a_cls, colnames[ci[i]]] / row_total
            else:
                val = 0.0
            overlaps[a_cls] = round(float(val), 4)

        # Record: n_clusters per res
        n_clusters = int(lab.nunique(dropna=True))

        # Record: Surface ectoderm row
        se_overlap = overlaps["Surface ectoderm"]
        se_cluster = None
        if ci[ann_cats.index("Surface ectoderm")] < ncols:
            se_cluster = colnames[ci[ann_cats.index("Surface ectoderm")]]
        print(f"scheme={scheme} res={res} n_clusters={n_clusters} "
              f"Surface_ectoderm_overlap={se_overlap} assigned_cluster={se_cluster}")

        filtered = {k: overlaps[k] for k in TISSUES}
        results.append({
            "scheme": scheme,
            "resolution": res,
            "source": "recompute",
            "overlaps": filtered,
        })

# Print ONLY one JSON object on the very last line.
print(json.dumps({"results": results}, ensure_ascii=False))
