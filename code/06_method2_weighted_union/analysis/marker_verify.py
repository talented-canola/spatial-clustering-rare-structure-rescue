"""Task C: marker-gene verification for scc_default + baseline_v2 (res=1.0)."""
import os
import numpy as np
import pandas as pd
import scanpy as sc

DATA = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
RES = r"F:/BGI/task3/spateo-release-main/results"
OUT = os.path.join(RES, "marker_verification")
os.makedirs(OUT, exist_ok=True)

sc.settings.verbosity = 1

# load + QC (same as pipeline) to get log1p-normalized X
adata = sc.read_h5ad(DATA)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=50)
print("QC'd:", adata.shape)

# attach labels
SCC = sc.read_h5ad(os.path.join(RES, "scc", "scc_slim.h5ad"))
B2 = sc.read_h5ad(os.path.join(RES, "baseline", "baseline_corrected_slim.h5ad"))
assert np.array_equal(adata.obs.index, SCC.obs.index)
assert np.array_equal(adata.obs.index, B2.obs.index)
adata.obs["scc_default"] = SCC.obs["scc_s6_r1.0"].astype(str).values
adata.obs["baseline_v2"] = B2.obs["baseline_corrected_r1.0"].astype(str).values
ann = adata.obs["annotation"].astype(str).values

# known markers (mouse)
MARKERS = {
    "Cartilage": ["Sox9", "Col2a1", "Acan", "Col1a1", "Comp", "Col11a2", "Hapln1"],
    "Brain/neural": ["Sox2", "Pax6", "Tbr1", "Neurod1", "Neurod2", "Tubb3", "Stmn2"],
    "Spinal cord": ["Olig2", "Hoxb4", "Hoxc4", "Nkx6-1", "Pax3"],
    "DRG": ["Prph", "Isl1", "Nefh", "Ngfr", "Runx3", "Ntrk1"],
}
genes_avail = set(adata.var_names)

def top_markers(groupby, scheme):
    adata.obs["__grp"] = adata.obs[groupby].astype(str)
    sc.tl.rank_genes_groups(adata, groupby="__grp", method="t-test_overestim_var", n_genes=20)
    res = adata.uns["rank_genes_groups"]
    groups = res["names"].dtype.names
    rows = []
    for g in groups:
        genes = [res["names"][g][i] for i in range(20)]
        rows.append({"scheme": scheme, "cluster": g, "n_cells": int((adata.obs["__grp"] == g).sum()),
                     "top20_markers": ",".join(genes)})
    return pd.DataFrame(rows)

def dominant_cluster(tissue_mask, labels):
    lab = labels
    sub = lab[tissue_mask]
    if len(sub) == 0:
        return None
    return pd.Series(sub).value_counts().idxmax(), pd.Series(sub).value_counts().iloc[0], len(sub)

def check_markers(cluster_genes, tissue):
    wanted = [g for g in MARKERS[tissue] if g in genes_avail]
    present = [g for g in wanted if g in cluster_genes]
    return wanted, present

for groupby, scheme in [("scc_default", "scc_default"), ("baseline_v2", "baseline_v2")]:
    print(f"\n========== {scheme} ==========", flush=True)
    tm = top_markers(groupby, scheme)
    tm.to_csv(os.path.join(OUT, f"top_markers_{scheme}.csv"), index=False)
    lab = adata.obs[groupby].astype(str).values

    # problem tissue checks
    checks = []
    for tissue in ["Cartilage", "Spinal cord", "DRG"]:
        # find the dominant cluster for cells of this annotation
        # (cartilage/spinal cord/DRG map to annotation names)
        ann_name = {"Cartilage": "Cartilage primordium", "Spinal cord": "Spinal cord", "DRG": "Dorsal root ganglion"}[tissue]
        mask = ann == ann_name
        dc = dominant_cluster(mask, lab)
        if dc is None:
            checks.append({"scheme": scheme, "tissue": tissue, "dominant_cluster": None, "n_cells": 0,
                           "cluster_top_markers": "", "marker_check": ""})
            continue
        cl, cnt, total = dc
        cgenes = tm[tm.cluster == cl]["top20_markers"].iloc[0].split(",")
        wanted, present = check_markers(cgenes, tissue)
        checks.append({"scheme": scheme, "tissue": tissue, "dominant_cluster": cl,
                       "n_cells_in_cluster": total, "cluster_top_markers": ",".join(cgenes),
                       f"{tissue}_markers_present": ",".join(present) if present else "NONE",
                       f"{tissue}_markers_looked": ",".join(wanted)})
        print(f"  [{tissue}] {total} bins -> dominant cluster {cl} (cnt {cnt}); "
              f"markers present: {present if present else 'NONE'}", flush=True)

    cdf = pd.DataFrame(checks)
    cdf.to_csv(os.path.join(OUT, f"problem_tissue_check_{scheme}.csv"), index=False)

print("\nDONE")
