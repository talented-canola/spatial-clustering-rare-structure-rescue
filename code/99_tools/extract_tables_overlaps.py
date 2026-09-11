# -*- coding: utf-8 -*-
"""Extract overlap_fraction from existing per-scheme matching tables (tables/matching_r{res}.csv).

Independent path: only reads stored CSVs, does NOT recompute from h5ad.
Output: one JSON object on the very last line.
"""
import glob
import json
import os
import re
import sys

RES = r"F:/BGI/task3/spateo-release-main/results"

MANIFEST = [
    "baseline_v2_arpack",
    "scc_s4",
    "scc_s8",
    "scc_s12",
    "smooth_s8",
    "smooth_s8_incl_self",
    "graphst_reference",
    "method1_bilateral_spatial_knn",
    "method1_bilateral_expr_knn",
]
RESOLUTIONS = ["0.5", "1.0", "1.5", "2.0"]
TISSUES = [
    "Brain",
    "Mesenchyme",
    "Head mesenchyme",
    "Cavity",
    "Spinal cord",
    "Dorsal root ganglion",
    "GI tract",
    "Surface ectoderm",
    "Liver",
    "Heart",
]

results = []

for scheme in MANIFEST:
    scheme_dir = os.path.join(RES, scheme)
    tables_dir = os.path.join(scheme_dir, "tables")
    if not os.path.isdir(tables_dir):
        print(f"WARNING: {scheme}: tables dir missing at {tables_dir}")
        for res in RESOLUTIONS:
            results.append({"scheme": scheme, "resolution": res, "source": "table", "overlaps": {}})
        continue

    # Map resolution -> csv path
    csv_by_res = {}
    for path in sorted(glob.glob(os.path.join(tables_dir, "matching_r*.csv"))):
        fname = os.path.basename(path)
        m = re.match(r"matching_r([0-9.]+)\.csv$", fname)
        if m:
            csv_by_res[m.group(1)] = path

    for res in RESOLUTIONS:
        path = csv_by_res.get(res)
        if path is None:
            print(f"WARNING: {scheme}: no matching table for resolution {res}")
            results.append({"scheme": scheme, "resolution": res, "source": "table", "overlaps": {}})
            continue

        try:
            import pandas as pd

            df = pd.read_csv(path, keep_default_na=False)
            df = df.astype(str).apply(lambda col: col.str.strip())
            # Build annotation -> overlap_fraction map (last row wins if duplicate)
            ov = {}
            for _, row in df.iterrows():
                annot = row["annotation"].strip()
                try:
                    val = float(row["overlap_fraction"])
                except (KeyError, ValueError):
                    print(f"WARNING: {scheme} r{res}: bad overlap_fraction for '{annot}'")
                    continue
                ov[annot] = round(val, 4)

            overlaps = {}
            for tissue in TISSUES:
                if tissue in ov:
                    overlaps[tissue] = ov[tissue]
                else:
                    print(f"NOTE: {scheme} r{res}: tissue '{tissue}' absent from matching table -> null")
                    overlaps[tissue] = None

            results.append({"scheme": scheme, "resolution": res, "source": "table", "overlaps": overlaps})
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: {scheme}: failed to read {path}: {e!r}")
            results.append({"scheme": scheme, "resolution": res, "source": "table", "overlaps": {}})

# Emit one JSON object on the very last line
print(json.dumps({"results": results}, ensure_ascii=False))
