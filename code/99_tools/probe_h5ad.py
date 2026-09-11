import h5py
import numpy as np

path = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
f = h5py.File(path, "r")

print("=== top-level keys ===")
print(list(f.keys()))

print("\n=== X (expression matrix) ===")
x = f["X"]
if isinstance(x, h5py.Group):
    print("X is a GROUP (sparse). keys:", list(x.keys()))
else:
    print("X is a DATASET (dense). shape:", x.shape)
print("X attrs:", {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in x.attrs.items()})

print("\n=== obs (cell metadata) ===")
obs = f["obs"]
print("obs keys:", list(obs.keys()))
if "_index" in obs:
    idx = obs["_index"][()]
    print("n_obs:", len(idx))
    print("first 3 obs index values:", idx[:3])
for k in obs.keys():
    if k == "__categories":
        print("  __categories keys:", list(obs[k].keys()))
        continue
    item = obs[k]
    if isinstance(item, h5py.Dataset):
        d = item[()] if item.shape and item.shape[0] <= 500000 else None
        sample = d[:5] if d is not None else None
        nuniq = None
        if d is not None and d.ndim == 1:
            try:
                nuniq = len(np.unique(d))
            except Exception:
                pass
        print(f"  [ds] {k}: shape={item.shape} dtype={item.dtype} n_unique={nuniq} sample={sample}")
    else:
        codes = item["codes"][()] if "codes" in item else None
        cats = item["categories"][()] if "categories" in item else None
        print(f"  [cat] {k}: n_codes={len(codes) if codes is not None else '?'} categories={cats[:20] if cats is not None else '?'}")

print("\n=== var (gene metadata) ===")
var = f["var"]
print("var keys:", list(var.keys()))
if "_index" in var:
    vidx = var["_index"][()]
    print("n_var:", len(vidx))
    print("first 5 gene names:", vidx[:5])
for k in var.keys():
    if k in ("_index", "__categories"):
        continue
    item = var[k]
    if isinstance(item, h5py.Dataset):
        print(f"  [ds] {k}: shape={item.shape} dtype={item.dtype}")
    else:
        cats = item["categories"][()] if "categories" in item else None
        print(f"  [cat] {k}: categories={cats[:20] if cats is not None else '?'}")

print("\n=== obsm (per-cell matrices, e.g. spatial coords) ===")
if "obsm" in f:
    for k in f["obsm"].keys():
        item = f["obsm"][k]
        if isinstance(item, h5py.Dataset):
            print(f"  {k}: shape={item.shape} dtype={item.dtype}")
            if item.ndim == 2 and item.shape[0] > 0:
                print("    first 5 rows:", item[:5, :min(item.shape[1], 5)])
        else:
            print(f"  {k}: group attrs={dict(item.attrs)}")

print("\n=== obsp (pairwise matrices) ===")
if "obsp" in f:
    for k in f["obsp"].keys():
        item = f["obsp"][k]
        print(f"  {k}: attrs={dict(item.attrs)}")

print("\n=== layers ===")
if "layers" in f:
    print("layers keys:", list(f["layers"].keys()))

print("\n=== uns (unstructured, depth-limited) ===")
if "uns" in f:
    def walk(g, prefix="", depth=0):
        if depth > 3:
            return
        for k in g.keys():
            p = f"{prefix}{k}"
            item = g[k]
            if isinstance(item, h5py.Dataset):
                d = item[()] if item.shape and item.shape[0] <= 50 else None
                print(f"  {p}: shape={item.shape} dtype={item.dtype} val={d}")
            else:
                print(f"  {p}/: keys={list(item.keys())[:20]}")
                walk(item, p + "/", depth + 1)
    walk(f["uns"])

f.close()
print("\nDONE")
