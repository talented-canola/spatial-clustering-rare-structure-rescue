import h5py
import numpy as np

path = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
f = h5py.File(path, "r")

print("=== var gene names ===")
var = f["var"]
vkey = "gene_short_name" if "gene_short_name" in var else None
if vkey:
    vidx = var[vkey][()]
    print("n_var:", len(vidx), "first 8:", [x.decode() if isinstance(x, bytes) else x for x in vidx[:8]],
          "last 3:", [x.decode() if isinstance(x, bytes) else x for x in vidx[-3:]])

print("\n=== obsm (spatial coords) ===")
for k in f["obsm"].keys():
    item = f["obsm"][k]
    if isinstance(item, h5py.Dataset):
        print(f"  {k}: shape={item.shape} dtype={item.dtype}")
        if item.ndim == 2 and item.shape[1] <= 8:
            print("    first 10 rows:")
            for r in item[:10]:
                print("      ", list(r))
            # grid spacing analysis
            arr = item[:500]
            print("    min per col:", arr.min(axis=0), "max per col:", arr.max(axis=0))
            if arr.shape[1] >= 2:
                dx = np.diff(np.sort(np.unique(arr[:, 0])))
                dy = np.diff(np.sort(np.unique(arr[:, 1])))
                print("    unique x spacing (min/max/median):", dx.min(), dx.max(), np.median(dx) if len(dx) else None)
                print("    unique y spacing (min/max/median):", dy.min(), dy.max(), np.median(dy) if len(dy) else None)
        else:
            print(f"    row0 first 8:", list(item[0][:8]))
    else:
        print(f"  {k}: group attrs={dict(item.attrs)} keys={list(item.keys())}")

print("\n=== layers ===")
if "layers" in f:
    for k in f["layers"].keys():
        item = f["layers"][k]
        print(f"  {k}: {dict(item.attrs)}")

print("\n=== varm ===")
if "varm" in f:
    print("  keys:", list(f["varm"].keys()))

print("\n=== uns (top-level) ===")
if "uns" in f:
    for k in f["uns"].keys():
        item = f["uns"][k]
        if isinstance(item, h5py.Dataset):
            print(f"  {k}: dataset shape={item.shape} dtype={item.dtype}")
        else:
            print(f"  {k}: group keys={list(item.keys())[:30]}")

print("\n=== X value stats ===")
x = f["X"]
d = x["data"][:]
print("X.data dtype:", d.dtype, "n_nonzero:", d.shape[0])
print("min:", float(d.min()), "max:", float(d.max()), "mean:", float(d.mean()))
print("all non-negative ints?", bool(np.all(d == np.round(d))), "| all >= 0:", bool(np.all(d >= 0)))
print("sum (total counts proxy):", float(d.sum()))

f.close()
print("\nDONE")
