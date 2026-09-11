import h5py
import numpy as np
from collections import Counter

path = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"
f = h5py.File(path, "r")

obs = f["obs"]

print("=== __categories (categorical obs columns) ===")
if "__categories" in obs:
    for k in obs["__categories"].keys():
        cats = obs["__categories"][k][()]
        print(f"  {k}: {len(cats)} categories -> {cats}")

print("\n=== annotation column ===")
if "annotation" in obs:
    a = obs["annotation"]
    print("type:", type(a).__name__)
    if isinstance(a, h5py.Dataset):
        vals = a[()]
        print("dtype:", a.dtype, "shape:", a.shape)
        print("n_unique:", len(np.unique(vals)))
        print("value counts:")
        for k, v in Counter(vals).most_common():
            print(f"    {k}: {v}")
    else:
        codes = a["codes"][()]
        cats = a["categories"][()]
        print("categorical: n_codes", len(codes), "n_cats", len(cats))
        cnt = Counter(codes)
        for ci, v in cnt.most_common():
            print(f"    {cats[ci]}: {v}")

print("\n=== cell_name column (first 15) ===")
if "cell_name" in obs:
    print(obs["cell_name"][()][:15])

print("\n=== obs index key? ===")
print("has _index:", "_index" in obs)

print("\n=== var ===")
var = f["var"]
print("var keys:", list(var.keys()))
vidx = var["_index"][()]
print("n_var:", len(vidx), "first 8:", vidx[:8], "last 3:", vidx[-3:])

print("\n=== obsm ===")
for k in f["obsm"].keys():
    item = f["obsm"][k]
    if isinstance(item, h5py.Dataset):
        print(f"  {k}: shape={item.shape} dtype={item.dtype}")
        if item.ndim == 2 and item.shape[1] <= 8:
            print("    rows 0-9:")
            for r in item[:10]:
                print("      ", list(r))
        if item.ndim == 2 and item.shape[1] > 8:
            print("    row0 (first 8):", list(item[0][:8]))
    else:
        print(f"  {k}: group attrs={dict(item.attrs)} keys={list(item.keys())}")

print("\n=== layers ===")
if "layers" in f:
    for k in f["layers"].keys():
        item = f["layers"][k]
        print(f"  {k}: {dict(item.attrs)}")

print("\n=== uns (1 level) ===")
if "uns" in f:
    for k in f["uns"].keys():
        item = f["uns"][k]
        if isinstance(item, h5py.Dataset):
            print(f"  {k}: dataset shape={item.shape} dtype={item.dtype}")
        else:
            print(f"  {k}: group keys={list(item.keys())[:30]}")

print("\n=== X value stats (sample of nonzero entries) ===")
x = f["X"]
d = x["data"][:200000]
print("X.data dtype:", d.dtype)
print("min:", d.min(), "max:", d.max(), "mean:", d.mean())
print("all integers?", np.all(d == np.round(d)))
print("num zeros in sample:", int(np.sum(d == 0)))

f.close()
print("\nDONE")
