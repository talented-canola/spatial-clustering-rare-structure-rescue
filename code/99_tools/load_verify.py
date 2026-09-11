import anndata as ad
import numpy as np

path = r"F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad"

print("reading h5ad ...")
adata = ad.read_h5ad(path)
print("raw shape:", adata.shape)

# index is auto-read correctly (obs.index.name='cell_name', var.index.name='gene_short_name')
print("obs index sample:", list(adata.obs_names[:3]))
print("var index sample:", list(adata.var_names[:3]))

# Annotation
ann = adata.obs["annotation"]
print("\nannotation dtype:", ann.dtype, "| is categorical:", str(ann.dtype))
if hasattr(ann, "cat"):
    print("annotation categories (%d):" % len(ann.cat.categories), list(ann.cat.categories))
    print("counts:", dict(ann.value_counts()))

# Spatial
print("\nobsm keys:", list(adata.obsm.keys()))
sp = adata.obsm["spatial"]
print("spatial shape:", sp.shape, "dtype:", sp.dtype)
print("spatial row0:", sp[0], "row1:", sp[1])

# Layers
print("\nlayers keys:", list(adata.layers.keys()))
cnt = adata.layers["count"]
print("count shape:", cnt.shape, "dtype:", cnt.dtype)
print("count sample nonzero:", cnt.data[:5] if hasattr(cnt, "data") else cnt[0, :5])

# X
X = adata.X
print("\nX type:", type(X).__name__, "shape:", X.shape, "dtype:", X.dtype)
print("X min/max/mean:", float(X.min()), float(X.max()), float(X.mean()))
print("X has NaN:", bool(np.isnan(X.data).any()) if hasattr(X, "data") else bool(np.isnan(X).any()))

# obs columns summary (non-module/regulon)
core_cols = [c for c in adata.obs.columns if not c.startswith(("Module", "Regulon"))]
print("\ncore obs cols:", core_cols)
print("n_genes_by_counts range:", float(adata.obs["n_genes_by_counts"].min()), "-", float(adata.obs["n_genes_by_counts"].max()))
print("total_counts range:", float(adata.obs["total_counts"].min()), "-", float(adata.obs["total_counts"].max()))

print("\nLOAD OK")
