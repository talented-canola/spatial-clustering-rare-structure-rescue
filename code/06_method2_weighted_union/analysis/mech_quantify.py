import numpy as np
import anndata as ad
from sklearn.neighbors import NearestNeighbors
import json

A = ad.read_h5ad('F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = A.obs['annotation'].to_numpy()
xy = np.asarray(A.obsm['spatial'], dtype=np.float64)
pca = np.asarray(A.obsm['X_pca'], dtype=np.float64)
n = ann.shape[0]
idx = np.arange(n)

# ---- spatial KNN (k=8 neighbors, n_neighbors=9) ----
nn_spat = NearestNeighbors(n_neighbors=9).fit(xy)
d_spat, i_spat = nn_spat.kneighbors(xy)  # (n,9)
spat_d = d_spat[:, 1:]                    # (n,8) distances to 8 spatial neighbors
spat_i = i_spat[:, 1:]                    # (n,8) indices

# expression distance to those same 8 spatial neighbors
pca_norm = np.asarray(pca)
expr_a = np.linalg.norm(pca_norm[spat_i] - pca_norm[:, None, :], axis=2)  # (n,8)

# purity / frac_diff based on the 8 spatial neighbors
neigh_ann = ann[spat_i]                          # (n,8)
self_ann = ann[:, None]
purity = (neigh_ann == self_ann).mean(axis=1)
frac_diff = (neigh_ann != self_ann).mean(axis=1)

# ---- global medians (variant a) ----
ss_a = np.median(spat_d)
se_a = np.median(expr_a)
w_a = np.exp(-spat_d**2 / (2.0 * ss_a**2)) * np.exp(-expr_a**2 / (2.0 * se_a**2))
bilat_spat_deg = w_a.sum(axis=1)  # (n,)

# ---- expr KNN (k=30, n_neighbors=31) ----
nn_expr = NearestNeighbors(n_neighbors=31).fit(pca_norm)
d_expr, i_expr = nn_expr.kneighbors(pca_norm)
expr_b = d_expr[:, 1:]             # (n,30)
expr_i = i_expr[:, 1:]             # (n,30)
spat_b = np.linalg.norm(xy[expr_i] - xy[:, None, :], axis=2)  # (n,30)

ss_b = np.median(spat_b)
se_b = np.median(expr_b)
w_b = np.exp(-spat_b**2 / (2.0 * ss_b**2)) * np.exp(-expr_b**2 / (2.0 * se_b**2))
bilat_expr_deg = w_b.sum(axis=1)  # (n,)

print(f'ss_a={ss_a:.6f} se_a={se_a:.6f} ss_b={ss_b:.6f} se_b={se_b:.6f}')

tissues = ['Surface ectoderm', 'Liver', 'Heart', 'Brain', 'Mesenchyme',
           'Head mesenchyme', 'Cavity', 'Spinal cord', 'Dorsal root ganglion',
           'GI tract', 'Connective tissue']

rows = []
for t in tissues:
    m = ann == t
    rows.append({
        'tissue': t,
        'n_cells': int(m.sum()),
        'spatial_purity_mean': float(purity[m].mean()),
        'bilat_spatial_deg_mean': float(bilat_spat_deg[m].mean()),
        'bilat_expr_deg_mean': float(bilat_expr_deg[m].mean()),
        'frac_neigh_diff_tissue': float(frac_diff[m].mean()),
    })

m_other = ann != 'Surface ectoderm'
rows.append({
    'tissue': 'ALL_OTHER',
    'n_cells': int(m_other.sum()),
    'spatial_purity_mean': float(purity[m_other].mean()),
    'bilat_spatial_deg_mean': float(bilat_spat_deg[m_other].mean()),
    'bilat_expr_deg_mean': float(bilat_expr_deg[m_other].mean()),
    'frac_neigh_diff_tissue': float(frac_diff[m_other].mean()),
})

out = {'tissue_stats': rows}
print(json.dumps(out, ensure_ascii=False))
