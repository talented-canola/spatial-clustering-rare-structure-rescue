"""Fix hypothesis test: bilateral kernel over EXPRESSION-KNN (30) neighbors (not spatial-KNN),
weighted by spatial x expr Gaussian. Compare to baseline/SCC."""
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import fast_leiden as fl

BASE = r"F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
RES = [0.5, 1.0, 1.5, 2.0]
base = sc.read_h5ad(BASE)
ann = base.obs["annotation"].astype(str).values
xy = base.obsm["spatial"]
X_pca = base.obsm["X_pca"]
n = xy.shape[0]

def coherence(labels):
    li = labels.astype(int)
    nn = NearestNeighbors(n_neighbors=9).fit(xy)
    _, nidx = nn.kneighbors(xy); nidx = nidx[:, 1:]
    return float((li[nidx] == li[:, None]).mean())

# expression-KNN (30 neighbors)
nn = NearestNeighbors(n_neighbors=31).fit(X_pca)
edist, eidx = nn.kneighbors(X_pca)
eidx = eidx[:, 1:]; edist = edist[:, 1:]           # (n, 30) expr distances
# spatial distances for those expression-neighbors
sdist = np.linalg.norm(xy[eidx] - xy[:, None, :], axis=2)  # (n, 30)
sigma_e = float(np.median(edist))
sigma_s = float(np.median(sdist))
print(f"sigma_e={sigma_e:.3f}, sigma_s={sigma_s:.3f}")

rows_n = np.repeat(np.arange(n), 30); cols_n = eidx.ravel()
results = []
for mode, w in [("expr_only", np.exp(-(edist**2)/(2*sigma_e**2))),
                ("bilateral", np.exp(-(edist**2)/(2*sigma_e**2)) * np.exp(-(sdist**2)/(2*sigma_s**2))),
                ("spatial_only", np.exp(-(sdist**2)/(2*sigma_s**2)))]:
    adj = csr_matrix((w.ravel(), (rows_n, cols_n)), shape=(n, n))
    for res in RES:
        lab = fl.fast_leiden_weighted(adj, res)
        ari = adjusted_rand_score(ann, lab.astype(str))
        nmi = normalized_mutual_info_score(ann, lab.astype(str))
        coh = coherence(lab)
        results.append({"mode": mode, "res": res, "n_clusters": len(np.unique(lab)),
                        "ARI": round(ari, 4), "NMI": round(nmi, 4), "coherence": round(coh, 4)})

df = pd.DataFrame(results)
print(df.to_string(index=False))
print("\n(ref) baseline unweighted expr-KNN: res1.0 ARI=0.423, 28 clusters")
