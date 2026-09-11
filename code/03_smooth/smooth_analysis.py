"""Analysis: (1) thin-strip cluster detection in smooth_s8; (2) ARI incl_self vs excl_self."""
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RES = r"F:/BGI/task3/spateo-release-main/results"

# (1) thin-strip detection in smooth_s8 (res=1.0, 35 clusters)
excl = sc.read_h5ad(f"{RES}/smooth_s8/data/smooth_s8_slim.h5ad")
xy = excl.obsm["spatial"]
lab = excl.obs["smooth_s8_r1.0"].astype(int).values
ann = excl.obs["annotation"].astype(str).values
print("=== smooth_s8 (excl self) thin-strip clusters (res=1.0) ===")
print(f"{'cluster':>7} {'n_cells':>8} {'bbox_w':>8} {'bbox_h':>8} {'aspect':>7}")
strips = []
for c in np.unique(lab):
    m = lab == c
    pts = xy[m]
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    ar = max(w, h) / max(min(w, h), 1e-9)
    nc = int(m.sum())
    if ar > 3.5 and nc < 250:
        strips.append((c, nc, ar))
        print(f"{c:>7} {nc:>8} {w:>8.0f} {h:>8.0f} {ar:>7.1f}")
print(f"\n# elongated small clusters (aspect>3.5, n<250): {len(strips)}")

# (2) ARI comparison
incl = sc.read_h5ad(f"{RES}/smooth_s8_incl_self/data/smooth_s8_incl_self_slim.h5ad")
print("\n=== ARI/NMI incl_self vs excl_self ===")
print(f"{'res':>5} {'excl_clusters':>14} {'excl_ARI':>9} {'incl_clusters':>14} {'incl_ARI':>9} {'incl_NMI':>9}")
for res in [0.5, 1.0, 1.5, 2.0]:
    rs = str(res)
    le = excl.obs[f"smooth_s8_r{rs}"].astype(str).values
    li = incl.obs[f"smooth_s8_incl_self_r{rs}"].astype(str).values
    ari_e = adjusted_rand_score(ann, le)
    ari_i = adjusted_rand_score(ann, li)
    nmi_i = normalized_mutual_info_score(ann, li)
    print(f"{rs:>5} {len(np.unique(le)):>14} {ari_e:>9.4f} {len(np.unique(li)):>14} {ari_i:>9.4f} {nmi_i:>9.4f}")
