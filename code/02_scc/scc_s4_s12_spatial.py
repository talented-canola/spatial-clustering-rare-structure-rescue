"""Add res=1.0 Hungarian-aligned spatial figure for scc_s4 and scc_s12 + update READMEs."""
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

RES = r"F:/BGI/task3/spateo-release-main/results"
slim = sc.read_h5ad(os.path.join(RES, "scc_s4", "data", "scc_s4812_slim.h5ad"))
ann = slim.obs["annotation"].astype(str).values
ann_cats = list(slim.obs["annotation"].cat.categories)
xy = slim.obsm["spatial"]
pool = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
ann_colors = {c: pool[i] for i, c in enumerate(ann_cats)}

def make_figure(sid, label_col):
    lab = slim.obs[label_col].astype(str).values
    ct = pd.crosstab(ann, lab)
    n_ann, n_cl = ct.shape
    N = max(n_ann, n_cl)
    Cpad = np.zeros((N, N)); Cpad[:n_ann, :n_cl] = ct.values.astype(float)
    ri, ci = linear_sum_assignment(-Cpad)
    clusters = list(ct.columns)
    ann2cl = {ann_cats[i]: clusters[ci[i]] for i in range(n_ann) if ci[i] < n_cl}
    cl2col = {cl: ann_colors[a] for a, cl in ann2cl.items()}
    matched = set(ann2cl.values())
    leftover = [c for c in clusters if c not in matched]
    for i, cl in enumerate(leftover):
        cl2col[cl] = pool[19 + i]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for a in ann_cats:
        m = (ann == a)
        axes[0].scatter(xy[m, 0], xy[m, 1], s=2, color=ann_colors[a], rasterized=True)
    axes[0].set_title("Annotation"); axes[0].set_aspect("equal"); axes[0].invert_yaxis()
    for cl in clusters:
        m = (lab == cl)
        axes[1].scatter(xy[m, 0], xy[m, 1], s=2, color=cl2col[cl], rasterized=True)
    axes[1].set_title(f"{sid} (res=1.0, re-colored)"); axes[1].set_aspect("equal"); axes[1].invert_yaxis()
    plt.tight_layout()
    outdir = os.path.join(RES, sid, "figures")
    os.makedirs(outdir, exist_ok=True)
    p = os.path.join(outdir, "spatial_aligned_r1.0.png")
    plt.savefig(p, dpi=120); plt.close()
    print(f"saved {p}")

make_figure("scc_s4", "scc_s4_r1.0")
make_figure("scc_s12", "scc_s12_r1.0")

# update READMEs: add figures/ entry to 产出文件清单, tweak 已知问题
for sid, note in [("scc_s4", "已补 res=1.0 空间图（支撑 s=12 改善 Brain 但吞 DRG 的可视化）"),
                  ("scc_s12", "已补 res=1.0 空间图（支撑 s=12 改善 Brain 但吞 DRG 的可视化）")]:
    p = os.path.join(RES, sid, "README.md")
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    marker = "## 本方案产出文件清单\n"
    fig_line = f"- figures/:\n  - spatial_aligned_r1.0.png\n"
    if "spatial_aligned_r1.0.png" not in txt:
        txt = txt.replace(marker, marker + fig_line, 1)
    # append note to 已知问题
    txt = txt.rstrip() + "\n- " + note + "\n"
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"updated {sid}/README.md")
