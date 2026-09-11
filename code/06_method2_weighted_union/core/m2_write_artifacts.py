"""VERIFIER: write method2 weighted-union scan CSV + fig7 figure.

CSV  F:/BGI/task3/spateo-release-main/results/analysis/method2_weighted_union_scan.csv  (UTF-8 BOM)
FIG  F:/BGI/task3/spateo-release-main/results/analysis/figures/fig7_method2_weighted_union_scan.png
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CSV = r"F:/BGI/task3/spateo-release-main/results/analysis/method2_weighted_union_scan.csv"
FIG = r"F:/BGI/task3/spateo-release-main/results/analysis/figures/fig7_method2_weighted_union_scan.png"

COLS = ["scheme", "Ws", "We", "n_clusters", "ARI", "NMI", "Silhouette", "coherence",
        "Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord",
        "Dorsal root ganglion", "GI tract", "Surface ectoderm"]

# scan rows (W_e = 1.0)
scan = [
    ("W_s=0.2", 0.2, 27, 0.4390, 0.6994, 0.1274, 0.8505,
     0.4160, 0.3570, 0.3810, 0.2686, 0.0, 0.9924, 0.8655, 0.6240),
    ("W_s=0.3", 0.3, 26, 0.3961, 0.6856, 0.1174, 0.8499,
     0.3272, 0.3589, 0.3883, 0.2780, 0.0, 0.0, 0.8855, 0.4442),
    ("W_s=0.5", 0.5, 27, 0.4041, 0.6819, 0.1214, 0.8617,
     0.4126, 0.3844, 0.3872, 0.2481, 0.0, 0.9924, 0.7450, 0.4689),
    ("W_s=0.7", 0.7, 25, 0.4077, 0.6779, 0.1186, 0.8745,
     0.4419, 0.3774, 0.3748, 0.2384, 0.0, 0.0, 0.8293, 0.4711),
    ("W_s=1.0", 1.0, 25, 0.4349, 0.6906, 0.1132, 0.8738,
     0.4806, 0.3678, 0.3805, 0.2449, 0.0, 0.9924, 0.7390, 0.4667),
]
# reference rows from STORED labels (m2_refs.py)
refs = [
    ("baseline_v2_arpack", np.nan, 28, 0.4234, 0.6987, 0.1337, 0.8405,
     0.4383, 0.3742, 0.3840, 0.2891, 1.0, 0.9924, 0.8594, 0.7119),
    ("scc_s8", np.nan, 22, 0.4437, 0.6979, 0.0995, 0.8791,
     0.4188, 0.3710, 0.3761, 0.2541, 0.0, 0.0, 0.8675, 0.4607),
]

rows = []
for name, ws, n, ari, nmi, sil, coh, b, m, hm, c, sc_, drg, gi, se in scan:
    rows.append((name, ws, 1.0, n, ari, nmi, sil, coh, b, m, hm, c, sc_, drg, gi, se))
for name, ws, n, ari, nmi, sil, coh, b, m, hm, c, sc_, drg, gi, se in refs:
    rows.append((name, ws, 1.0, n, ari, nmi, sil, coh, b, m, hm, c, sc_, drg, gi, se))

df = pd.DataFrame(rows, columns=COLS)
os.makedirs(os.path.dirname(CSV), exist_ok=True)
df.to_csv(CSV, index=False, encoding="utf-8-sig")
print("WROTE", CSV)
print(df.to_string(index=False))

# ---------------- FIGURE ----------------
TARGETS = ["Spinal cord", "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
TCOL = {"Spinal cord": "#d62728", "Dorsal root ganglion": "#2ca02c",
        "GI tract": "#1f77b4", "Surface ectoderm": "#ff7f0e"}
CLASS_ORDER = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity",
               "Spinal cord", "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
WS = [0.2, 0.3, 0.5, 0.7, 1.0]

baseline_ari = 0.4234
scc8_ari = 0.4437

scan_rows = df[df["scheme"].str.startswith("W_s")].set_index("Ws").loc[WS]
ari = scan_rows["ARI"].values
nmi = scan_rows["NMI"].values

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# Panel A
axA.plot(WS, ari, "o-", color="#1f77b4", label="ARI", ms=7, lw=2)
axA.plot(WS, nmi, "s-", color="#ff7f0e", label="NMI", ms=7, lw=2)
axA.axhline(baseline_ari, color="gray", ls="--", lw=1.2, label=f"baseline ARI = {baseline_ari:.4f}")
axA.axhline(scc8_ari, color="k", ls="--", lw=1.4, label=f"scc_s8 ARI = {scc8_ari:.4f}")
for x, y in zip(WS, ari):
    axA.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=8, color="#1f77b4")
axA.set_xlabel("Spatial weight W_s  (W_e = 1, res = 1.0)")
axA.set_ylabel("Score")
axA.set_title("A  ARI & NMI vs W_s  (weighted-union scan)", fontsize=11)
axA.set_xticks(WS)
axA.set_xticklabels([str(w) for w in WS])
axA.grid(alpha=0.3)
axA.legend(fontsize=8, loc="lower left")

# Panel B: heatmap
mat = df[df["scheme"] == "scc_s8"][CLASS_ORDER].values[0]
for w in WS:
    r = scan_rows.loc[w][CLASS_ORDER].values.astype(float)
    mat = np.vstack([mat, r])
# rows: scc_s8 first col then W_s 0.2..1.0; reorder so x-col = [0.2,0.3,0.5,0.7,1.0,scc_s8]
m2 = np.vstack([mat[1:], mat[0:1]]).T  # cols: W_s 0.2..1.0 then scc_s8; rows: 8 classes

im = axB.imshow(m2, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
axB.set_xticks(range(6))
axB.set_xticklabels(["0.2", "0.3", "0.5", "0.7", "1.0", "scc_s8"], fontsize=8, rotation=45)
axB.set_yticks(range(8))
ytl = []
for c in CLASS_ORDER:
    ytl.append(axB.get_yticklabels())
axB.set_yticklabels(CLASS_ORDER, fontsize=8)
for i, c in enumerate(CLASS_ORDER):
    lbl = axB.get_yticklabels()[i]
    if c in TARGETS:
        lbl.set_fontweight("bold")
        lbl.set_color(TCOL[c])
for i in range(8):
    for j in range(6):
        axB.text(j, i, f"{m2[i, j]:.3f}", ha="center", va="center",
                 fontsize=7,
                 color="k" if not (0.25 < m2[i, j] < 0.85) else "w")
axB.set_title("B  Class overlap by W_s  (scc_s8 reference col)", fontsize=11)
fig.colorbar(im, ax=axB, shrink=0.85, label="overlap fraction")

fig.tight_layout()
os.makedirs(os.path.dirname(FIG), exist_ok=True)
fig.savefig(FIG, dpi=130)
print("WROTE", FIG)
