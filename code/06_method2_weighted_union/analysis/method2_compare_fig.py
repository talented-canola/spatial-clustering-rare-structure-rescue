"""Cross-method comparison figure: method2 vs baseline_v2_arpack vs scc_s8 at res=1.0.
Panel 1: global metrics (ARI/NMI/Silhouette/coherence/SC/DRG). Panel 2: 19-class
Hungarian overlap fraction. method2 shown as 5-seed mean +/- sample std; the other two
are single-run (seed888 / single). Saves to analysis/figures/ per repo convention.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "F:/BGI/task3/spateo-release-main/results"
OUT = RES + "/analysis/figures/fig9_method2_vs_baseline_scc.png"

METHODS = ["baseline_v2_arpack", "scc_s8", "method2_weighted_union_final"]
LABELS = {"baseline_v2_arpack": "baseline_v2\n(non-spatial)",
          "scc_s8": "scc_s8\n(SCC union)",
          "method2_weighted_union_final": "method2\n(weighted union + fix)"}
COLORS = {"baseline_v2_arpack": "#9aa0a6", "scc_s8": "#1f77b4", "method2_weighted_union_final": "#d62728"}

# ---- global metrics at res=1.0 ------------------------------------------
# method2 = 5-seed mean; baseline/scc_s8 = single run (registry rows 7, 19)
metric_names = ["ARI", "NMI", "Silhouette", "Spatial coherence", "SC overlap", "DRG overlap"]
metric_vals = {
    "baseline_v2_arpack": [0.4234, 0.6987, 0.1348, 0.8405, 1.0, 0.9924],
    "scc_s8":             [0.4437, 0.6979, 0.1050, 0.8791, 0.0, 0.0],
    "method2_weighted_union_final": [0.4234, 0.6980, 0.1303, 0.8427, 1.0, 0.9924],
}
metric_std = {  # method2 only (5-seed sample std, ddof=1)
    "ARI": 0.0032, "NMI": 0.0044, "Silhouette": 0.0100, "Spatial coherence": 0.0050,
    "SC overlap": 0.0, "DRG overlap": 0.0,
}

# ---- 19-class overlap ---------------------------------------------------
def load_overlap(sid):
    df = pd.read_csv(f"{RES}/{sid}/tables/matching_r1.0.csv", encoding="utf-8-sig")
    if "annotation" in df.columns:
        return dict(zip(df["annotation"], df["overlap_fraction"]))
    else:  # method2: class / overlap / matched_cluster
        return dict(zip(df["class"], df["overlap"]))

ov = {m: load_overlap(m) for m in METHODS}
order = list(ov["baseline_v2_arpack"].keys())  # alphabetical annotation order

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13.5),
                               gridspec_kw={"height_ratios": [1.0, 2.2]})

# ---------------- Panel 1: global metrics ----------------
x = np.arange(len(metric_names))
w = 0.27
for i, m in enumerate(METHODS):
    off = (i - 1) * w
    bars = ax1.bar(x + off, metric_vals[m], w, color=COLORS[m], label=LABELS[m], zorder=3)
    if m == "method2":
        err = [metric_std[n] for n in metric_names]
        ax1.errorbar(x + off, metric_vals[m], yerr=err, fmt="none", ecolor="black",
                     elinewidth=1.2, capsize=3, zorder=4)
ax1.set_xticks(x)
ax1.set_xticklabels(metric_names, fontsize=10)
ax1.set_ylabel("value", fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.set_title("Global metrics (res=1.0)", fontsize=11, loc="left", pad=6)
ax1.grid(axis="y", color="0.9", lw=0.6, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)
ax1.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=3, bbox_to_anchor=(0, -0.02))

# ---------------- Panel 2: 19-class overlap ----------------
y = np.arange(len(order))[::-1]  # top = first class
h = 0.27
for i, m in enumerate(METHODS):
    vals = [ov[m][a] for a in order]
    off = (i - 1) * h
    ax2.barh(y + off, vals, h, color=COLORS[m], label=LABELS[m], zorder=3)
ax2.set_yticks(y)
ax2.set_yticklabels(order, fontsize=9)
ax2.set_xlabel("Hungarian-aligned overlap fraction (per 19-class annotation)", fontsize=10)
ax2.set_xlim(0, 1.0)
ax2.set_title("19-class identification rate (res=1.0)", fontsize=11, loc="left", pad=6)
ax2.grid(axis="x", color="0.9", lw=0.6, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)
# highlight the two small rescued classes
for a, yv in zip(order, y):
    if a in ("Spinal cord", "Dorsal root ganglion"):
        ax2.text(1.02, yv, "◀", color="#d62728", fontsize=9, va="center", ha="left",
                 transform=ax2.get_yaxis_transform(), clip_on=False)

fig.suptitle("method2 vs baseline_v2 vs scc_s8  (res=1.0; method2 = 5-seed mean ± sd)",
             fontsize=12.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
