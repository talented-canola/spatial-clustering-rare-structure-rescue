"""Marker verification figure: dot plot of log2 fold-change vs background for
SC / DRG candidate markers + negative controls. Dot color = region (SC vs DRG),
dot size = % bins expressing in that region. Reads the archived marker_verify.csv.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables/marker_verify.csv"
OUT_PNG = "F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/figures/marker_verify.png"

SC_GENES = ["Olig2", "Pax6", "Mnx1", "Isl1", "Nkx2.2", "Sox2", "Neurog1",
            "Neurod1", "Tubb3", "Nefm", "Nkx6-1", "Olig1"]
DRG_GENES = ["Sox10", "Prph", "Ntrk1", "Ngfr", "Pmp2", "S100b", "Pou4f1",
             "Ntrk2", "Ntrk3", "Nefl", "Ret", "Pax2", "Runx1", "Etv1"]
NEG_GENES = ["Krt14", "Col2a1", "Myh11", "Cyp2e1", "Prss28", "Alb", "Lyz2"]

df = pd.read_csv(CSV, encoding="utf-8-sig")
df = df.drop_duplicates("gene", keep="first").set_index("gene")
# coerce numeric (empty strings from present=False rows become NaN)
for c in ["SC_mean_log1p", "DRG_mean_log1p", "bg_mean_log1p", "SC_bg_fc", "DRG_bg_fc",
          "SC_pct", "DRG_pct", "bg_pct"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def log2fc(v):
    return np.log2(np.asarray(v, dtype=float))

# order each group by its own primary fold-change (descending)
def ordered(genes, key):
    vals = {g: (df.loc[g, key] if g in df.index and pd.notna(df.loc[g, key]) else -np.inf) for g in genes}
    return sorted(genes, key=lambda g: vals[g], reverse=True)

sc_ord = ordered(SC_GENES, "SC_bg_fc")
drg_ord = ordered(DRG_GENES, "DRG_bg_fc")
neg_ord = NEG_GENES  # keep natural order

groups = [("Spinal cord candidates", sc_ord, "SC_bg_fc", "SC_pct"),
          ("Dorsal root ganglion candidates", drg_ord, "DRG_bg_fc", "DRG_pct"),
          ("Negative controls", neg_ord, "SC_bg_fc", "SC_pct")]

SC_COLOR = "#1f77b4"   # blue = SC region
DRG_COLOR = "#ff7f0e"  # orange = DRG region

# vertical layout proportional to gene count
n_rows = [len(g[1]) for g in groups]
heights = [n / max(n_rows) for n in n_rows]
fig, axes = plt.subplots(3, 1, figsize=(9.5, 2.2 + 0.34 * sum(n_rows)),
                         gridspec_kw={"height_ratios": heights}, sharex=True)

fc_ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

for ax, (title, genes, key, pct_key) in zip(axes, groups):
    ys = np.arange(len(genes))
    for y, g in zip(ys, genes):
        if g not in df.index:
            continue
        r = df.loc[g]
        sc_fc, drg_fc = r["SC_bg_fc"], r["DRG_bg_fc"]
        sc_pct, drg_pct = r["SC_pct"], r["DRG_pct"]
        if pd.isna(sc_fc) and pd.isna(drg_fc):  # not detected
            ax.text(0, y, "N.D.", ha="center", va="center", color="0.4", fontsize=8, style="italic")
            continue
        if pd.notna(sc_fc):
            ax.scatter(log2fc(sc_fc), y, s=25 + 1.6 * (sc_pct if pd.notna(sc_pct) else 0),
                       color=SC_COLOR, alpha=0.85, edgecolors="none", zorder=3)
        if pd.notna(drg_fc):
            ax.scatter(log2fc(drg_fc), y, s=25 + 1.6 * (drg_pct if pd.notna(drg_pct) else 0),
                       color=DRG_COLOR, alpha=0.85, edgecolors="none", zorder=3)
    ax.axvline(0, color="0.3", lw=1.0, ls="--", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(genes, fontsize=9)
    ax.set_ylim(-0.8, len(genes) - 0.2)
    ax.set_title(title, fontsize=11, loc="left", pad=6)
    ax.grid(axis="x", color="0.9", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

axes[-1].set_xticks(log2fc(fc_ticks))
axes[-1].set_xticklabels([f"{t:g}" for t in fc_ticks])
axes[-1].set_xlabel("fold-change vs background (log2 scale)", fontsize=10)
axes[0].set_xlim(log2fc(0.2), log2fc(80))

# legend
handles = [Line2D([], [], marker="o", linestyle="", color=SC_COLOR, markersize=7, label="SC region"),
           Line2D([], [], marker="o", linestyle="", color=DRG_COLOR, markersize=7, label="DRG region")]
for pct in [25, 50, 75]:
    handles.append(Line2D([], [], marker="o", linestyle="", color="0.55",
                          markersize=np.sqrt(25 + 1.6 * pct) * 0.6,
                          label=f"{pct}% bins expressing"))
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=5, frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.2)

fig.suptitle("Marker gene verification: rescued SC / DRG regions vs whole-tissue background",
             fontsize=12.5, y=0.995, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
print("saved", OUT_PNG)
