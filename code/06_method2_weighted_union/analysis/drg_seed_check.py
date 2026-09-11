# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [
    {"Ws":0.2,"seed":888,"n_clusters":27,"ARI":0.439,"NMI":0.6994,"Silhouette":0.1274,"coherence":0.8505,"Brain":0.416,"Mesenchyme":0.357,"Head mesenchyme":0.381,"Cavity":0.2686,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.8655,"Surface ectoderm":0.624},
    {"Ws":0.2,"seed":1,"n_clusters":24,"ARI":0.4335,"NMI":0.6961,"Silhouette":0.1238,"coherence":0.8519,"Brain":0.4024,"Mesenchyme":0.3097,"Head mesenchyme":0.3177,"Cavity":0.2672,"Spinal cord":0,"Dorsal root ganglion":0,"GI tract":0.8876,"Surface ectoderm":0.6949},
    {"Ws":0.2,"seed":2,"n_clusters":26,"ARI":0.4306,"NMI":0.6929,"Silhouette":0.1375,"coherence":0.845,"Brain":0.4134,"Mesenchyme":0.3544,"Head mesenchyme":0.3398,"Cavity":0.2737,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.8755,"Surface ectoderm":0.4574},
    {"Ws":0.2,"seed":3,"n_clusters":26,"ARI":0.4113,"NMI":0.6906,"Silhouette":0.1389,"coherence":0.8518,"Brain":0.4498,"Mesenchyme":0.3736,"Head mesenchyme":0.381,"Cavity":0.2466,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.8715,"Surface ectoderm":0.4695},
    {"Ws":0.2,"seed":4,"n_clusters":27,"ARI":0.4026,"NMI":0.6925,"Silhouette":0.1302,"coherence":0.8476,"Brain":0.3954,"Mesenchyme":0.3704,"Head mesenchyme":0.3805,"Cavity":0.2563,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.7369,"Surface ectoderm":0.4656},
    {"Ws":0.3,"seed":888,"n_clusters":26,"ARI":0.3961,"NMI":0.6856,"Silhouette":0.1174,"coherence":0.8499,"Brain":0.3272,"Mesenchyme":0.3589,"Head mesenchyme":0.3883,"Cavity":0.278,"Spinal cord":0,"Dorsal root ganglion":0,"GI tract":0.8855,"Surface ectoderm":0.4442},
    {"Ws":0.3,"seed":1,"n_clusters":27,"ARI":0.3934,"NMI":0.6823,"Silhouette":0.129,"coherence":0.8539,"Brain":0.4008,"Mesenchyme":0.3851,"Head mesenchyme":0.3522,"Cavity":0.2866,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.8835,"Surface ectoderm":0.4508},
    {"Ws":0.3,"seed":2,"n_clusters":26,"ARI":0.4047,"NMI":0.6823,"Silhouette":0.1218,"coherence":0.8565,"Brain":0.4137,"Mesenchyme":0.3665,"Head mesenchyme":0.3805,"Cavity":0.2683,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.8635,"Surface ectoderm":0.7372},
    {"Ws":0.3,"seed":3,"n_clusters":26,"ARI":0.4391,"NMI":0.6963,"Silhouette":0.1236,"coherence":0.8594,"Brain":0.5242,"Mesenchyme":0.3633,"Head mesenchyme":0.3791,"Cavity":0.2489,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.8675,"Surface ectoderm":0.4662},
    {"Ws":0.3,"seed":4,"n_clusters":27,"ARI":0.4203,"NMI":0.6956,"Silhouette":0.13,"coherence":0.852,"Brain":0.4373,"Mesenchyme":0.3059,"Head mesenchyme":0.3802,"Cavity":0.2729,"Spinal cord":0,"Dorsal root ganglion":0.9924,"GI tract":0.7369,"Surface ectoderm":0.7273},
]

# sha256 per seed (ws02 order)
sha = {"0.2_888":"affcec9b991bf89a","0.2_1":"77d8e455805f27f4","0.2_2":"40b6c6f603307ba3","0.2_3":"1f14f3013d2770c4","0.2_4":"2b3bc12fe3d18818",
       "0.3_888":"802dfc795c641855","0.3_1":"3c114d66c03ed474","0.3_2":"d3c302d2b4d809ca","0.3_3":"9bc4b26123b5b321","0.3_4":"e1f2c0855283fc83"}

# ---------- 1) CSV ----------
cols = ["Ws","seed","n_clusters","ARI","NMI","Silhouette","coherence","Brain","Mesenchyme","Head mesenchyme","Cavity","Spinal cord","Dorsal root ganglion","GI tract","Surface ectoderm"]
df = pd.DataFrame(ROWS, columns=cols)
csv_path = "F:/BGI/task3/spateo-release-main/results/analysis/method2_drg_seed_check.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("CSV written:", csv_path)
print(df.head(3).to_string())

# ---------- 2) Aggregates ----------
def agg(ws):
    sub = df[df["Ws"] == ws]
    drg = sub["Dorsal root ganglion"].values
    surf = sub["Surface ectoderm"].values
    ari = sub["ARI"].values
    sha_set = {sha[f"{ws}_{s}"] for s in sub["seed"].values}
    return {
        "drg_mean": float(np.mean(drg)),
        "drg_std": float(np.std(drg, ddof=1)),
        "drg_ge09": int(np.sum(drg >= 0.9)),
        "drg_le005": int(np.sum(drg <= 0.05)),
        "surf_mean": float(np.mean(surf)),
        "surf_std": float(np.std(surf, ddof=1)),
        "surf_min": float(np.min(surf)),
        "surf_max": float(np.max(surf)),
        "ari_mean": float(np.mean(ari)),
        "ari_std": float(np.std(ari, ddof=1)),
        "spc_mean": float(np.mean(sub["Spinal cord"].values)),
        "git_mean": float(np.mean(sub["GI tract"].values)),
        "distinct_partitions": len(sha_set),
    }

a02 = agg(0.2)
a03 = agg(0.3)
print("\nWS02:", json.dumps(a02, indent=2))
print("\nWS03:", json.dumps(a03, indent=2))

# ---------- 3) Judgment ----------
drg_stable_at_ws02 = (a02["drg_ge09"] >= 4) and (a02["drg_mean"] >= 0.8)
can_report = drg_stable_at_ws02
surf_stable_at_ws02 = not (a02["surf_std"] > 0.1)
ari_stable = a02["ari_std"] < 0.01

print("\nDRG stable at 0.2 (ge09>=4 and mean>=0.8):", drg_stable_at_ws02)
print("surf_stable_at_ws02 (std<=0.1):", surf_stable_at_ws02, "std=", round(a02["surf_std"],4))
print("ari_stable (std<0.01):", ari_stable, "std=", round(a02["ari_std"],5))
print("can_report_drg_rescued:", can_report)

# ---------- 4) Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
seeds = [888, 1, 2, 3, 4]
for ax, ws in zip(axes, [0.2, 0.3]):
    sub = df[df["Ws"] == ws].set_index("seed").reindex(seeds)
    x = [str(s) for s in seeds]
    drg = sub["Dorsal root ganglion"].values
    surf = sub["Surface ectoderm"].values
    ax.plot(x, drg, "r-o", label="DRG")
    ax.plot(x, surf, "b-s", label="Surface ectoderm")
    for i, (d, s) in enumerate(zip(drg, surf)):
        ax.annotate(f"{d:.2f}", (x[i], d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="red")
        ax.annotate(f"{s:.2f}", (x[i], s), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="blue")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("random state (seed)")
    ax.set_ylabel("overlap fraction")
    ax.set_title(f"Panel {'A' if ws==0.2 else 'B'}: W_s = {ws}")
    ax.legend(loc="lower left", fontsize=8)
fig.suptitle("DRG / Surface ectoderm overlap across random states (res=1.0)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig_path = "F:/BGI/task3/spateo-release-main/results/analysis/figures/fig8_drg_seed_check.png"
fig.savefig(fig_path, dpi=130)
print("Figure written:", fig_path)

# ---------- 5) JSON ----------
verdict = (
    f"At W_s=0.2, DRG overlap is >=0.9 in 4/5 seeds (888,2,3,4; mean {a02['drg_mean']:.4f}, "
    f"std {a02['drg_std']:.3f}) but seed=1 collapses to 0.0, so the mean {a02['drg_mean']:.4f} "
    f"falls just short of the 0.8 bar — strictly NOT reportable as stable rescue (mean<0.8). "
    f"Same 4/5 pattern recurs at W_s=0.3 (mean {a03['drg_mean']:.4f}), confirming the binary "
    f"jump is a seed-sensitive boundary artifact rather than a W_s=0.2-specific effect; "
    f"Surface ectoderm std {a02['surf_std']:.3f} (>0.1) and ARI std {a02['ari_std']:.4f} "
    f"(>=0.01) are also not stable. Report must state DRG rescue at W_s=0.2 is highly seed-sensitive/unstable."
)
out = {
    "verdict": verdict,
    "drg_ws02_mean": round(a02["drg_mean"], 4),
    "drg_ws02_std": round(a02["drg_std"], 4),
    "drg_ws02_ge09": a02["drg_ge09"],
    "drg_ws02_le005": a02["drg_le005"],
    "surf_ws02_mean": round(a02["surf_mean"], 4),
    "surf_ws02_std": round(a02["surf_std"], 4),
    "ari_ws02_mean": round(a02["ari_mean"], 4),
    "ari_ws02_std": round(a02["ari_std"], 4),
    "drg_ws03_mean": round(a03["drg_mean"], 4),
    "drg_ws03_ge09": a03["drg_ge09"],
    "drg_ws03_le005": a03["drg_le005"],
    "distinct_partitions_ws02": a02["distinct_partitions"],
    "drg_stable_at_ws02": bool(drg_stable_at_ws02),
    "surf_stable_at_ws02": bool(surf_stable_at_ws02),
    "ari_stable": bool(ari_stable),
    "can_report_drg_rescued": bool(can_report),
    "csv_path": csv_path,
    "fig_path": fig_path,
}
print("\nJSON:")
print(json.dumps(out, indent=2))
