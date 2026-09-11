"""生成方法示意图 docs/method_schematic.png（纯示意，非实验结果）。

上图：method2 的完整流程（加权并集 → 加权 Leiden → 共识块 → 确定性拆分）。
下图：三种图构建方式在同一组合成数据上的对比 —— 表达 KNN 图 E / SCC 并集 E∨S /
      加权并集。用合成点集真实计算 KNN，不是手绘假图。

所有点与边均为合成数据，仅用于说明方法机制；图中不含任何真实实验结果。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from sklearn.neighbors import NearestNeighbors

OUT = "docs/method_schematic.png"
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- 合成点集
def blob(cx, cy, n, spread):
    return np.c_[rng.normal(cx, spread, n), rng.normal(cy, spread, n)]

A = blob(0.30, 0.60, 13, 0.045)   # 大群 A
B = blob(0.76, 0.60, 13, 0.045)   # 大群 B
R = blob(0.30, 0.34, 5, 0.020)    # 稀有结构（空间上紧邻 A 的下缘）
X = np.vstack([A, B, R])
group = np.array(["A"] * len(A) + ["B"] * len(B) + ["R"] * len(R))

# 表达空间：以簇身份为主轴 —— 簇内表达相近，与空间位置无关
expr = np.array([[{"A": 0.0, "B": 1.0, "R": 0.5}[g]] for g in group])
expr = expr + rng.normal(0, 0.06, (len(group), 1))

# 表达 KNN 取 k=3：R 簇 5 个点互为近邻 → 在表达空间中自成一连通分量
# 空间 KNN 取 k=5：R 的 4 个内部邻居用尽后，第 5 个邻居必然落在紧邻的 A 上 → 形成桥接边
K_EXPR, K_SPAT = 3, 5


def knn_edges(feat, k):
    """KNN 是【有向】图：i→j 不一定伴随 j→i。

    注意不能用 `if i < j` 去重 —— 那会把"低位点指向高位点"的边整批丢掉
    （如 R 簇点指向索引更小的 A 簇点）。这里改为规范化为 (min, max) 再集合去重，
    保证方向无关且不丢边。
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feat)
    _, idx = nn.kneighbors(feat)
    return {(min(i, int(j)), max(i, int(j))) for i, row in enumerate(idx) for j in row[1:]}


E = knn_edges(expr, K_EXPR)   # 表达 KNN 边
S = knn_edges(X, K_SPAT)      # 空间 KNN 边

COL = {"A": "#4c72b0", "B": "#55a868", "R": "#c44e52"}


def draw_graph(ax, edges, title, subtitle, weighted=False):
    for i, j in edges:
        rare_edge = group[i] == "R" and group[j] == "R"
        if weighted and rare_edge:
            ax.plot(*zip(X[i], X[j]), color="#c44e52", lw=3.0, zorder=1, alpha=0.95)
        else:
            ax.plot(*zip(X[i], X[j]), color="0.62", lw=0.9, zorder=1, alpha=0.8)
    for g in ("A", "B", "R"):
        m = group == g
        ax.scatter(X[m, 0], X[m, 1], s=46 if g != "R" else 62, c=COL[g],
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.add_patch(Circle((0.30, 0.34), 0.072, fill=False, ls="--", lw=1.2,
                        ec="#c44e52", zorder=2))
    ax.set_title(title, fontsize=10.5, pad=7, fontweight="bold")
    ax.text(0.5, -0.06, subtitle, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.6, color="0.32")
    ax.set_xlim(0.12, 0.92); ax.set_ylim(0.20, 0.76)
    ax.set_aspect("equal"); ax.axis("off")


# ---------------------------------------------------------------- 画布
fig = plt.figure(figsize=(13.6, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[0.82, 1.0], hspace=0.30, wspace=0.12)

# ============ 上：流程 ============
axf = fig.add_subplot(gs[0, :]); axf.axis("off")
axf.set_xlim(0, 100); axf.set_ylim(0, 30)


def box(x, y, w, h, text, fc, ec="0.45", fs=8.8, bold=False):
    axf.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                 fc=fc, ec=ec, lw=1.1))
    axf.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
             fontweight="bold" if bold else "normal", linespacing=1.45)


def arrow(x1, y1, x2, y2, color="0.35", lw=1.5, style="-|>"):
    axf.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                  mutation_scale=13, color=color, lw=lw))


box(1, 17.5, 15.5, 9, "Expression KNN graph $E$\n(PCA-30, $e_{neigh}$=30)", "#dce8f5")
box(1, 3.5, 15.5, 9, "Spatial KNN graph $S$\n($s_{neigh}$=8)", "#dce8f5")
box(22, 10.5, 20, 9,
    "Weighted union  $A$\n$\\alpha E|_{block}+E+W_s S$\n($\\alpha$=1.5, $W_s$=0.2)",
    "#fde2c8", bold=False)
box(47, 10.5, 13.5, 9, "Weighted\nLeiden", "#fde2c8")
box(65.5, 10.5, 15, 9, "6-seed\nconsensus block $B$", "#e3d7f0")
box(85.5, 10.5, 13.5, 9, "Deterministic\nsplit", "#f6cccc", bold=True)

arrow(16.5, 22, 22, 17)
arrow(16.5, 8, 22, 13)
arrow(42, 15, 47, 15)
arrow(60.5, 15, 65.5, 15)
arrow(80.5, 15, 85.5, 15)
axf.text(90, 6.6, "rare structure recovered", ha="center", fontsize=8.4,
         color="#c44e52", style="italic")
arrow(92.2, 10.4, 92.2, 7.9, color="#c44e52", lw=1.3)
axf.text(1, 27.6, "method2 pipeline: weighted union + consensus-block relabeling",
         fontsize=11.5, fontweight="bold")

# ============ 下：三种图 ============
ax1 = fig.add_subplot(gs[1, 0])
draw_graph(ax1, E, "(a) baseline — expression graph $E$",
           "rare cluster isolated in expression space")

ax2 = fig.add_subplot(gs[1, 1])
draw_graph(ax2, E | S, "(b) SCC — boolean union $E \\vee S$",
           "spatial edges bridge the rare cluster into the main group")

ax3 = fig.add_subplot(gs[1, 2])
draw_graph(ax3, E | S, "(c) method2 — weighted union",
           "intra-block expression edges amplified by $\\alpha$", weighted=True)

for ax, note in ((ax2, "merged into main group"), (ax3, "kept separate")):
    ax.text(0.30, 0.252, note, ha="center", va="top", fontsize=8.4,
            color="#c44e52", fontweight="bold")

# (c) 面板标出块内表达边被放大的位置，否则与 (b) 的差异过于隐蔽
ax3.annotate("$\\alpha\\cdot E$  (intra-block\nedges amplified)",
             xy=(0.345, 0.355), xytext=(0.50, 0.30), fontsize=8.4,
             color="#c44e52", ha="left", va="center",
             arrowprops=dict(arrowstyle="-|>", color="#c44e52", lw=1.2,
                             shrinkA=0, shrinkB=2))

# ---------------------------------------------------------------- 自检
_bridge = [(i, j) for i, j in S if (group[i] == "R") != (group[j] == "R")]
_rin = [(i, j) for i, j in E if group[i] == "R" and group[j] == "R"]
print(f"[自检] 表达图中 R 簇内部边 = {len(_rin)}  (应为 4 条以上, 否则 (a) 面板无意义)")
print(f"[自检] 空间图中 R→主群 桥接边 = {len(_bridge)}  (必须 >0, 否则 (b)(c) 面板无意义)")
assert len(_rin) > 0 and len(_bridge) > 0, "合成几何未产生预期的图结构，示意图会误导"

fig.suptitle("Weighted-union clustering with consensus-block relabeling — method schematic",
             fontsize=13, fontweight="bold", y=0.985)
fig.text(0.5, 0.012, "Synthetic toy data for illustration only; no experimental results shown.",
         ha="center", fontsize=8.2, color="0.45", style="italic")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("saved", OUT)
