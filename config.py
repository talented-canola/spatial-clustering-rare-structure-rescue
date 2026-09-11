"""路径配置 —— 复现本工作时唯一需要修改的文件。

原始脚本中的路径均为作者本机的绝对路径。若要在别的机器上复现，只需：

  1. 修改下面 4 个变量（WORK_DIR / PROJECT_ROOT / DATA_H5AD / RESULTS_DIR）；
  2. 运行 `python port_paths.py` 一键把 code/ 下所有脚本里的硬编码路径改写到新位置；
  3. 按 README 的顺序跑各流程。

路径分布（本仓库实测，供参考）：
  F:/tmp/...                                  141 处  →  WORK_DIR
  F:/BGI/task3/spateo-release-main/results/    97 处  →  RESULTS_DIR
  F:/BGI/task3/spateo-release-main/data/...    22 处  →  DATA_H5AD
"""

import os

# ============ 需要修改的 4 个变量 ============

# 输入数据与结果所在的工程目录（原始下载的 Spateo 包所在处）
PROJECT_ROOT = "F:/BGI/task3/spateo-release-main"

# 中间产物工作目录（.npz 二值图、逐 seed 的标签 .npy 等都落在这里）
WORK_DIR = "F:/tmp"

# 原始数据（需自行下载，见 DATA.md）
DATA_H5AD = PROJECT_ROOT + "/data/E11.5_E1S3.MOSTA.h5ad"

# 结果归档目录（本仓库不含其内容，见 README「结果结构」）
RESULTS_DIR = PROJECT_ROOT + "/results"

# ============ 以下由上面推导，通常无需修改 ============

# 基线产出的精简 AnnData：含 obs["annotation"] / obsm["spatial"] / obsm["X_pca"]
SLIM_H5AD = RESULTS_DIR + "/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"

# method2 的中间产物（由 core/m2_prepare.py 与 core/m2_prepare_f6.py 生成）
EXPR_BIN = WORK_DIR + "/m2_expr_bin.npz"        # 表达空间 KNN 二值图 E
SPAT_BIN = WORK_DIR + "/m2_spatial_bin_s8.npz"  # 物理空间 KNN 二值图 S
SUBSAMPLE = WORK_DIR + "/m2_subsample.npy"      # Silhouette 用的 4000 节点子采样
PROTECT_BLOCKS = WORK_DIR + "/m2_f6_protect.npz"  # 无标注共识保护块 B

# method2 核心超参数（见 README「最终方法」）
ALPHA = 1.5   # 块内表达边放大系数
W_S = 0.2     # 空间边权重
E_NEIGH = 30  # 表达空间 KNN 邻居数
S_NEIGH = 8   # 物理空间 KNN 邻居数
RESOLUTIONS = [0.5, 1.0, 1.5, 2.0]
SEEDS = [888, 1, 2, 3, 4]


def describe():
    """打印当前路径配置，便于确认改对了。"""
    for k in ("PROJECT_ROOT", "WORK_DIR", "DATA_H5AD", "RESULTS_DIR", "SLIM_H5AD"):
        v = globals()[k]
        print(f"{k:14s} = {v}   [{'存在' if os.path.exists(v) else '不存在'}]")


if __name__ == "__main__":
    describe()
