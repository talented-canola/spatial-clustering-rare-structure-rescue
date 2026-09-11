# 数据说明

**本仓库不包含任何数据文件。** 本文档说明数据来源与复现所需的中间产物。

## 1. 原始数据

本工作使用 **MOSTA（Mouse Organogenesis Spatiotemporal Transcriptomic Atlas）** 的
**E11.5 小鼠胚胎** Stereo-seq 切片，文件名为 `E11.5_E1S3.MOSTA.h5ad`（E1S3 = Embryo 1, Section 3）。

| 项目 | 值 |
|---|---|
| 物种 | *Mus musculus* (mm10), C57BL/6 |
| 技术 | Stereo-seq (DNA nanoball-patterned arrays) |
| 发育阶段 | E11.5 |
| 规模 | 27,378 个空间 bin |
| 金标准标注 | 19 类解剖学标注（`obs["annotation"]`，由原作者提供） |

### 下载入口

- **MOSTA 官方门户**：<https://db.cngb.org/stomics/mosta/>
- **STOmicsDB 数据集页**（数据集 ID `STDS0000058`）：<https://db.cngb.org/stomics/datasets/STDS0000058>
- **CNGB 原始数据**（项目号 `CNP0001543`）：<https://db.cngb.org/search/project/CNP0001543>

### 引用

Chen, A., Liao, S., Cheng, M., et al. (2022).
*Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays.*
**Cell** 185(10), 1777–1792.e21. DOI: [10.1016/j.cell.2022.04.003](https://doi.org/10.1016/j.cell.2022.04.003)

## 2. 数据放置位置

脚本中使用了作者本机的绝对路径，复现时需对应调整。默认约定为：

```
<该项目根目录>/
├── data/
│   └── E11.5_E1S3.MOSTA.h5ad      # 原始数据，需自行下载
└── results/
    └── baseline_v2_arpack/data/
        └── baseline_corrected_slim.h5ad    # 由基线流程产出的精简文件
```

要求 `E11.5_E1S3.MOSTA.h5ad` 至少包含：

- `obs["annotation"]` —— 19 类解剖学标注（金标准，仅用于**事后评估**）
- `obsm["spatial"]` —— 空间坐标
- `.X` —— log1p 归一化表达；原始计数在 `layers["count"]`

## 3. 复现所需的中间产物

以下文件由脚本生成，**未包含在仓库中**（均为 `.npz`/`.npy`，被 `.gitignore` 屏蔽）。
按顺序运行即可重建：

| 中间文件 | 生成脚本 | 内容 |
|---|---|---|
| `m2_expr_bin.npz` | `core/m2_prepare.py` | 表达空间 KNN 二值图 `E`（PCA-30, e_neigh=30） |
| `m2_spatial_bin_s8.npz` | `core/m2_prepare.py` | 物理空间 KNN 二值图 `S`（s_neigh=8） |
| `m2_subsample.npy` | `core/m2_prepare.py` | 4000 节点子采样索引（Silhouette 用） |
| `m2_f6_protect.npz` | `core/m2_prepare_f6.py` | 无标注共识保护块 `B`（块大小 ∈ [50,200]） |

> `m2_prepare.py` 顶部注释记录了与 `scc_s8` 并集图（915,412 条边）的一致性校验，
> 复现时可用于确认构图正确。

## 4. 评估用标注的使用边界

金标准标注 `obs["annotation"]` **只出现在两处**：事后指标计算（ARI / NMI / 匈牙利重叠）
与 marker 基因验证。核心算法 `m2_final_lib.py` 中的 `sc_split_block()` **不读取任何标注**，
脊髓的拆分依据是纯无监督的 6-seed 共识块（`compute_marking.py` 产出）。
