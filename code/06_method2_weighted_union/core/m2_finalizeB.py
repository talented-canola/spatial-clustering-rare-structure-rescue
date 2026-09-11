import numpy as np, os, scanpy as sc, pandas as pd, json
a = sc.read_h5ad('F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
os.makedirs('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/data', exist_ok=True)
for res in [0.5, 1.0, 1.5, 2.0]:
    lab = np.load('F:/tmp/m2_final_corrected_alpha1.5_res'+'{:g}'.format(res)+'_seed888.npy')
    a.obs['method2_weighted_union_final_r'+str(res)] = pd.Categorical(lab.astype(str))
X_out = a.X[:0].copy() if a.X is not None else None
slim = sc.AnnData(X=X_out, obs=a.obs[['annotation','method2_weighted_union_final_r0.5','method2_weighted_union_final_r1.0','method2_weighted_union_final_r1.5','method2_weighted_union_final_r2.0']].copy(),
                  var=a.var.copy())
slim.obsm['spatial'] = a.obsm['spatial'].copy()
slim.obsm['X_pca'] = a.obsm['X_pca'].copy()
slim.write('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/data/method2_weighted_union_final_slim.h5ad')
# registry update
summ = json.load(open('F:/tmp/m2_final_summary.json'))
per_res = {float(r['res']): r for r in summ['per_res']}
reg = 'F:/BGI/task3/spateo-release-main/results/experiment_registry.csv'
df = pd.read_csv(reg, keep_default_na=False, encoding='utf-8-sig')
df['resolution'] = df['resolution'].astype(str).str.strip()
rows = []
for res in [0.5, 1.0, 1.5, 2.0]:
    m = per_res[res]
    rows.append({
        '方案ID': 'method2_weighted_union_final', '聚类算法': 'Leiden(加权)',
        'PCA求解器': 'arpack', 'PCA维度': '30', 'e_neigh': '30', 's_neigh': '8',
        'resolution': str(res), '是否用了spatial_adj': '是(加权并集+共识块内W_e放大+SC确定性拆分)',
        '整体ARI': str(m['ari_mean']), '整体NMI': str(m['nmi_mean']), 'Silhouette': str(m['sil_mean']), '空间连贯性': str(m['coh_mean']),
        '完整19类重叠表路径': 'method2_weighted_union_final/tables/matching_r'+str(res)+'.csv',
        '空间图路径': 'method2_weighted_union_final/figures/spatial_aligned_r'+str(res)+'.png',
        '混淆矩阵路径': 'method2_weighted_union_final/figures/confusion_r'+str(res)+'.png',
        'UMAP路径': 'method2_weighted_union_final/figures/umap.png',
        'marker基因验证路径': '缺失', '状态': '已完成',
    })
out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
out.to_csv(reg, index=False, encoding='utf-8-sig')
print('finalizeB done')
