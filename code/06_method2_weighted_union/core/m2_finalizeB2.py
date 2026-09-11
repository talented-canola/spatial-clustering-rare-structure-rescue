"""Regenerate slim h5ad (with annotation-free corrected labels) and update the
registry rows for method2_weighted_union_final IN PLACE (rows 45-48), including
the marker-verification path. Does NOT append new rows.
"""
import numpy as np, os, scanpy as sc, pandas as pd, json

slim_in = 'F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad'
out_dir = 'F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/data'
os.makedirs(out_dir, exist_ok=True)
a = sc.read_h5ad(slim_in)
for res in [0.5, 1.0, 1.5, 2.0]:
    lab = np.load('F:/tmp/m2_final_corrected_alpha1.5_res' + '{:g}'.format(res) + '_seed888.npy')
    a.obs['method2_weighted_union_final_r' + str(res)] = pd.Categorical(lab.astype(str))
X_out = a.X[:0].copy() if a.X is not None else None
slim = sc.AnnData(X=X_out, obs=a.obs[['annotation', 'method2_weighted_union_final_r0.5',
                                      'method2_weighted_union_final_r1.0',
                                      'method2_weighted_union_final_r1.5',
                                      'method2_weighted_union_final_r2.0']].copy(),
                  var=a.var.copy())
slim.obsm['spatial'] = a.obsm['spatial'].copy()
slim.obsm['X_pca'] = a.obsm['X_pca'].copy()
slim.write(out_dir + '/method2_weighted_union_final_slim.h5ad')

# --- registry update IN PLACE -------------------------------------------
summ = json.load(open('F:/tmp/m2_final_summary.json'))
per_res = {float(r['res']): r for r in summ['per_res']}
reg = 'F:/BGI/task3/spateo-release-main/results/experiment_registry.csv'
df = pd.read_csv(reg, keep_default_na=False, encoding='utf-8-sig')
df['resolution'] = df['resolution'].astype(str).str.strip()
mask = (df['方案ID'] == 'method2_weighted_union_final')
if mask.sum() != 4:
    print('WARNING: expected 4 rows, found', mask.sum())
for res in [0.5, 1.0, 1.5, 2.0]:
    m = per_res[res]
    idx = df.index[mask & (df['resolution'] == str(res))][0]
    df.at[idx, '整体ARI'] = str(m['ari_mean'])
    df.at[idx, '整体NMI'] = str(m['nmi_mean'])
    df.at[idx, 'Silhouette'] = str(m['sil_mean'])
    df.at[idx, '空间连贯性'] = str(m['coh_mean'])
    df.at[idx, 'marker基因验证路径'] = 'method2_weighted_union_final/tables/marker_verify.csv'
    df.at[idx, '状态'] = '已完成'
df.to_csv(reg, index=False, encoding='utf-8-sig')
print('slim + registry updated in place (marker path set). rows now:', len(df))
