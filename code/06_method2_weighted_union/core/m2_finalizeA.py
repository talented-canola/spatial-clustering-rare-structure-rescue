import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os
import scanpy as sc
from scipy.optimize import linear_sum_assignment
import sys; sys.path.insert(0,'F:/tmp')
import m2_final_lib as F
a = sc.read_h5ad('F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
xy = a.obsm['spatial']; ann = a.obs['annotation'].astype(str).values
cats = list(a.obs['annotation'].cat.categories)
os.makedirs('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/figures', exist_ok=True); os.makedirs('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables', exist_ok=True)
ann_col = plt.cm.tab20(np.linspace(0,1,len(cats)))
res_suf = {0.5:'0.5', 1.0:'1', 1.5:'1.5', 2.0:'2'}
for res in [0.5, 1.0, 1.5, 2.0]:
    lab = np.load('F:/tmp/m2_final_corrected_alpha1.5_res'+res_suf[res]+'_seed888.npy')
    # spatial aligned
    fig, axes = plt.subplots(1,2, figsize=(15,6.5))
    for k,c in enumerate(cats):
        m = ann==c
        axes[0].scatter(xy[m,0], xy[m,1], s=1, c=[ann_col[k]])
    axes[0].set_title('annotation (19 classes)'); axes[0].set_aspect('equal')
    ncl = int(lab.max())+1
    lab_col = plt.cm.tab20(np.random.RandomState(0).permutation(ncl) % 20)
    axes[1].scatter(xy[:,0], xy[:,1], s=1, c=lab_col[lab])
    axes[1].set_title('method2_weighted_union_final r=%s' % res); axes[1].set_aspect('equal')
    fig.savefig('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/figures/spatial_aligned_r%s.png' % res, dpi=130, bbox_inches='tight'); plt.close(fig)
    # confusion table + figure
    ct = pd.crosstab(pd.Series(ann), pd.Series(lab))
    ct.to_csv('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables/confusion_r%s.csv' % res, encoding='utf-8-sig')
    norm = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(max(8, ncl*0.35), 8))
    im = ax.imshow(norm.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(ncl)); ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(ct.columns, fontsize=6); ax.set_yticklabels(cats, fontsize=7)
    ax.set_title('confusion (row-normalized) r=%s' % res)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.savefig('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/figures/confusion_r%s.png' % res, dpi=130, bbox_inches='tight'); plt.close(fig)
    # matching (Hungarian-aligned 19-class overlap)
    C = ct.values.astype(float)
    N = max(C.shape); Cpad = np.zeros((N,N)); Cpad[:C.shape[0], :C.shape[1]] = C
    ri, ci = linear_sum_assignment(-Cpad)
    mrows = []
    for i in range(C.shape[0]):
        j = ci[i]
        ov = round(float(C[i,j]/C[i].sum()), 4) if j < C.shape[1] else 0.0
        mrows.append({'class': cats[i], 'overlap': ov, 'matched_cluster': int(ct.columns[j]) if j < C.shape[1] else -1})
    pd.DataFrame(mrows).to_csv('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables/matching_r%s.csv' % res, index=False, encoding='utf-8-sig')
# UMAP on the scheme graph (res=1.0 labels)
lab1 = np.load('F:/tmp/m2_final_corrected_alpha1.5_res1_seed888.npy')
expr, spat, blocks, _, _, _, _ = F.load_inputs()
adj = F.weighted_adj(expr, spat, blocks, alpha=1.5, ws=0.2)
adj.data[:] = 1.0
ua = a.copy()
ua.obsp['connectivities'] = adj.tocsr()
ua.uns['neighbors'] = {'connectivities_key':'connectivities','distances_key':'distances','n_neighbors':30,'params':{'n_neighbors':30,'method':'umap'}}
sc.tl.umap(ua)
emb = ua.obsm['X_umap']
fig, axes = plt.subplots(1,2, figsize=(15,6.5))
for k,c in enumerate(cats):
    m = ann==c
    axes[0].scatter(emb[m,0], emb[m,1], s=1, c=[ann_col[k]])
axes[0].set_title('annotation');
ncl = int(lab1.max())+1
lab_col1 = plt.cm.tab20(np.random.RandomState(0).permutation(ncl) % 20)
axes[1].scatter(emb[:,0], emb[:,1], s=1, c=lab_col1[lab1])
axes[1].set_title('method2_weighted_union_final r=1.0')
fig.savefig('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/figures/umap.png', dpi=130, bbox_inches='tight'); plt.close(fig)
print('finalizeA done')
