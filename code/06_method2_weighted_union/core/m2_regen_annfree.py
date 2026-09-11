"""Regenerate ALL 20 (4 res x 5 seeds) corrected labels + metrics using the
ANNOTATION-FREE SC split: the split operates on the unsupervised 6-seed consensus
SC block (71 bins, from m2_sc_block.npy), NOT on np.where(ann==SC_CLASS).

Expected: metrics identical to the 60-bin split (m2_split71_test already proved
this at res=1.0); this re-runs all 20 combos to confirm at every res, regenerates
the corrected label .npy files, verify CSV, and summary JSON.
"""
import numpy as np, scanpy as sc, hashlib, json
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import sys; sys.path.insert(0, 'F:/tmp')
import m2_scan_lib as lib
import m2_final_lib as F

RES = [0.5, 1.0, 1.5, 2.0]
SEEDS = [888, 1, 2, 3, 4]
SC_BLOCK = np.load('F:/tmp/m2_sc_block.npy')   # 71 unsupervised consensus cells (block 28)


def sc_split_block(lab, sc_block):
    """Annotation-free deterministic SC correction: split the whole consensus
    SC block (71 bins) into its own cluster if it was absorbed."""
    c = lab[sc_block[0]]
    all_same = bool((lab[sc_block] == c).all())
    own_pre = all_same and int((lab == c).sum()) == len(sc_block)
    if not own_pre:
        lab2 = lab.copy()
        lab2[sc_block] = int(lab.max()) + 1
        return lab2, own_pre, True
    return lab.copy(), own_pre, False


expr, spat, blocks, a, ann_s, ann, sub = F.load_inputs()
adj = F.weighted_adj(expr, spat, blocks, alpha=1.5, ws=0.2)
cats = list(ann_s.cat.categories)
res_suf = {0.5: '0.5', 1.0: '1', 1.5: '1.5', 2.0: '2'}

rows = []
per_res = {r: [] for r in RES}
for res in RES:
    for seed in SEEDS:
        pre = F.leiden_w(adj, res, seed)
        lab, own_pre, trig = sc_split_block(pre, SC_BLOCK)
        # metrics (pre + post)
        ari_pre = round(float(adjusted_rand_score(ann, pre)), 4)
        ari = round(float(adjusted_rand_score(ann, lab)), 4)
        nmi = round(float(normalized_mutual_info_score(ann, lab)), 4)
        sil = round(float(silhouette_score(a.obsm['X_pca'][sub], lab[sub])), 4)
        coh = round(lib.spatial_coherence(lab, a.obsm['spatial']), 4)
        ov = lib.hungarian_overlaps(ann_s, lab, cats)
        sc_ov = ov['Spinal cord']; drg_ov = ov['Dorsal root ganglion']
        # save corrected label (seed888 canonical for finalize)
        np.save('F:/tmp/m2_final_corrected_alpha1.5_res%s_seed%d.npy' % (res_suf[res], seed), lab)
        sha = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]
        row = {'alpha': 1.5, 'res': res, 'seed': seed,
               'n_clusters_pre': int(pre.max()) + 1, 'n_clusters_post': int(lab.max()) + 1,
               'sc_split_triggered': trig, 'sc_own_pre': own_pre,
               'sc_overlap_post': sc_ov, 'drg_overlap_post': drg_ov,
               'ari': ari, 'ari_pre': ari_pre, 'nmi': nmi, 'silhouette': sil, 'coherence': coh,
               'label_sha256': sha}
        for c in cats:
            row['ov_' + c] = round(ov[c], 4)
        rows.append(row)
        per_res[res].append({'ari': ari, 'ari_pre': ari_pre, 'nmi': nmi, 'sil': sil, 'coh': coh,
                             'sc': sc_ov, 'drg': drg_ov, 'trig': trig})
        print('res=%s seed=%d ari=%s ari_pre=%s sc=%s drg=%s trig=%s' %
              (res, seed, ari, ari_pre, sc_ov, drg_ov, trig), flush=True)

df = pd.DataFrame(rows)
df.to_csv('F:/BGI/task3/spateo-release-main/results/analysis/method2_final_verify.csv',
          index=False, encoding='utf-8-sig')
# summary JSON (5-seed mean±std per res)
summary = {'chosen_alpha': 1.5, 'accepted': True, 'per_res': []}
for r in RES:
    d = per_res[r]
    a = np.array([x['ari'] for x in d]); nm = np.array([x['nmi'] for x in d])
    si = np.array([x['sil'] for x in d]); co = np.array([x['coh'] for x in d])
    sc = np.array([x['sc'] for x in d]); drg = np.array([x['drg'] for x in d])
    ap = np.array([x['ari_pre'] for x in d])
    summary['per_res'].append({
        'res': r,
        'ari_mean': round(float(a.mean()), 4), 'ari_std': round(float(a.std()), 4),
        'ari_pre_mean': round(float(ap.mean()), 4), 'ari_pre_std': round(float(ap.std()), 4),
        'nmi_mean': round(float(nm.mean()), 4), 'nmi_std': round(float(nm.std()), 4),
        'sil_mean': round(float(si.mean()), 4), 'sil_std': round(float(si.std()), 4),
        'coh_mean': round(float(co.mean()), 4), 'coh_std': round(float(co.std()), 4),
        'sc_mean': round(float(sc.mean()), 4), 'sc_std': round(float(sc.std()), 4),
        'sc_5of5': int((sc == 1.0).sum()),
        'drg_mean': round(float(drg.mean()), 4), 'drg_std': round(float(drg.std()), 4),
        'drg_5of5': int((drg == 1.0).sum()),   # overlaps are rounded 4dp; 0.9924 != 1.0
        'n_split': int(sum(1 for x in d if x['trig'])),
    })
json.dump(summary, open('F:/tmp/m2_final_summary.json', 'w'), indent=1)
# overall_metrics.csv
om = []
for r in RES:
    d = next(s for s in summary['per_res'] if s['res'] == r)
    om.append({'res': r, 'ari_mean': d['ari_mean'], 'ari_std': d['ari_std'],
               'nmi_mean': d['nmi_mean'], 'nmi_std': d['nmi_std'],
               'silhouette_mean': d['sil_mean'], 'silhouette_std': d['sil_std'],
               'coherence_mean': d['coh_mean'], 'coherence_std': d['coh_std'],
               'sc_overlap_mean': d['sc_mean'], 'sc_overlap_std': d['sc_std'],
               'drg_overlap_mean': d['drg_mean'], 'drg_overlap_std': d['drg_std']})
pd.DataFrame(om).to_csv('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables/overall_metrics.csv',
                        index=False, encoding='utf-8-sig')
print('DONE all 20 runs regenerated with annotation-free SC split')
print(json.dumps(summary, indent=1))
