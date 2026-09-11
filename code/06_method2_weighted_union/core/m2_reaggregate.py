"""Re-aggregate the annotation-free (71-bin split) 20 runs with the SAME
conventions the prior workflow used: sample std (ddof=1), drg_5of5 = count
of seeds with DRG overlap > 0.9, n_split = count of triggered splits.
Recomputes metrics from the saved corrected labels, rewrites verify CSV +
summary JSON + overall_metrics.csv. Adds sc_overlap_pre / drg_overlap_pre /
sc_own_pre per row so the "graph weighting alone never rescues SC" claim can
be stated per-resolution honestly.
"""
import numpy as np, scanpy as sc, hashlib, json
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import sys; sys.path.insert(0, 'F:/tmp')
import m2_scan_lib as lib
import m2_final_lib as F

RES = [0.5, 1.0, 1.5, 2.0]
SEEDS = [888, 1, 2, 3, 4]
SC_BLOCK = np.load('F:/tmp/m2_sc_block.npy')
expr, spat, blocks, a, ann_s, ann, sub = F.load_inputs()
cats = list(ann_s.cat.categories)
res_suf = {0.5: '0.5', 1.0: '1', 1.5: '1.5', 2.0: '2'}
adj = F.weighted_adj(expr, spat, blocks, alpha=1.5, ws=0.2)

rows = []
per_res = {r: [] for r in RES}
for res in RES:
    for seed in SEEDS:
        lab = np.load('F:/tmp/m2_final_corrected_alpha1.5_res%s_seed%d.npy' % (res_suf[res], seed))
        # pre-correction label: recompute Leiden (deterministic) to get sc_own_pre / pre overlaps
        pre = F.leiden_w(adj, res, seed)
        c = pre[SC_BLOCK[0]]
        own_pre = bool((pre[SC_BLOCK] == c).all()) and int((pre == c).sum()) == len(SC_BLOCK)
        trig = not own_pre
        ari_pre = round(float(adjusted_rand_score(ann, pre)), 4)
        ari = round(float(adjusted_rand_score(ann, lab)), 4)
        nmi = round(float(normalized_mutual_info_score(ann, lab)), 4)
        sil = round(float(silhouette_score(a.obsm['X_pca'][sub], lab[sub])), 4)
        coh = round(lib.spatial_coherence(lab, a.obsm['spatial']), 4)
        ov = lib.hungarian_overlaps(ann_s, lab, cats)
        ov_pre = lib.hungarian_overlaps(ann_s, pre, cats)
        sc_ov = ov['Spinal cord']; drg_ov = ov['Dorsal root ganglion']
        sc_pre = ov_pre['Spinal cord']; drg_pre = ov_pre['Dorsal root ganglion']
        sha = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]
        row = {'alpha': 1.5, 'res': res, 'seed': seed,
               'n_clusters_pre': int(pre.max()) + 1, 'n_clusters_post': int(lab.max()) + 1,
               'sc_split_triggered': trig, 'sc_own_pre': own_pre,
               'sc_overlap_pre': sc_pre, 'drg_overlap_pre': drg_pre,
               'sc_overlap_post': sc_ov, 'drg_overlap_post': drg_ov,
               'ari': ari, 'ari_pre': ari_pre, 'nmi': nmi, 'silhouette': sil, 'coherence': coh,
               'label_sha256': sha}
        for c in cats:
            row['ov_' + c] = round(ov[c], 4)
        rows.append(row)
        per_res[res].append({'ari': ari, 'ari_pre': ari_pre, 'nmi': nmi, 'sil': sil, 'coh': coh,
                             'sc': sc_ov, 'drg': drg_ov, 'sc_pre': sc_pre, 'trig': trig})
        print('res=%s seed=%d ari=%s ari_pre=%s SCpost=%s SCpre=%s DRGpost=%s trig=%s' %
              (res, seed, ari, ari_pre, sc_ov, sc_pre, drg_ov, trig), flush=True)

df = pd.DataFrame(rows)
df.to_csv('F:/BGI/task3/spateo-release-main/results/analysis/method2_final_verify.csv',
          index=False, encoding='utf-8-sig')


def stats(v):
    return round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)

summary = {'chosen_alpha': 1.5, 'accepted': True, 'per_res': []}
for r in RES:
    d = per_res[r]
    a = np.array([x['ari'] for x in d]); nm = np.array([x['nmi'] for x in d])
    si = np.array([x['sil'] for x in d]); co = np.array([x['coh'] for x in d])
    sc = np.array([x['sc'] for x in d]); drg = np.array([x['drg'] for x in d])
    ap = np.array([x['ari_pre'] for x in d]); scp = np.array([x['sc_pre'] for x in d])
    am, asd = stats(a); apm, apsd = stats(ap); nm_, nsd = stats(nm)
    sm, ssd = stats(si); cm, csd = stats(co); scm, scsd = stats(sc); dm, dsd = stats(drg)
    summary['per_res'].append({
        'res': r,
        'ari_mean': am, 'ari_std': asd,
        'ari_pre_mean': apm, 'ari_pre_std': apsd,
        'nmi_mean': nm_, 'nmi_std': nsd,
        'sil_mean': sm, 'sil_std': ssd,
        'coh_mean': cm, 'coh_std': csd,
        'sc_mean': scm, 'sc_std': scsd,
        'sc_5of5': int((sc == 1.0).sum()),
        'sc_pre_5of5': int((scp == 1.0).sum()),          # how many seeds self-cluster before correction
        'drg_mean': dm, 'drg_std': dsd,
        'drg_5of5': int((drg > 0.9).sum()),
        'n_split': int(sum(1 for x in d if x['trig'])),
    })
json.dump(summary, open('F:/tmp/m2_final_summary.json', 'w'), indent=1)
# overall_metrics.csv
om = [{'res': s['res'], 'ari_mean': s['ari_mean'], 'ari_std': s['ari_std'],
       'nmi_mean': s['nmi_mean'], 'nmi_std': s['nmi_std'],
       'silhouette_mean': s['sil_mean'], 'silhouette_std': s['sil_std'],
       'coherence_mean': s['coh_mean'], 'coherence_std': s['coh_std'],
       'sc_overlap_mean': s['sc_mean'], 'sc_overlap_std': s['sc_std'],
       'drg_overlap_mean': s['drg_mean'], 'drg_overlap_std': s['drg_std']}
      for s in summary['per_res']]
pd.DataFrame(om).to_csv('F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables/overall_metrics.csv',
                        index=False, encoding='utf-8-sig')
print('\n=== SUMMARY (ddof=1) ===')
print(json.dumps(summary, indent=1))
