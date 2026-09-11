"""Marker verification (#27): confirm the rescued SC / DRG bins actually express
their known marker genes.

Design (honest):
  - Target cells are the GOLD-STANDARD annotation bins (Spinal cord=60, DRG=131),
    i.e. the classes the method rescues. We verify the rescued regions are
    biologically Spinal cord / DRG by their marker expression, not by the
    clustering itself (the clustering's SC/DRG overlap already matches these bins).
  - Expression is taken from the ORIGINAL data (E11.5_E1S3.MOSTA.h5ad): X is
    log1p-normalized; layers['count'] holds raw counts. Slim cells are aligned to
    original cells via exact spatial-coordinate match (27378/27378).
  - Metrics per marker: mean log1p in target vs background (all other bins),
    fold-change (count-ratio, log1p-safe), %cells expressing (X>0).
Output: results/method2_weighted_union_final/tables/marker_verify.csv
"""
import numpy as np, pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree

SC_CLASS = "Spinal cord"
DRG_CLASS = "Dorsal root ganglion"

SC_MARKERS = ['Olig2', 'Pax6', 'Mnx1', 'Isl1', 'Nkx2.2', 'Sox2', 'Neurog1',
              'Neurod1', 'Tubb3', 'Nefm', 'Nkx6-1', 'Olig1']
DRG_MARKERS = ['Sox10', 'Prph', 'Ntrk1', 'Ngfr', 'Pmp2', 'S100b', 'Pou4f1',
               'Isl1', 'Tubb3', 'Ntrk2', 'Ntrk3', 'Nefl', 'Ret', 'Pax2', 'Runx1', 'Etv1']
NEG_MARKERS = ['Krt14', 'Col2a1', 'Myh11', 'Cyp2e1', 'Prss28', 'Alb', 'Lyz2']  # non-neural negatives (surface ectoderm/cartilage/smooth muscle/liver/pancreas/blood)

# --- load + align ---------------------------------------------------------
o = sc.read_h5ad('F:/BGI/task3/spateo-release-main/data/E11.5_E1S3.MOSTA.h5ad')
s = sc.read_h5ad('F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = s.obs['annotation'].astype(str).values

T = cKDTree(np.round(o.obsm['spatial'], 3))
_, idx = T.query(np.round(s.obsm['spatial'], 3))
assert int((np.abs(o.obsm['spatial'][idx] - s.obsm['spatial']).sum(1) < 1e-2).sum()) == len(s)

X = o.X[idx]                      # log1p-normalized, slim order (27378 x 25741)
C = o.layers['count'][idx]        # raw counts, same order
var = o.var_names

def mean_log1p(sub, gene):
    g = np.where(var == gene)[0]
    if len(g) == 0:
        return np.nan
    return float(np.asarray(X[sub, g[0]].todense()).mean())

def mean_count(sub, gene):
    g = np.where(var == gene)[0]
    if len(g) == 0:
        return np.nan
    return float(np.asarray(C[sub, g[0]].todense()).mean())

def pct_expr(sub, gene):
    g = np.where(var == gene)[0]
    if len(g) == 0:
        return np.nan
    return float((np.asarray(X[sub, g[0]].todense()).ravel() > 0).mean())

sc_mask = ann == SC_CLASS
drg_mask = ann == DRG_CLASS
bg_mask = ~(sc_mask | drg_mask)
print('SC bins:', int(sc_mask.sum()), ' DRG bins:', int(drg_mask.sum()), ' bg:', int(bg_mask.sum()))

rows = []
for g in SC_MARKERS + DRG_MARKERS + NEG_MARKERS:
    if g not in var:
        rows.append({'gene': g, 'class': 'SC' if g in SC_MARKERS else ('DRG' if g in DRG_MARKERS else 'negative'),
                     'present': False}); continue
    mS, mD, mB = mean_log1p(sc_mask, g), mean_log1p(drg_mask, g), mean_log1p(bg_mask, g)
    cS, cD, cB = mean_count(sc_mask, g), mean_count(drg_mask, g), mean_count(bg_mask, g)
    pS, pD, pB = pct_expr(sc_mask, g), pct_expr(drg_mask, g), pct_expr(bg_mask, g)
    rows.append({'gene': g,
                 'class': 'SC' if g in SC_MARKERS else ('DRG' if g in DRG_MARKERS else 'negative'),
                 'present': True,
                 'SC_mean_log1p': round(mS, 3), 'DRG_mean_log1p': round(mD, 3), 'bg_mean_log1p': round(mB, 3),
                 'SC_bg_fc': round((cS + 0.1) / (cB + 0.1), 3),
                 'DRG_bg_fc': round((cD + 0.1) / (cB + 0.1), 3),
                 'SC_pct': round(100 * pS, 1), 'DRG_pct': round(100 * pD, 1), 'bg_pct': round(100 * pB, 1)})
    print('%-8s SC=%.3f DRG=%.3f bg=%.3f | SCpct=%.0f%% DRGpct=%.0f%%' %
          (g, mS, mD, mB, 100 * pS, 100 * pD))

df = pd.DataFrame(rows)
out = 'F:/BGI/task3/spateo-release-main/results/method2_weighted_union_final/tables/marker_verify.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print('\nsaved:', out, ' rows:', len(df))

# --- verdict summary ------------------------------------------------------
sc_ok = df[(df['class'] == 'SC') & df['present']]
drg_ok = df[(df['class'] == 'DRG') & df['present']]
pos = df[(df['class'].isin(['SC', 'DRG'])) & df['present']]
print('\n=== VERDICT ===')
print('SC markers: %d tested, SC_mean > bg_mean in %d/%d' %
      (len(sc_ok), int((sc_ok['SC_mean_log1p'] > sc_ok['bg_mean_log1p']).sum()), len(sc_ok)))
print('DRG markers: %d tested, DRG_mean > bg_mean in %d/%d' %
      (len(drg_ok), int((drg_ok['DRG_mean_log1p'] > drg_ok['bg_mean_log1p']).sum()), len(drg_ok)))
# specificity: markers for SC should be HIGHER in SC than DRG, etc.
print('SC-specific (SC>bg & SC>DRG): %d/%d' % (
    int(((sc_ok['SC_mean_log1p'] > sc_ok['bg_mean_log1p']) & (sc_ok['SC_mean_log1p'] > sc_ok['DRG_mean_log1p'])).sum()), len(sc_ok)))
print('DRG-specific (DRG>bg & DRG>SC): %d/%d' % (
    int(((drg_ok['DRG_mean_log1p'] > drg_ok['bg_mean_log1p']) & (drg_ok['DRG_mean_log1p'] > drg_ok['SC_mean_log1p'])).sum()), len(drg_ok)))
