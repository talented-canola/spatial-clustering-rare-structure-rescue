"""Diagnose whether the SC/DRG cores are robust to seed-set choice (fair test: same-size
5-seed consensuses A={888,1,2,3,4} vs B={1,2,3,4,42}, plus 6-seed consensus)."""
import sys
sys.path.insert(0, 'F:/tmp')
import numpy as np, scanpy as sc, json
from collections import defaultdict, Counter

SLIM = 'F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad'
a = sc.read_h5ad(SLIM)
ann = a.obs['annotation'].astype(str).values
n = len(ann)

def groups_for(seeds):
    L = np.stack([np.load('F:/tmp/m2_mark_r1_seed%d.npy' % s) for s in seeds], axis=1)
    g = defaultdict(list)
    for i in range(n):
        g[tuple(int(x) for x in L[i])].append(i)
    return g

def sc_drg_block(g):
    out = {}
    for name, mask in [('SC', ann == 'Spinal cord'), ('DRG', ann == 'Dorsal root ganglion')]:
        cnt = {}
        for k, v in g.items():
            c = int(sum(1 for i in v if mask[i]))
            if c > 0:
                cnt[k] = c
        kk = max(cnt, key=cnt.get)
        out[name] = (len(g[kk]), cnt[kk] / float(mask.sum()), set(g[kk]))
    return out

def F_for(g, lo=50, hi=200):
    return set().union(*[set(v) for k, v in g.items() if lo <= len(v) <= hi])

def jac(a, b):
    return len(a & b) / max(1, len(a | b))

gA = groups_for([888, 1, 2, 3, 4])
gB = groups_for([1, 2, 3, 4, 42])
g6 = groups_for([888, 1, 2, 3, 4, 42])

FA, FB, F6 = F_for(gA), F_for(gB), F_for(g6)
iA, iB, i6 = sc_drg_block(gA), sc_drg_block(gB), sc_drg_block(g6)

print('F sizes: A=%d B=%d 6=%d' % (len(FA), len(FB), len(F6)))
print('Jaccard FA vs FB =', round(jac(FA, FB), 4))
print('Jaccard FA vs F6 =', round(jac(FA, F6), 4))
print()
for nm in ['SC', 'DRG']:
    print('%s: block size A/B/6 = %d/%d/%d  recall A/B/6 = %.4f/%.4f/%.4f'
          % (nm, iA[nm][0], iB[nm][0], i6[nm][0], iA[nm][1], iB[nm][1], i6[nm][1]))
    print('   SC/DRG-block Jaccard A-vs-B =', round(jac(iA[nm][2], iB[nm][2]), 4),
          ' A-vs-6 =', round(jac(iA[nm][2], i6[nm][2]), 4))
print()
for nm in ['SC', 'DRG']:
    mask = ann == ('Spinal cord' if nm == 'SC' else 'Dorsal root ganglion')
    for lbl, F in [('A', FA), ('B', FB), ('6', F6)]:
        r = int(sum(1 for i in F if mask[i])) / float(mask.sum())
        print('%s recall in F%s = %.4f' % (nm, lbl, r))
print()
print('FA-only bins = %d, FB-only bins = %d (union %d)' % (len(FA - FB), len(FB - FA), len(FA | FB)))
print()
# dominant annotation of the protected blocks in A and B (to see what shifted)
for lbl, g in [('A', gA), ('B', gB)]:
    prot = sorted(((len(v), k) for k, v in g.items() if 50 <= len(v) <= 200), reverse=True)
    print('--- protected blocks in consensus', lbl)
    for sz, k in prot:
        c2 = Counter(ann[list(g[k])])
        dom, c = c2.most_common(1)[0]
        print('   size=%3d dom=%-20s dom_count=%3d n_classes=%d' % (sz, str(dom), c, len(c2)))
