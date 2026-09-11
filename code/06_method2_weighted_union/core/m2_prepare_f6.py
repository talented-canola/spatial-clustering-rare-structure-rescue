"""Compute the F6 consensus protection set (方案A: full 6-seed consensus, block size in [50,200])
from the saved marking labels, save to m2_f6_protect.npz, and smoke-test the shared lib."""
import sys
sys.path.insert(0, "F:/tmp")
import numpy as np
from collections import defaultdict
import scanpy as sc

SEEDS = [888, 1, 2, 3, 4, 42]
SLIM = "F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad"
a = sc.read_h5ad(SLIM)
ann = a.obs["annotation"].astype(str).values
n = len(ann)

L = np.stack([np.load("F:/tmp/m2_mark_r1_seed%d.npy" % s) for s in SEEDS], axis=1)
g = defaultdict(list)
for i in range(n):
    g[tuple(int(x) for x in L[i])].append(i)

blocks = np.zeros(n, dtype=np.int32)
bid = 0
detail = []
for k, cells in g.items():
    if 50 <= len(cells) <= 200:
        bid += 1
        blocks[cells] = bid
        # dominant annotation of the block
        c2 = defaultdict(int)
        for i in cells:
            c2[ann[i]] += 1
        dom, cnt = max(c2.items(), key=lambda kv: kv[1])
        detail.append((len(cells), dom, cnt, len(c2)))

F = blocks > 0
sc_mask = ann == "Spinal cord"
drg_mask = ann == "Dorsal root ganglion"
sc_rec = int((F & sc_mask).sum()) / float(sc_mask.sum())
drg_rec = int((F & drg_mask).sum()) / float(drg_mask.sum())
print("F size = %d  blocks = %d" % (int(F.sum()), bid))
print("SC recall in F = %.4f   DRG recall in F = %.4f" % (sc_rec, drg_rec))
detail.sort(reverse=True)
print("top 15 blocks (size, dominant annotation, dom_count, n_classes):")
for sz, dom, cnt, nc in detail[:15]:
    print("   size=%4d dom=%-20s dom_count=%4d n_classes=%d" % (sz, dom, cnt, nc))
np.savez("F:/tmp/m2_f6_protect.npz", blocks=blocks)
print("saved m2_f6_protect.npz")
