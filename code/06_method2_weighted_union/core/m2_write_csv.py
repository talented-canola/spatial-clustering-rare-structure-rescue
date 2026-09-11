import sys; sys.path.insert(0,'F:/tmp')
import numpy as np, scanpy as sc, pandas as pd
from collections import defaultdict, Counter

a = sc.read_h5ad(r'F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = a.obs['annotation'].astype(str).values
n = len(ann)
SEEDS = [888,1,2,3,4]
L = np.stack([np.load('F:/tmp/m2_mark_r1_seed%d.npy' % s) for s in SEEDS], axis=1)
groups = defaultdict(list)
for i in range(n):
    groups[tuple(int(x) for x in L[i])].append(i)

lower, upper = 50, 200
prot = sorted(((len(v), k) for k, v in groups.items() if lower <= len(v) <= upper), reverse=True)
detail = []
for sz, k in prot:
    cnt = Counter(ann[list(groups[k])])
    dom, c = cnt.most_common(1)[0]
    detail.append({"size": int(sz), "dominant_annotation": str(dom), "dom_count": int(c), "n_classes": int(len(cnt))})

sizes = sorted(len(v) for v in groups.values())
buckets = {'1-9': 0, '10-49': 0, '50-199': 0, '200-499': 0, '500-1999': 0, '2000+': 0}
for s in sizes:
    if s <= 9: buckets['1-9'] += 1
    elif s <= 49: buckets['10-49'] += 1
    elif s <= 199: buckets['50-199'] += 1
    elif s <= 499: buckets['200-499'] += 1
    elif s <= 1999: buckets['500-1999'] += 1
    else: buckets['2000+'] += 1

F = sorted(i for k, v in groups.items() if lower <= len(v) <= upper for i in v)
F_set = set(F)
L6 = np.stack([np.load('F:/tmp/m2_mark_r1_seed%d.npy' % s) for s in [888,1,2,3,4,42]], axis=1)
g6 = defaultdict(list)
for i in range(n): g6[tuple(int(x) for x in L6[i])].append(i)
F6 = set(i for k, v in g6.items() if lower <= len(v) <= upper for i in v)
js = round(len(F_set & F6) / max(1, len(F_set | F6)), 4)
sc_rec_F = round(int(sum(1 for i in F if ann[i] == 'Spinal cord')) / float((ann == 'Spinal cord').sum()), 4)
drg_rec_F = round(int(sum(1 for i in F if ann[i] == 'Dorsal root ganglion')) / float((ann == 'Dorsal root ganglion').sum()), 4)

def block_for(mask, groups):
    cnt = {}
    for k, v in groups.items():
        c = int(sum(1 for i in v if mask[i]))
        if c > 0: cnt[k] = c
    kk = max(cnt, key=cnt.get)
    return len(groups[kk]), cnt[kk] / float(mask.sum())
sc_size, sc_rec = block_for(ann == 'Spinal cord', groups)
drg_size, drg_rec = block_for(ann == 'Dorsal root ganglion', groups)

# Build CSV: section 1 (bucket distribution), blank, section 2 (metrics), blank, section 3 (protected blocks)
lines = []
lines.append("# Section 1: consensus-block size distribution (5-seed, n=%d)" % len(groups))
lines.append("bucket,count")
for k in ['1-9', '10-49', '50-199', '200-499', '500-1999', '2000+']:
    lines.append("%s,%d" % (k, buckets[k]))
lines.append("")
lines.append("# Section 2: protection range & recall")
lines.append("metric,value")
for k, v in [
    ("chosen_lower", lower), ("chosen_upper", upper), ("n_blocks", len(groups)),
    ("F_size", len(F)), ("sc_block_size", sc_size), ("sc_block_recall", sc_rec),
    ("drg_block_size", drg_size), ("drg_block_recall", drg_rec), ("jaccard_f5_f6", js),
    ("sc_recall_in_F", sc_rec_F), ("drg_recall_in_F", drg_rec_F),
]:
    lines.append("%s,%s" % (k, v))
lines.append("")
lines.append("# Section 3: protected blocks in [50,200] (size, dominant_annotation, dom_count, n_classes)")
lines.append("size,dominant_annotation,dom_count,n_classes")
for d in detail:
    lines.append("%d,%s,%d,%d" % (d["size"], d["dominant_annotation"], d["dom_count"], d["n_classes"]))

with open(r'F:/BGI/task3/spateo-release-main/results/analysis/method2_consensus_marking.csv', 'w', encoding='utf-8-sig', newline='') as f:
    f.write("\n".join(lines) + "\n")
print("WROTE", len(lines), "lines")
