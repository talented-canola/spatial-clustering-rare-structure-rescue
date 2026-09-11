import sys; sys.path.insert(0,'F:/tmp')
import numpy as np, scanpy as sc, json
from collections import defaultdict, Counter

a = sc.read_h5ad(r'F:/BGI/task3/spateo-release-main/results/baseline_v2_arpack/data/baseline_corrected_slim.h5ad')
ann = a.obs['annotation'].astype(str).values
n = len(ann)
SEEDS = [888,1,2,3,4]
L = np.stack([np.load('F:/tmp/m2_mark_r1_seed%d.npy' % s) for s in SEEDS], axis=1)  # (n,5)
groups = defaultdict(list)
for i in range(n):
    groups[tuple(int(x) for x in L[i])].append(i)
sizes = sorted(len(v) for v in groups.values())
buckets = {'1-9': 0, '10-49': 0, '50-199': 0, '200-499': 0, '500-1999': 0, '2000+': 0}
for s in sizes:
    if s <= 9: buckets['1-9'] += 1
    elif s <= 49: buckets['10-49'] += 1
    elif s <= 199: buckets['50-199'] += 1
    elif s <= 499: buckets['200-499'] += 1
    elif s <= 1999: buckets['500-1999'] += 1
    else: buckets['2000+'] += 1

def block_for(mask, groups):
    cnt = {}
    for k, v in groups.items():
        c = int(sum(1 for i in v if mask[i]))
        if c > 0: cnt[k] = c
    kk = max(cnt, key=cnt.get)
    return len(groups[kk]), cnt[kk] / float(mask.sum())

sc_size, sc_rec = block_for(ann == 'Spinal cord', groups)
drg_size, drg_rec = block_for(ann == 'Dorsal root ganglion', groups)

# protection range: [50, 200] — SC(71) and DRG(154) both inside; EXCLUDES the 21 large
# 500-1999-bin expression blocks that would otherwise blow F up to ~95% of cells
lower, upper = 50, 200
F = sorted(i for k, v in groups.items() if lower <= len(v) <= upper for i in v)
np.save('F:/tmp/m2_protect_bins.npy', np.array(F, dtype=np.int64))

# what are we protecting? size + dominant annotation per protected block (diagnostic only)
prot = sorted(((len(v), k) for k, v in groups.items() if lower <= len(v) <= upper), reverse=True)
detail = []
for sz, k in prot:
    cnt = Counter(ann[list(groups[k])])
    dom, c = cnt.most_common(1)[0]
    detail.append({"size": int(sz), "dom": str(dom), "dom_count": int(c), "n_classes": int(len(cnt))})

# robustness: add seed 42 -> 6-tuple consensus, same range
L6 = np.stack([np.load('F:/tmp/m2_mark_r1_seed%d.npy' % s) for s in [888,1,2,3,4,42]], axis=1)
g6 = defaultdict(list)
for i in range(n): g6[tuple(int(x) for x in L6[i])].append(i)
F6 = sorted(i for k, v in g6.items() if lower <= len(v) <= upper for i in v)
js = round(len(set(F) & set(F6)) / max(1, len(set(F) | set(F6))), 4)
sc_rec_F = round(int(sum(1 for i in F if ann[i] == 'Spinal cord')) / float((ann == 'Spinal cord').sum()), 4)
drg_rec_F = round(int(sum(1 for i in F if ann[i] == 'Dorsal root ganglion')) / float((ann == 'Dorsal root ganglion').sum()), 4)

out = {"chosen_lower": lower, "chosen_upper": upper, "n_blocks": len(groups),
       "block_distribution": buckets, "sc_block_size": sc_size, "sc_block_recall": sc_rec,
       "drg_block_size": drg_size, "drg_block_recall": drg_rec,
       "F_size": len(F), "jaccard_f5_f6": js, "sc_recall_in_F": sc_rec_F, "drg_recall_in_F": drg_rec_F,
       "protected_block_detail": detail}
print(json.dumps(out))
