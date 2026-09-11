import json, csv, itertools, hashlib, statistics

with open('F:/tmp/marking_runs.json', encoding='utf-8') as f:
    runs = json.load(f)

print("parsed runs:", [r['seed'] for r in runs])
print("counts:", [(r['seed'], len(r['sc_proto_bins']), len(r['flag_bins'])) for r in runs])

def jaccard(a, b):
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 1.0

def pair_stats(key):
    vals = []
    for (i, j) in itertools.combinations(range(len(runs)), 2):
        v = jaccard(runs[i][key], runs[j][key])
        vals.append(v)
    return statistics.mean(vals), min(vals), vals

pm, pmin, pvals = pair_stats('sc_proto_bins')
fm, fmin, fvals = pair_stats('flag_bins')

recalls = [r['sc_recall'] for r in runs]
precs = [r['sc_precision'] for r in runs]

print("PROTO Jaccard pairs:", [round(v,4) for v in pvals])
print("PROTO mean=%.4f min=%.4f" % (pm, pmin))
print("FLAG  Jaccard pairs:", [round(v,4) for v in fvals])
print("FLAG  mean=%.4f min=%.4f" % (fm, fmin))
print("recall mean=%.4f std=%.4f" % (statistics.mean(recalls), statistics.pstdev(recalls)))
print("precision mean=%.4f std=%.4f" % (statistics.mean(precs), statistics.pstdev(precs)))

# check each seed proto contains full SC set (recall 1) - verify proto sizes vs SC total 60
for r in runs:
    print("seed", r['seed'], "proto_size", r['sc_proto_size'], "flag_n", len(r['flag_bins']))

# write CSV
csv_path = 'F:/BGI/task3/spateo-release-main/results/analysis/method2_marking_stability.csv'
with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['seed','n_clusters','ARI','sc_proto_size','sc_recall','sc_precision','flag_bins_n','flag_sizes','label_sha256'])
    for r in runs:
        w.writerow([r['seed'], r['n_clusters'], r['ARI'], r['sc_proto_size'], r['sc_recall'], r['sc_precision'],
                    len(r['flag_bins']), repr(r['flag_sizes']), r['label_sha256']])
print("CSV written:", csv_path)
