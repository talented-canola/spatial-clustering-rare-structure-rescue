import sys; sys.path.insert(0,'F:/tmp')
import json, numpy as np, hashlib
import m2_final_lib as F
expr, spat, blocks, a, ann_s, ann, sub = F.load_inputs()
adj = F.weighted_adj(expr, spat, blocks, alpha=1.5, ws=0.2)
lab = F.leiden_w(adj, 1.0, 2)
np.save('F:/tmp/m2_final_pre_alpha1.5_seed2.npy', lab)
sc_core = np.where(ann == F.SC_CLASS)[0]
c = lab[sc_core[0]]
sc_all_together = bool((lab[sc_core] == c).all())
sc_own_pre = sc_all_together and int((lab == c).sum()) == len(sc_core)
m = F.metrics(lab, a, ann_s, sub)
ov = m['overlaps']
sha = hashlib.sha256(lab.astype(np.int32).tobytes()).hexdigest()[:16]
print(json.dumps({"alpha":1.5,"seed":2,"drg_overlap":ov["Dorsal root ganglion"],"sc_overlap":ov["Spinal cord"],"sc_all_together":sc_all_together,"sc_own_pre":sc_own_pre,"n_clusters":m["n_clusters"],"ari":m["ari"],"nmi":m["nmi"],"silhouette":m["silhouette"],"coherence":m["coherence"],"label_sha256":sha}))
