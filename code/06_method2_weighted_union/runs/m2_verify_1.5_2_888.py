import sys; sys.path.insert(0,'F:/tmp')
import json, numpy as np, hashlib
import m2_final_lib as F
expr, spat, blocks, a, ann_s, ann, sub = F.load_inputs()
adj = F.weighted_adj(expr, spat, blocks, alpha=1.5, ws=0.2)
lab = F.leiden_w(adj, 2, 888)
n_pre = int(lab.max()) + 1
lab2, sc_own_pre, triggered = F.sc_split(lab, ann)
np.save('F:/tmp/m2_final_corrected_alpha1.5_res2_seed888.npy', lab2)
m = F.metrics(lab2, a, ann_s, sub)
ov = m['overlaps']
sha = hashlib.sha256(lab2.astype(np.int32).tobytes()).hexdigest()[:16]
print(json.dumps({"alpha":1.5,"res":2,"seed":888,"n_clusters_pre":n_pre,"n_clusters_post":m["n_clusters"],"sc_split_triggered":triggered,"sc_own_pre":sc_own_pre,"sc_overlap_post":ov["Spinal cord"],"drg_overlap_post":ov["Dorsal root ganglion"],"ari":m["ari"],"nmi":m["nmi"],"silhouette":m["silhouette"],"coherence":m["coherence"],"label_sha256":sha,"overlaps":ov}))
