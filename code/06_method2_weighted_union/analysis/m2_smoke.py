import sys; sys.path.insert(0, "F:/tmp")
import numpy as np, json
import m2_final_lib as F

expr, spat, blocks, a, ann_s, ann, sub = F.load_inputs()
print("loaded ok; expr nnz=%d spat nnz=%d blocks_in_F=%d" % (expr.nnz, spat.nnz, int((blocks>0).sum())))

adj = F.weighted_adj(expr, spat, blocks, alpha=2.0, ws=0.2)
print("adj nnz=%d symmetric=%s" % (adj.nnz, (abs(adj-adj.T).sum()==0)))

lab = F.leiden_w(adj, 1.0, 888)
print("leiden res=1.0 seed=888 n_clusters=%d" % (lab.max()+1))

lab2, sc_own_pre, triggered = F.sc_split(lab, ann)
print("SC split triggered=%s own_pre=%s" % (triggered, sc_own_pre))
print("SC overlap post =", F.metrics(lab2, a, ann_s, sub)["overlaps"]["Spinal cord"])
print("DRG overlap post =", F.metrics(lab2, a, ann_s, sub)["overlaps"]["Dorsal root ganglion"])

m = F.metrics(lab2, a, ann_s, sub)
print("n_clusters=%d ari=%.4f nmi=%.4f sil=%.4f coh=%.4f" % (m["n_clusters"], m["ari"], m["nmi"], m["silhouette"], m["coherence"]))
