import sys; sys.path.insert(0, "F:/tmp")
import numpy as np
import m2_final_lib as F

expr, spat, blocks, a, ann_s, ann, sub = F.load_inputs()
for alpha in [1.0, 1.5, 2.0, 3.0]:
    adj = F.weighted_adj(expr, spat, blocks, alpha=alpha, ws=0.2)
    lab = F.leiden_w(adj, 1.0, 888)
    m_pre = F.metrics(lab, a, ann_s, sub)
    ov = m_pre["overlaps"]
    lab2, own, trig = F.sc_split(lab, ann)
    m_post = F.metrics(lab2, a, ann_s, sub)
    print("alpha=%s  pre: ARI=%.4f NMI=%.4f ncl=%d  DRG=%.4f SC=%.4f | post: ARI=%.4f ncl=%d SC=%.4f split=%s"
          % (alpha, m_pre["ari"], m_pre["nmi"], m_pre["n_clusters"], ov["Dorsal root ganglion"], ov["Spinal cord"],
             m_post["ari"], m_post["n_clusters"], m_post["overlaps"]["Spinal cord"], trig))
