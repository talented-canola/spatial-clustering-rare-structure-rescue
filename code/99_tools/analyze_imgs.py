from PIL import Image
import numpy as np, os, glob

def analyze(path):
    im = Image.open(path).convert('RGB')
    # downscale so color counting is fast
    im2 = im.resize((max(1, im.width//4), max(1, im.height//4)))
    a = np.asarray(im2).astype(int)
    h, w, _ = a.shape
    tot = h * w
    white = (np.all(a > 235, axis=2)).mean() * 100
    q = (a // 24) * 24
    flat = q.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    nz = counts >= tot * 0.0005
    cols = cols[nz]; counts = counts[nz]
    nonwhite = ~(np.all(cols > 228, axis=1))
    cols = cols[nonwhite]; counts = counts[nonwhite]
    ncol = len(cols)
    return im.width, im.height, round(white, 1), ncol

samples = ['Y40365K1', 'Y40365P3', 'Y40627HD']
keys = ['violin_raw', 'violin_filt', 'umap_scDblFinder', 'umap_cluster',
        'umap_anno', 'dotplot', 'umap_T_marker', 'umap_core_marker']
for s in samples:
    print('####', s)
    for v in ['', '_v2']:
        d = s + v
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(d + '/' + s + '_*.png')):
            b = os.path.basename(f)
            if 'T_marker_group' in b:
                continue
            try:
                W, H, white, ncol = analyze(f)
                print('  %-3s %-62s %dx%d  white=%5.1f%%  colors~%d' % (v or 'v1', b, W, H, white, ncol))
            except Exception as e:
                print('  ERR', f, e)
