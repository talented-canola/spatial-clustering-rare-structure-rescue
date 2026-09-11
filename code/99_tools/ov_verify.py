# -*- coding: utf-8 -*-
"""Verifier: cross-check recompute vs tables, write surface_ectoderm_check.csv and fig6 figure."""
import os
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import seaborn as sns
    HAVE_SNS = True
except Exception:
    HAVE_SNS = False

RECOMPUTE = [
    {"scheme": "baseline_v2_arpack", "resolution": "0.5", "overlaps": {"Brain": 0.6241, "Mesenchyme": 0.0006, "Head mesenchyme": 0.3129, "Cavity": 0.2877, "Spinal cord": 0, "Dorsal root ganglion": 0.9924, "GI tract": 0.8614, "Surface ectoderm": 0.7235, "Liver": 0.9951, "Heart": 0.9823}},
    {"scheme": "baseline_v2_arpack", "resolution": "1.0", "overlaps": {"Brain": 0.4383, "Mesenchyme": 0.3742, "Head mesenchyme": 0.384, "Cavity": 0.2891, "Spinal cord": 1, "Dorsal root ganglion": 0.9924, "GI tract": 0.8594, "Surface ectoderm": 0.7119, "Liver": 0.9951, "Heart": 0.8137}},
    {"scheme": "baseline_v2_arpack", "resolution": "1.5", "overlaps": {"Brain": 0.4129, "Mesenchyme": 0.2912, "Head mesenchyme": 0.3783, "Cavity": 0.2869, "Spinal cord": 0, "Dorsal root ganglion": 0.9924, "GI tract": 0.753, "Surface ectoderm": 0.4541, "Liver": 0.9963, "Heart": 0.4962}},
    {"scheme": "baseline_v2_arpack", "resolution": "2.0", "overlaps": {"Brain": 0.254, "Mesenchyme": 0.3008, "Head mesenchyme": 0.3759, "Cavity": 0.2096, "Spinal cord": 0, "Dorsal root ganglion": 0.9924, "GI tract": 0.8815, "Surface ectoderm": 0.4816, "Liver": 0.9951, "Heart": 0.4833}},
    {"scheme": "scc_s4", "resolution": "0.5", "overlaps": {"Brain": 0.6125, "Mesenchyme": 0, "Head mesenchyme": 0.3484, "Cavity": 0.2407, "Spinal cord": 1, "Dorsal root ganglion": 0.9924, "GI tract": 0.006, "Surface ectoderm": 0.7455, "Liver": 0.9951, "Heart": 0.9795}},
    {"scheme": "scc_s4", "resolution": "1.0", "overlaps": {"Brain": 0.2176, "Mesenchyme": 0.0313, "Head mesenchyme": 0.3358, "Cavity": 0.2735, "Spinal cord": 0, "Dorsal root ganglion": 0.9924, "GI tract": 0.8635, "Surface ectoderm": 0.4557, "Liver": 0.9951, "Heart": 0.7679}},
    {"scheme": "scc_s4", "resolution": "1.5", "overlaps": {"Brain": 0.457, "Mesenchyme": 0.2669, "Head mesenchyme": 0.3778, "Cavity": 0.2552, "Spinal cord": 0, "Dorsal root ganglion": 0.9924, "GI tract": 0.9016, "Surface ectoderm": 0.4667, "Liver": 0.9975, "Heart": 0.4867}},
    {"scheme": "scc_s4", "resolution": "2.0", "overlaps": {"Brain": 0.2294, "Mesenchyme": 0.3218, "Head mesenchyme": 0.3751, "Cavity": 0.2829, "Spinal cord": 1, "Dorsal root ganglion": 0.9924, "GI tract": 0.743, "Surface ectoderm": 0.4568, "Liver": 0.9939, "Heart": 0.4573}},
    {"scheme": "scc_s8", "resolution": "0.5", "overlaps": {"Brain": 0.622, "Mesenchyme": 0.0006, "Head mesenchyme": 0.269, "Cavity": 0.2469, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0, "Surface ectoderm": 0.751, "Liver": 0.9951, "Heart": 0.9829}},
    {"scheme": "scc_s8", "resolution": "1.0", "overlaps": {"Brain": 0.4188, "Mesenchyme": 0.371, "Head mesenchyme": 0.3761, "Cavity": 0.2541, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0.8675, "Surface ectoderm": 0.4607, "Liver": 0.9939, "Heart": 0.9706}},
    {"scheme": "scc_s8", "resolution": "1.5", "overlaps": {"Brain": 0.2569, "Mesenchyme": 0.2867, "Head mesenchyme": 0.3807, "Cavity": 0.2387, "Spinal cord": 0, "Dorsal root ganglion": 0.9924, "GI tract": 0.8594, "Surface ectoderm": 0.4678, "Liver": 0.9939, "Heart": 0.8041}},
    {"scheme": "scc_s8", "resolution": "2.0", "overlaps": {"Brain": 0.2558, "Mesenchyme": 0.2912, "Head mesenchyme": 0.3363, "Cavity": 0.2492, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0.743, "Surface ectoderm": 0.4552, "Liver": 0.9951, "Heart": 0.6573}},
    {"scheme": "scc_s12", "resolution": "0.5", "overlaps": {"Brain": 0.6223, "Mesenchyme": 0.3844, "Head mesenchyme": 0.4593, "Cavity": 0.1634, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0, "Surface ectoderm": 0.7455, "Liver": 0.9988, "Heart": 0.9836}},
    {"scheme": "scc_s12", "resolution": "1.0", "overlaps": {"Brain": 0.6225, "Mesenchyme": 0.3033, "Head mesenchyme": 0.3449, "Cavity": 0.2367, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0.743, "Surface ectoderm": 0.4722, "Liver": 0.9975, "Heart": 0.9843}},
    {"scheme": "scc_s12", "resolution": "1.5", "overlaps": {"Brain": 0.261, "Mesenchyme": 0.2969, "Head mesenchyme": 0.3837, "Cavity": 0.2412, "Spinal cord": 0, "Dorsal root ganglion": 0.9771, "GI tract": 0.8032, "Surface ectoderm": 0.4722, "Liver": 0.9988, "Heart": 0.6587}},
    {"scheme": "scc_s12", "resolution": "2.0", "overlaps": {"Brain": 0.2207, "Mesenchyme": 0.2957, "Head mesenchyme": 0.378, "Cavity": 0.2501, "Spinal cord": 0, "Dorsal root ganglion": 0.0153, "GI tract": 0.745, "Surface ectoderm": 0.4695, "Liver": 0.9951, "Heart": 0.7092}},
    {"scheme": "smooth_s8", "resolution": "0.5", "overlaps": {"Brain": 0.4524, "Mesenchyme": 0.2848, "Head mesenchyme": 0.318, "Cavity": 0.1982, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0.8454, "Surface ectoderm": 0.4706, "Liver": 0.984, "Heart": 0.9904}},
    {"scheme": "smooth_s8", "resolution": "1.0", "overlaps": {"Brain": 0.2684, "Mesenchyme": 0.2899, "Head mesenchyme": 0.3148, "Cavity": 0.1982, "Spinal cord": 0, "Dorsal root ganglion": 0.0229, "GI tract": 0.6185, "Surface ectoderm": 0.2501, "Liver": 0.984, "Heart": 0.5386}},
    {"scheme": "smooth_s8", "resolution": "1.5", "overlaps": {"Brain": 0.244, "Mesenchyme": 0.3301, "Head mesenchyme": 0.2132, "Cavity": 0.1383, "Spinal cord": 0, "Dorsal root ganglion": 0.9389, "GI tract": 0.5863, "Surface ectoderm": 0.1842, "Liver": 0.9816, "Heart": 0.4334}},
    {"scheme": "smooth_s8", "resolution": "2.0", "overlaps": {"Brain": 0.1747, "Mesenchyme": 0.272, "Head mesenchyme": 0.136, "Cavity": 0.1454, "Spinal cord": 0, "Dorsal root ganglion": 0.9618, "GI tract": 0.6426, "Surface ectoderm": 0.2183, "Liver": 0.9804, "Heart": 0.4382}},
    {"scheme": "smooth_s8_incl_self", "resolution": "0.5", "overlaps": {"Brain": 0.4475, "Mesenchyme": 0.2656, "Head mesenchyme": 0.2929, "Cavity": 0.2404, "Spinal cord": 0, "Dorsal root ganglion": 0.0305, "GI tract": 0, "Surface ectoderm": 0.4596, "Liver": 0.9865, "Heart": 0.9754}},
    {"scheme": "smooth_s8_incl_self", "resolution": "1.0", "overlaps": {"Brain": 0.2746, "Mesenchyme": 0.2771, "Head mesenchyme": 0.3172, "Cavity": 0.2227, "Spinal cord": 0, "Dorsal root ganglion": 0.0229, "GI tract": 0.751, "Surface ectoderm": 0.464, "Liver": 0.9865, "Heart": 0.5693}},
    {"scheme": "smooth_s8_incl_self", "resolution": "1.5", "overlaps": {"Brain": 0.2212, "Mesenchyme": 0.3308, "Head mesenchyme": 0.2291, "Cavity": 0.1543, "Spinal cord": 0.05, "Dorsal root ganglion": 0.0305, "GI tract": 0.6044, "Surface ectoderm": 0.2496, "Liver": 0.9865, "Heart": 0.4471}},
    {"scheme": "smooth_s8_incl_self", "resolution": "2.0", "overlaps": {"Brain": 0.1996, "Mesenchyme": 0.2765, "Head mesenchyme": 0.1939, "Cavity": 0.1503, "Spinal cord": 0.05, "Dorsal root ganglion": 0.9618, "GI tract": 0.6064, "Surface ectoderm": 0.2512, "Liver": 0.9988, "Heart": 0.4143}},
    {"scheme": "graphst_reference", "resolution": "0.5", "overlaps": {"Brain": 0.4421, "Mesenchyme": 0.2586, "Head mesenchyme": 0.3271, "Cavity": 0.203, "Spinal cord": 0, "Dorsal root ganglion": 0.0229, "GI tract": 0.7711, "Surface ectoderm": 0.3397, "Liver": 0.9767, "Heart": 0.9823}},
    {"scheme": "graphst_reference", "resolution": "1.0", "overlaps": {"Brain": 0.3072, "Mesenchyme": 0.2739, "Head mesenchyme": 0.3236, "Cavity": 0.1851, "Spinal cord": 0, "Dorsal root ganglion": 0.9695, "GI tract": 0.7711, "Surface ectoderm": 0.2694, "Liver": 0.9816, "Heart": 0.5474}},
    {"scheme": "graphst_reference", "resolution": "1.5", "overlaps": {"Brain": 0.1714, "Mesenchyme": 0.2867, "Head mesenchyme": 0.336, "Cavity": 0.1751, "Spinal cord": 0.9, "Dorsal root ganglion": 0.9695, "GI tract": 0.7711, "Surface ectoderm": 0.2259, "Liver": 0.9804, "Heart": 0.8205}},
    {"scheme": "graphst_reference", "resolution": "2.0", "overlaps": {"Brain": 0.1978, "Mesenchyme": 0.2874, "Head mesenchyme": 0.3026, "Cavity": 0.1477, "Spinal cord": 0, "Dorsal root ganglion": 0.9695, "GI tract": 0.755, "Surface ectoderm": 0.2248, "Liver": 0.9816, "Heart": 0.4314}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "0.5", "overlaps": {"Brain": 0.2286, "Mesenchyme": 0.2918, "Head mesenchyme": 0.2596, "Cavity": 0.1354, "Spinal cord": 0, "Dorsal root ganglion": 0.0153, "GI tract": 0.6506, "Surface ectoderm": 0.0874, "Liver": 0.9558, "Heart": 0.5509}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "1.0", "overlaps": {"Brain": 0.215, "Mesenchyme": 0.1584, "Head mesenchyme": 0.2321, "Cavity": 0.1178, "Spinal cord": 0, "Dorsal root ganglion": 0.0076, "GI tract": 0.5502, "Surface ectoderm": 0.1539, "Liver": 0.9411, "Heart": 0.5072}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "1.5", "overlaps": {"Brain": 0.203, "Mesenchyme": 0.1711, "Head mesenchyme": 0.1516, "Cavity": 0.0707, "Spinal cord": 0, "Dorsal root ganglion": 0.0229, "GI tract": 0.5984, "Surface ectoderm": 0.116, "Liver": 0.4319, "Heart": 0.1986}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "2.0", "overlaps": {"Brain": 0.1645, "Mesenchyme": 0.2133, "Head mesenchyme": 0.1177, "Cavity": 0.0827, "Spinal cord": 1, "Dorsal root ganglion": 0.9771, "GI tract": 0.4378, "Surface ectoderm": 0.1215, "Liver": 0.589, "Heart": 0.2034}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "0.5", "overlaps": {"Brain": 0.2969, "Mesenchyme": 0.4004, "Head mesenchyme": 0.3172, "Cavity": 0.2198, "Spinal cord": 0, "Dorsal root ganglion": 0.9695, "GI tract": 0.753, "Surface ectoderm": 0.4151, "Liver": 0.9767, "Heart": 0.9099}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "1.0", "overlaps": {"Brain": 0.2299, "Mesenchyme": 0.3653, "Head mesenchyme": 0.3064, "Cavity": 0.1916, "Spinal cord": 0, "Dorsal root ganglion": 0.9695, "GI tract": 0.761, "Surface ectoderm": 0.4162, "Liver": 0.9264, "Heart": 0.6608}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "1.5", "overlaps": {"Brain": 0.1901, "Mesenchyme": 0.3269, "Head mesenchyme": 0.248, "Cavity": 0.1411, "Spinal cord": 0, "Dorsal root ganglion": 0.9695, "GI tract": 0.761, "Surface ectoderm": 0.4217, "Liver": 0.9129, "Heart": 0.7106}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "2.0", "overlaps": {"Brain": 0.1324, "Mesenchyme": 0.3301, "Head mesenchyme": 0.2418, "Cavity": 0.1323, "Spinal cord": 0, "Dorsal root ganglion": 0.9695, "GI tract": 0.761, "Surface ectoderm": 0.3364, "Liver": 0.9288, "Heart": 0.4014}},
]

TABLES = [
    {"scheme": "baseline_v2_arpack", "resolution": "0.5", "overlaps": {"Brain": 0.624, "Mesenchyme": 0.001, "Head mesenchyme": 0.313, "Cavity": 0.288, "Spinal cord": 0, "Dorsal root ganglion": 0.992, "GI tract": 0.861, "Surface ectoderm": 0.723, "Liver": 0.995, "Heart": 0.982}},
    {"scheme": "baseline_v2_arpack", "resolution": "1.0", "overlaps": {"Brain": 0.438, "Mesenchyme": 0.374, "Head mesenchyme": 0.384, "Cavity": 0.289, "Spinal cord": 1, "Dorsal root ganglion": 0.992, "GI tract": 0.859, "Surface ectoderm": 0.712, "Liver": 0.995, "Heart": 0.814}},
    {"scheme": "baseline_v2_arpack", "resolution": "1.5", "overlaps": {"Brain": 0.413, "Mesenchyme": 0.291, "Head mesenchyme": 0.378, "Cavity": 0.287, "Spinal cord": 0, "Dorsal root ganglion": 0.992, "GI tract": 0.753, "Surface ectoderm": 0.454, "Liver": 0.996, "Heart": 0.496}},
    {"scheme": "baseline_v2_arpack", "resolution": "2.0", "overlaps": {"Brain": 0.254, "Mesenchyme": 0.301, "Head mesenchyme": 0.376, "Cavity": 0.21, "Spinal cord": 0, "Dorsal root ganglion": 0.992, "GI tract": 0.882, "Surface ectoderm": 0.482, "Liver": 0.995, "Heart": 0.483}},
    {"scheme": "scc_s8", "resolution": "1.0", "overlaps": {"Brain": 0.419, "Mesenchyme": 0.371, "Head mesenchyme": 0.376, "Cavity": 0.254, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0.867, "Surface ectoderm": 0.461, "Liver": 0.994, "Heart": 0.971}},
    {"scheme": "smooth_s8", "resolution": "0.5", "overlaps": {"Brain": 0.452, "Mesenchyme": 0.285, "Head mesenchyme": 0.318, "Cavity": 0.198, "Spinal cord": 0, "Dorsal root ganglion": 0, "GI tract": 0.845, "Surface ectoderm": 0.471, "Liver": 0.984, "Heart": 0.99}},
    {"scheme": "smooth_s8", "resolution": "1.0", "overlaps": {"Brain": 0.268, "Mesenchyme": 0.29, "Head mesenchyme": 0.315, "Cavity": 0.198, "Spinal cord": 0, "Dorsal root ganglion": 0.023, "GI tract": 0.618, "Surface ectoderm": 0.25, "Liver": 0.984, "Heart": 0.539}},
    {"scheme": "smooth_s8", "resolution": "1.5", "overlaps": {"Brain": 0.244, "Mesenchyme": 0.33, "Head mesenchyme": 0.213, "Cavity": 0.138, "Spinal cord": 0, "Dorsal root ganglion": 0.939, "GI tract": 0.586, "Surface ectoderm": 0.184, "Liver": 0.982, "Heart": 0.433}},
    {"scheme": "smooth_s8", "resolution": "2.0", "overlaps": {"Brain": 0.175, "Mesenchyme": 0.272, "Head mesenchyme": 0.136, "Cavity": 0.145, "Spinal cord": 0, "Dorsal root ganglion": 0.962, "GI tract": 0.643, "Surface ectoderm": 0.218, "Liver": 0.98, "Heart": 0.438}},
    {"scheme": "smooth_s8_incl_self", "resolution": "0.5", "overlaps": {"Brain": 0.448, "Mesenchyme": 0.266, "Head mesenchyme": 0.293, "Cavity": 0.24, "Spinal cord": 0, "Dorsal root ganglion": 0.031, "GI tract": 0, "Surface ectoderm": 0.46, "Liver": 0.987, "Heart": 0.975}},
    {"scheme": "smooth_s8_incl_self", "resolution": "1.0", "overlaps": {"Brain": 0.275, "Mesenchyme": 0.277, "Head mesenchyme": 0.317, "Cavity": 0.223, "Spinal cord": 0, "Dorsal root ganglion": 0.023, "GI tract": 0.751, "Surface ectoderm": 0.464, "Liver": 0.987, "Heart": 0.569}},
    {"scheme": "smooth_s8_incl_self", "resolution": "1.5", "overlaps": {"Brain": 0.221, "Mesenchyme": 0.331, "Head mesenchyme": 0.229, "Cavity": 0.154, "Spinal cord": 0.05, "Dorsal root ganglion": 0.031, "GI tract": 0.604, "Surface ectoderm": 0.25, "Liver": 0.987, "Heart": 0.447}},
    {"scheme": "smooth_s8_incl_self", "resolution": "2.0", "overlaps": {"Brain": 0.2, "Mesenchyme": 0.277, "Head mesenchyme": 0.194, "Cavity": 0.15, "Spinal cord": 0.05, "Dorsal root ganglion": 0.962, "GI tract": 0.606, "Surface ectoderm": 0.251, "Liver": 0.999, "Heart": 0.414}},
    {"scheme": "graphst_reference", "resolution": "0.5", "overlaps": {"Brain": 0.442, "Mesenchyme": 0.259, "Head mesenchyme": 0.327, "Cavity": 0.203, "Spinal cord": 0, "Dorsal root ganglion": 0.023, "GI tract": 0.771, "Surface ectoderm": 0.34, "Liver": 0.977, "Heart": 0.982}},
    {"scheme": "graphst_reference", "resolution": "1.0", "overlaps": {"Brain": 0.307, "Mesenchyme": 0.274, "Head mesenchyme": 0.324, "Cavity": 0.185, "Spinal cord": 0, "Dorsal root ganglion": 0.969, "GI tract": 0.771, "Surface ectoderm": 0.269, "Liver": 0.982, "Heart": 0.547}},
    {"scheme": "graphst_reference", "resolution": "1.5", "overlaps": {"Brain": 0.171, "Mesenchyme": 0.287, "Head mesenchyme": 0.336, "Cavity": 0.175, "Spinal cord": 0.9, "Dorsal root ganglion": 0.969, "GI tract": 0.771, "Surface ectoderm": 0.226, "Liver": 0.98, "Heart": 0.82}},
    {"scheme": "graphst_reference", "resolution": "2.0", "overlaps": {"Brain": 0.198, "Mesenchyme": 0.287, "Head mesenchyme": 0.303, "Cavity": 0.148, "Spinal cord": 0, "Dorsal root ganglion": 0.969, "GI tract": 0.755, "Surface ectoderm": 0.225, "Liver": 0.982, "Heart": 0.431}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "0.5", "overlaps": {"Brain": 0.229, "Mesenchyme": 0.292, "Head mesenchyme": 0.26, "Cavity": 0.135, "Spinal cord": 0, "Dorsal root ganglion": 0.015, "GI tract": 0.651, "Surface ectoderm": 0.087, "Liver": 0.956, "Heart": 0.551}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "1.0", "overlaps": {"Brain": 0.215, "Mesenchyme": 0.158, "Head mesenchyme": 0.232, "Cavity": 0.118, "Spinal cord": 0, "Dorsal root ganglion": 0.008, "GI tract": 0.55, "Surface ectoderm": 0.154, "Liver": 0.941, "Heart": 0.507}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "1.5", "overlaps": {"Brain": 0.203, "Mesenchyme": 0.171, "Head mesenchyme": 0.152, "Cavity": 0.071, "Spinal cord": 0, "Dorsal root ganglion": 0.023, "GI tract": 0.598, "Surface ectoderm": 0.116, "Liver": 0.432, "Heart": 0.199}},
    {"scheme": "method1_bilateral_spatial_knn", "resolution": "2.0", "overlaps": {"Brain": 0.164, "Mesenchyme": 0.213, "Head mesenchyme": 0.118, "Cavity": 0.083, "Spinal cord": 1, "Dorsal root ganglion": 0.977, "GI tract": 0.438, "Surface ectoderm": 0.121, "Liver": 0.589, "Heart": 0.203}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "0.5", "overlaps": {"Brain": 0.297, "Mesenchyme": 0.4, "Head mesenchyme": 0.317, "Cavity": 0.22, "Spinal cord": 0, "Dorsal root ganglion": 0.969, "GI tract": 0.753, "Surface ectoderm": 0.415, "Liver": 0.977, "Heart": 0.91}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "1.0", "overlaps": {"Brain": 0.23, "Mesenchyme": 0.365, "Head mesenchyme": 0.306, "Cavity": 0.192, "Spinal cord": 0, "Dorsal root ganglion": 0.969, "GI tract": 0.761, "Surface ectoderm": 0.416, "Liver": 0.926, "Heart": 0.661}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "1.5", "overlaps": {"Brain": 0.19, "Mesenchyme": 0.327, "Head mesenchyme": 0.248, "Cavity": 0.141, "Spinal cord": 0, "Dorsal root ganglion": 0.969, "GI tract": 0.761, "Surface ectoderm": 0.422, "Liver": 0.913, "Heart": 0.711}},
    {"scheme": "method1_bilateral_expr_knn", "resolution": "2.0", "overlaps": {"Brain": 0.132, "Mesenchyme": 0.33, "Head mesenchyme": 0.242, "Cavity": 0.132, "Spinal cord": 0, "Dorsal root ganglion": 0.969, "GI tract": 0.761, "Surface ectoderm": 0.336, "Liver": 0.929, "Heart": 0.401}},
]

# ---------- Check 1: cross-check recompute vs tables ----------
mismatches = []
checked = 0
recompute_lookup = {(r["scheme"], r["resolution"]): r["overlaps"] for r in RECOMPUTE}
for t in TABLES:
    key = (t["scheme"], t["resolution"])
    rv = recompute_lookup.get(key)
    if rv is None:
        continue
    if not t["overlaps"]:
        continue
    for tissue, tv in t["overlaps"].items():
        if tissue not in rv:
            mismatches.append(f"{t['scheme']}@{t['resolution']}/{tissue} recompute missing")
            continue
        diff = abs(rv[tissue] - tv)
        checked += 1
        if diff > 0.005:
            mismatches.append(f"{t['scheme']}@{t['resolution']}/{tissue} recompute={rv[tissue]:.4f} table={tv:.4f}")

print("CHECK1: numeric comparisons done:", checked, "mismatches:", len(mismatches))
for m in mismatches:
    print("  MISMATCH:", m)

# Which table rows are missing vs recompute?
missing_rows = []
for r in RECOMPUTE:
    key = (r["scheme"], r["resolution"])
    present = any(k[0] == key[0] and k[1] == key[1] and t["overlaps"] for k, t in ((k, tk) for k, tk in [( (r["scheme"], r["resolution"]), t) for t in TABLES]))
    if not present:
        missing_rows.append(f"{r['scheme']}@{r['resolution']}")
print("TABLE MISSING (recompute has data, tables empty/missing):", sorted(set(missing_rows)))

# ---------- Anchor check ----------
r10 = recompute_lookup[("baseline_v2_arpack", "1.0")]
smooth10 = recompute_lookup[("smooth_s8", "1.0")]
print("anchor baseline@1.0 Surface ectoderm recompute:", r10["Surface ectoderm"])
print("anchor smooth_s8@1.0 Surface ectoderm recompute:", smooth10["Surface ectoderm"])
print("smooth_s8@0.5 Surface ectoderm (res=0.5):", recompute_lookup[("smooth_s8", "0.5")]["Surface ectoderm"])

# ---------- CHECK 2: fig5 numbers at res=1.0 ----------
print("\nCHECK2 Surface ectoderm @ res=1.0:")
for s in ["baseline_v2_arpack", "scc_s4", "scc_s8", "scc_s12", "smooth_s8", "smooth_s8_incl_self", "graphst_reference", "method1_bilateral_spatial_knn", "method1_bilateral_expr_knn"]:
    print(f"  {s}: {recompute_lookup[(s,'1.0')]['Surface ectoderm']:.4f}")

# ---------- CHECK 4: fig3 numbers at res=1.0 ----------
print("\nCHECK4 Liver/Heart @ res=1.0:")
for s in ["baseline_v2_arpack", "graphst_reference"]:
    o = recompute_lookup[(s, "1.0")]
    print(f"  {s}: Liver={o['Liver']:.4f} Heart={o['Heart']:.4f}")
print("graphst Liver/Heart across res:", [(res, recompute_lookup[('graphst_reference', res)]['Liver'], recompute_lookup[('graphst_reference', res)]['Heart']) for res in ['0.5','1.0','1.5','2.0']])
print("baseline Liver/Heart across res:", [(res, recompute_lookup[('baseline_v2_arpack', res)]['Liver'], recompute_lookup[('baseline_v2_arpack', res)]['Heart']) for res in ['0.5','1.0','1.5','2.0']])

# ---------- Mechanism ----------
MECH = [
    {"tissue": "Surface ectoderm", "n_cells": 1819, "spatial_purity_mean": 0.7758, "bilat_spatial_deg_mean": 3.0767, "bilat_expr_deg_mean": 8.3875, "frac_neigh_diff_tissue": 0.2242},
    {"tissue": "Liver", "n_cells": 815, "spatial_purity_mean": 0.938, "bilat_spatial_deg_mean": 3.2398, "bilat_expr_deg_mean": 8.7281, "frac_neigh_diff_tissue": 0.062},
    {"tissue": "Heart", "n_cells": 1465, "spatial_purity_mean": 0.9266, "bilat_spatial_deg_mean": 2.6459, "bilat_expr_deg_mean": 8.9416, "frac_neigh_diff_tissue": 0.0734},
    {"tissue": "Brain", "n_cells": 3897, "spatial_purity_mean": 0.936, "bilat_spatial_deg_mean": 3.1634, "bilat_expr_deg_mean": 10.0948, "frac_neigh_diff_tissue": 0.064},
    {"tissue": "Mesenchyme", "n_cells": 1566, "spatial_purity_mean": 0.64, "bilat_spatial_deg_mean": 3.3469, "bilat_expr_deg_mean": 9.1065, "frac_neigh_diff_tissue": 0.36},
    {"tissue": "Head mesenchyme", "n_cells": 3714, "spatial_purity_mean": 0.7929, "bilat_spatial_deg_mean": 2.7933, "bilat_expr_deg_mean": 8.3688, "frac_neigh_diff_tissue": 0.2071},
    {"tissue": "Cavity", "n_cells": 3507, "spatial_purity_mean": 0.8035, "bilat_spatial_deg_mean": 2.9948, "bilat_expr_deg_mean": 9.3436, "frac_neigh_diff_tissue": 0.1965},
    {"tissue": "Spinal cord", "n_cells": 60, "spatial_purity_mean": 0.7625, "bilat_spatial_deg_mean": 3.1258, "bilat_expr_deg_mean": 11.8729, "frac_neigh_diff_tissue": 0.2375},
    {"tissue": "Dorsal root ganglion", "n_cells": 131, "spatial_purity_mean": 0.8302, "bilat_spatial_deg_mean": 3.0377, "bilat_expr_deg_mean": 12.491, "frac_neigh_diff_tissue": 0.1698},
    {"tissue": "GI tract", "n_cells": 498, "spatial_purity_mean": 0.7957, "bilat_spatial_deg_mean": 3.7549, "bilat_expr_deg_mean": 11.7098, "frac_neigh_diff_tissue": 0.2043},
    {"tissue": "Connective tissue", "n_cells": 901, "spatial_purity_mean": 0.8302, "bilat_spatial_deg_mean": 3.1042, "bilat_expr_deg_mean": 8.771, "frac_neigh_diff_tissue": 0.1698},
    {"tissue": "ALL_OTHER", "n_cells": 25559, "spatial_purity_mean": 0.8277, "bilat_spatial_deg_mean": 3.0677, "bilat_expr_deg_mean": 9.3555, "frac_neigh_diff_tissue": 0.1723},
]
print("\nCHECK3 Mechanism ranking (Surface ectoderm):")
se = next(m for m in MECH if m["tissue"] == "Surface ectoderm")
ao = next(m for m in MECH if m["tissue"] == "ALL_OTHER")
for metric in ["spatial_purity_mean", "bilat_spatial_deg_mean", "bilat_expr_deg_mean", "frac_neigh_diff_tissue"]:
    ranked = sorted(MECH, key=lambda m: m[metric])
    pos = ranked.index(se) + 1
    print(f"  {metric}: SE={se[metric]:.4f} vs ALL_OTHER={ao[metric]:.4f}; rank {pos}/{len(ranked)} (asc), rank (desc) {len(ranked)-pos+1}/{len(ranked)}")

# ---------- ARTIFACT (a): CSV ----------
import csv
csv_path = r"F:/BGI/task3/spateo-release-main/results/analysis/surface_ectoderm_check.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["scheme", "resolution", "tissue", "overlap"])
    n_rows = 0
    for r in RECOMPUTE:
        for tissue, v in r["overlaps"].items():
            w.writerow([r["scheme"], r["resolution"], tissue, round(float(v), 6)])
            n_rows += 1
print("\nCSV written:", csv_path, "rows:", n_rows)
# verify readback
with open(csv_path, "r", encoding="utf-8-sig") as f:
    lines = f.readlines()
print("CSV readback lines (incl header):", len(lines))
if lines[0].startswith("\ufeff"):
    print("BOM present: OK")

# ---------- ARTIFACT (b): Figure ----------
schemes = ["baseline_v2_arpack", "scc_s4", "scc_s8", "scc_s12", "smooth_s8", "smooth_s8_incl_self",
           "graphst_reference", "method1_bilateral_spatial_knn", "method1_bilateral_expr_knn"]
labels = {
    "baseline_v2_arpack": "Baseline",
    "scc_s4": "SCC s4",
    "scc_s8": "SCC s8",
    "scc_s12": "SCC s12",
    "smooth_s8": "Smooth s8",
    "smooth_s8_incl_self": "Smooth s8+self",
    "graphst_reference": "GraphST",
    "method1_bilateral_spatial_knn": "Method1 Spat",
    "method1_bilateral_expr_knn": "Method1 Expr",
}
RED = "#d62728"
BLUE = "#1f77b4"
RES = "1.0"

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
fig.suptitle("Cross-scheme overlap at resolution = 1.0 (Hungarian-aligned 19-class)", fontsize=13)

# Panel A: Surface ectoderm horizontal bar, sorted descending
ax = axes[0]
vals = sorted([(s, recompute_lookup[(s, RES)]["Surface ectoderm"]) for s in schemes], key=lambda x: x[1], reverse=True)
y = np.arange(len(vals))[::-1]
ovl = [v for _, v in vals]
cols = [RED if s in ("method1_bilateral_spatial_knn", "method1_bilateral_expr_knn") else BLUE for s, _ in vals]
ax.barh(y, ovl, color=cols, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels([labels[s] for s, _ in vals], fontsize=9)
for yy, o in zip(y, ovl):
    ax.text(o + 0.008, yy, f"{o:.3f}", va="center", fontsize=9)
ax.axvline(0.71, color="grey", ls="--", lw=1.2)
ax.text(0.71, len(vals) - 0.35, " baseline\n 0.71", fontsize=8, color="grey", va="top", ha="left")
ax.axvline(0.47, color="grey", ls=":", lw=1.2)
ax.text(0.47, len(vals) - 0.35, " smooth\n anchor 0.47", fontsize=8, color="grey", va="top", ha="right")
ax.set_xlim(0, 1.0)
ax.set_title("A. Surface ectoderm overlap (res=1.0)", fontsize=10)
ax.set_xlabel("Overlap fraction")

# Panel B: grouped bar Liver & Heart
ax = axes[1]
x = np.arange(len(schemes))
w = 0.38
liver = [recompute_lookup[(s, RES)]["Liver"] for s in schemes]
heart = [recompute_lookup[(s, RES)]["Heart"] for s in schemes]
b1 = ax.bar(x - w / 2, liver, w, color="#2ca02c", label="Liver")
b2 = ax.bar(x + w / 2, heart, w, color="#ff7f0e", label="Heart")
for xi, lv, hv in zip(x, liver, heart):
    ax.text(xi - w / 2, lv + 0.02, f"{lv:.2f}", ha="center", va="bottom", fontsize=7)
    ax.text(xi + w / 2, hv + 0.02, f"{hv:.2f}", ha="center", va="bottom", fontsize=7)
ax.axhline(0.9951, color="#2ca02c", ls="--", lw=1.0)
ax.axhline(0.8137, color="#ff7f0e", ls="--", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([labels[s] for s in schemes], rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title("B. Extreme tissues Liver & Heart (res=1.0)", fontsize=10)
ax.set_ylabel("Overlap fraction")
ax.legend(fontsize=8)

# Panel C: heatmap 8 tissues x 9 schemes, Surface ectoderm LAST row boxed
ax = axes[2]
tissues8 = ["Brain", "Mesenchyme", "Head mesenchyme", "Cavity", "Spinal cord", "Dorsal root ganglion", "GI tract", "Surface ectoderm"]
mat = np.array([[recompute_lookup[(s, RES)][t] for s in schemes] for t in tissues8])
if HAVE_SNS:
    sns.heatmap(mat, cmap="RdBu_r", vmin=0, vmax=1, annot=True, fmt=".3f", annot_kws={"fontsize": 7},
                xticklabels=[labels[s] for s in schemes], yticklabels=tissues8, ax=ax, cbar=True,
                linewidths=0.5, linecolor="white")
    ax.tick_params(axis="x", rotation=45)
else:
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels([labels[s] for s in schemes], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(tissues8)))
    ax.set_yticklabels(tissues8, fontsize=8)
# bold box around Surface ectoderm (last) row
ax.add_patch(Rectangle((-0.5, len(tissues8) - 1 - 0.5), len(schemes), 1, fill=False, edgecolor="black", lw=2.5))
ax.set_title("C. 8-tissue overlap heatmap (res=1.0)", fontsize=10)

fig.tight_layout()
fig_path = r"F:/BGI/task3/spateo-release-main/results/analysis/figures/fig6_surface_ectoderm_extreme_tissues.png"
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
fig.savefig(fig_path, dpi=130)
plt.close(fig)
print("Figure written:", fig_path)
print("exists:", os.path.exists(fig_path), "size:", os.path.getsize(fig_path) if os.path.exists(fig_path) else 0)
