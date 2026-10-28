#!/usr/bin/env python3
"""86_zerov.py — 生成网络的零重叠边比例 (解剖学可行性指标)
用法: python3 86_zerov.py <prefix> [nsamp]
"""
import numpy as np, sys, json
D = "/mnt/agents/output/果蝇生成器/data_female/"
_npz = np.load(D + "np_masks.npz")
OUT_lo, OUT_hi, IN_lo, IN_hi = _npz["OUT_lo"], _npz["OUT_hi"], _npz["IN_lo"], _npz["IN_hi"]
prefix = sys.argv[1]; nsamp = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000
gp = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
gq = np.fromfile(prefix + "_post.i32", dtype=np.int32)
rng = np.random.default_rng(0)
idx = rng.integers(0, len(gp), min(nsamp, len(gp)))
lo = OUT_lo[gp[idx]] & IN_lo[gq[idx]]
hi = OUT_hi[gp[idx]] & IN_hi[gq[idx]]
def popc(a):
    a = np.asarray(a, np.uint64)
    return np.array([int(x).bit_count() for x in a]) if a.ndim else int(a).bit_count()
ol = popc(lo) + popc(hi)
out = {"prefix": prefix, "n": int(len(idx)),
       "zero_overlap_frac": float((ol == 0).mean()),
       "overlap_mean": float(ol.mean()),
       "overlap_ge8_frac": float((ol >= 8).mean())}
print(json.dumps(out))
