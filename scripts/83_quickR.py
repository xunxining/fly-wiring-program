#!/usr/bin/env python3
"""83_quickR.py — 快速互惠率: 生成图 recip vs 实值。用法: python3 83_quickR.py <prefix> [out.json]"""
import numpy as np, sys, json
N = 138327
D = "/mnt/agents/output/果蝇生成器/data_female/"
def recip(f1, f2):
    a = np.fromfile(f1, dtype=np.int32).astype(np.int64)
    b = np.fromfile(f2, dtype=np.int32).astype(np.int64)
    k = np.sort(a * N + b); r = b * N + a
    pos = np.minimum(np.searchsorted(k, r), len(k) - 1)
    return float((k[pos] == r).mean())
out = {"recip_gen": recip(sys.argv[1] + "_pre.i32", sys.argv[1] + "_post.i32"),
       "recip_real": recip(D + "e_pre.i32", D + "e_post.i32")}
print(json.dumps(out))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
