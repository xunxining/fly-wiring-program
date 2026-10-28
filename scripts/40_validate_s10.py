#!/usr/bin/env python3
"""Stage 40: v4.1 S=10 验证 —— 度数 KS(分块) + 精确互惠性(对称哈希分桶)"""
import numpy as np
import json, os, gc

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
GEN = "/tmp/flydata/gen/genv4_S10"
NB, CH = 256, 10_000_000

def log(*a): print(*a, flush=True)

def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    q = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, q, side="right") / len(a)
                        - np.searchsorted(b, q, side="right") / len(b)).max())

def bincnt(path, n):
    mm = np.memmap(path, dtype=np.int32, mode="r")
    out = np.zeros(n, np.int64)
    for i0 in range(0, len(mm), 20_000_000):
        out += np.bincount(mm[i0:i0 + 20_000_000], minlength=n)
    return out

g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = g['type_count'].astype(np.int64)
N_gen = int((tc * 10).sum())
node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
tm = node_type >= 0

r_out = bincnt(OUT + "e_pre.i32", len(node_type))[tm]
r_in = bincnt(OUT + "e_post.i32", len(node_type))[tm]
g_out = bincnt(GEN + "_pre.i32", N_gen)
g_in = bincnt(GEN + "_post.i32", N_gen)
res = {"ks_out": ks(r_out, g_out), "ks_in": ks(r_in, g_in)}
log(f"S=10 degree KS: out={res['ks_out']:.4f} in={res['ks_in']:.4f}")
del r_out, r_in, g_out, g_in; gc.collect()

def recip_exact(pre_path, post_path, tag):
    pre = np.memmap(pre_path, dtype=np.int32, mode="r")
    post = np.memmap(post_path, dtype=np.int32, mode="r")
    E = len(pre)
    bdir = f"/tmp/flydata/rb_{tag}/"
    os.makedirs(bdir, exist_ok=True)
    fps = [open(bdir + f"{i}.i64", "wb") for i in range(NB)]
    n_noloop = 0
    for i0 in range(0, E, CH):
        i1 = min(i0 + CH, E)
        a = np.asarray(pre[i0:i1], dtype=np.int64)
        b = np.asarray(post[i0:i1], dtype=np.int64)
        keep = a != b
        a, b = a[keep], b[keep]
        n_noloop += len(a)
        mn = np.minimum(a, b); mx = np.maximum(a, b)
        bk = (mn * 2654435761 + mx * 40503) % NB
        key = a * (1 << 32) + b          # 有向键; 反向边因 (min,max) 相同必同桶
        for i in range(NB):
            m = bk == i
            if m.any():
                key[m].tofile(fps[i])
    for f in fps: f.close()
    n_rec = 0
    for i in range(NB):
        ks_ = np.fromfile(bdir + f"{i}.i64", dtype=np.int64)
        if not len(ks_): continue
        ku = np.unique(ks_)
        a = ks_ >> 32; b = ks_ & 0xFFFFFFFF
        rev = (b << 32) | a
        j = np.searchsorted(ku, rev)
        j = np.clip(j, 0, len(ku) - 1)
        n_rec += int((ku[j] == rev).sum())
    for i in range(NB): os.remove(bdir + f"{i}.i64")
    os.rmdir(bdir)
    return n_rec / max(n_noloop, 1)

res["recip_real"] = recip_exact(OUT + "e_pre.i32", OUT + "e_post.i32", "real")
log(f"recip real: {res['recip_real']:.4f}")
res["recip_v4_s10"] = recip_exact(GEN + "_pre.i32", GEN + "_post.i32", "s10v4")
log(f"recip v4 S=10: {res['recip_v4_s10']:.4f}")
json.dump(res, open(ROOT + "v4_s10_validation.json", "w"), indent=1)
log("DONE")
