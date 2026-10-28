#!/usr/bin/env python3
# 119_rst_guard.py — §35 guard: 类型对互惠剖面 Spearman (real vs repaired gen)
# f_{s→t} = 互惠有向边数(s→t) / 总边数(s→t), 仅取 E_st >= 500 的有向类型对
# 用法: python3 119_rst_guard.py female gen_edges/g_mutq  ->  data_female/rst_mutq.json
import json, sys, time
import numpy as np
from scipy.stats import spearmanr

ROOT = "/mnt/agents/output/果蝇生成器"
SEX, GEN = sys.argv[1], sys.argv[2]
D = f"{ROOT}/data_{'female' if SEX == 'female' else 'male'}"
OUT = f"{D}/rst_{'mutq' if SEX == 'female' else 'mutqm'}.json"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

nt = np.fromfile(f"{D}/node_type.i32", dtype=np.int32)
N = int(nt.size)

def pair_profile(pre, post):
    E = pre.size
    kk = np.unique(pre.astype(np.int64) * N + post)
    rev = post.astype(np.int64) * N + pre
    p = np.clip(np.searchsorted(kk, rev), 0, kk.size - 1)
    mut = kk[p] == rev
    pk = nt[pre].astype(np.int64) * 65536 + nt[post]
    up, inv = np.unique(pk, return_inverse=True)
    tot = np.bincount(inv, minlength=up.size)
    mtc = np.bincount(inv[mut], minlength=up.size)
    return up, tot, mtc

rp = np.fromfile(f"{D}/e_pre.i32", dtype=np.int32)
rq = np.fromfile(f"{D}/e_post.i32", dtype=np.int32)
up_r, tot_r, mut_r = pair_profile(rp, rq)
log(f"real pairs={up_r.size:,}")
del rp, rq
gp = np.fromfile(f"{ROOT}/{GEN}_pre.i32", dtype=np.int32)
gq = np.fromfile(f"{ROOT}/{GEN}_post.i32", dtype=np.int32)
up_g, tot_g, mut_g = pair_profile(gp, gq)
log(f"gen pairs={up_g.size:,}")
del gp, gq

common, ir, ig = np.intersect1d(up_r, up_g, return_indices=True)
keep = (tot_r[ir] >= 500) & (tot_g[ig] >= 500)
fr = mut_r[ir][keep] / tot_r[ir][keep]
fg = mut_g[ig][keep] / tot_g[ig][keep]
rho, pval = spearmanr(fr, fg)
log(f"common pairs(E>=500)={keep.sum():,}  Spearman rho={rho:.4f} (p={pval:.2e})")
res = dict(sex=SEX, gen=GEN, pairs=int(keep.sum()), spearman=round(float(rho), 4),
           p=float(pval), guard=">=0.85", passed=bool(rho >= 0.85),
           ts=time.strftime("%F %T"))
with open(OUT, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log("saved", OUT)
