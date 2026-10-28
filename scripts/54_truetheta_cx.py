#!/usr/bin/env python3
"""Stage 54: 真 θ 下的 CX R² —— 区分"放置毁环" vs "θ恢复毁环"
gen 图用生成器导出的真 θ (cxtheta.npz); real 用 theta_learned.npy (§12 拟合值, 上限参考)
"""
import numpy as np
import json, gc

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]

def build_K(pre, post, ntype):
    tn = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
    cx_idx = [tn.index(x) for x in CX]
    sel = np.isin(ntype[pre], cx_idx) & np.isin(ntype[post], cx_idx)
    cp, cq = pre[sel], post[sel]
    cells = np.unique(np.r_[cp, cq])
    bidx = np.full(len(ntype), -1, np.int64)
    bidx[cells] = np.arange(len(cells))
    ctname = np.array([tn[ntype[b]] for b in cells])
    W = np.zeros((len(cells), len(cells)), np.float32)
    np.add.at(W, (bidx[cq], bidx[cp]), 1)
    epg = np.where(ctname == "EPG")[0]
    K = W[np.ix_(epg, epg)].copy()
    for x in ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]:
        mid = np.where(ctname == x)[0]
        if len(mid):
            K += W[np.ix_(epg, mid)] @ W[mid][:, epg]
    return K, cells, epg, ctname

def fit_r2(K, th):
    d = th[:, None] - th[None, :]
    y = K.ravel()
    s = K.sum(1)
    r1 = np.outer(s, s).ravel(); r1 /= (r1.mean() + 1e-9)
    cosf = np.cos(d).ravel(); one = np.ones_like(y)
    def f(X):
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return (float(f(np.stack([one, cosf], -1))),
            float(f(np.stack([one, r1, cosf], -1))))

res = {}
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = g['type_count'].astype(np.int64)
gen_type = np.repeat(np.arange(len(tc), dtype=np.int32), tc)
toff = np.zeros(len(tc), np.int64); toff[1:] = np.cumsum(tc)[:-1]

import os
# 2026-10-22: gen 支路产物在 /tmp/flydata/gen (非持久, 已失) —— 缺失时保留既有 json 中的旧值
_old = json.load(open(ROOT + "truetheta_cx.json")) if os.path.exists(ROOT + "truetheta_cx.json") else {}
res.update({k: v for k, v in _old.items()})
for tag, path in [("v4.1", "/tmp/flydata/gen/genv4_S1"), ("v4.2", "/tmp/flydata/gen/genv42_S1")]:
    if not os.path.exists(path + "_pre.i32"):
        print(tag, ": gen products unavailable, keeping previous values", flush=True)
        continue
    pre = np.fromfile(path + "_pre.i32", dtype=np.int32)
    post = np.fromfile(path + "_post.i32", dtype=np.int32)
    K, cells, epg, ctname = build_K(pre, post, gen_type)
    # 真 θ: cxtheta.npz 按类型拼接, 节点 -> 类型内序号
    cz = np.load(path + "_cxtheta.npz")
    th_node = np.full(len(gen_type), np.nan)
    p = 0
    for k, ty in enumerate(cz["tidx"]):
        n = int(tc[ty])
        th_node[toff[ty]:toff[ty] + n] = cz["theta"][p:p + n]
        p += n
    epg_nodes = cells[epg]
    th_true = th_node[epg_nodes]
    assert not np.isnan(th_true).any(), "missing true theta"
    r2t, r2tf = fit_r2(K, th_true)
    res[tag] = {"r2_true_theta": r2t, "r2_true_theta_full": r2tf}
    print(tag, ":", res[tag], flush=True)
    del pre, post, K; gc.collect()

# real 参考: theta_learned (§12 拟合) 作为其自身 K 的最优 θ (上限)
node_type = np.fromfile(OUT + "node_type.i32", dtype=np.int32)
tm = node_type >= 0
remap = np.cumsum(tm, dtype=np.int64) - 1
pre = remap[np.fromfile(OUT + "e_pre.i32", dtype=np.int32)]
post = remap[np.fromfile(OUT + "e_post.i32", dtype=np.int32)]
nt = node_type[tm]
K, cells, epg, ctname = build_K(pre, post, nt)
import pandas as pd
cx_cells = pd.read_csv(ROOT + "cx_cells.csv") if False else None
th_learned = np.load(ROOT + "theta_learned.npy")
# theta_learned 的顺序 = cx_cells 表顺序 (按类型); 需要与 cells 对齐:
# cx_cells 含 (id, type); 建立 real typed id -> learned theta
# 2026-10-22 修复: 原 cc["id"] 但 cx_cells.csv 列名为 bodyId,
# KeyError 被 except Exception 吞掉, real 支路从未产出 (truetheta_cx.json 缺 real_fitted_theta)。
# 修正列名; except 收窄为 FileNotFoundError —— 映射失败必须显式报错。
try:
    cc = pd.read_csv(ROOT + "data_cx/cx_cells.csv")
except FileNotFoundError as ex:
    print("real fitted-theta eval skipped:", ex)
else:
    th_map = dict(zip(cc["bodyId"].to_numpy(), th_learned))
    # cells 是 typed 子图连续索引: typed 索引 -> 全图索引 -> bodyId 两跳映射
    node_ids = np.fromfile(OUT + "node_ids.i64", dtype=np.int64)
    full_idx_of_typed = np.flatnonzero(node_type >= 0)
    epg_nodes = cells[epg]
    th_real = np.array([th_map.get(int(node_ids[full_idx_of_typed[int(x)]]), np.nan)
                        for x in epg_nodes])
    if np.isnan(th_real).any():
        raise ValueError("unmapped EPG nodes in theta alignment")
    r2t, r2tf = fit_r2(K, th_real)
    res["real_fitted_theta"] = {"r2_true_theta": r2t, "r2_true_theta_full": r2tf}
    print("real(fitted θ):", res["real_fitted_theta"], flush=True)

json.dump(res, open(ROOT + "truetheta_cx.json", "w"), indent=1)
print("SAVED truetheta_cx.json")
