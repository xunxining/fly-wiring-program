#!/usr/bin/env python3
"""Stage 55: CX 有效核逐项分解 —— 定位 v4.2 环幅丢失在哪一项
对 direct EPG-EPG 与每条两跳路径分别算 真θ cos R²
"""
import numpy as np
import json, gc

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
CX = ["EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"]
HOPS = ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]

def build_W(pre, post, ntype):
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
    return W, cells, ctname

def r2_cos(M, th):
    d = th[:, None] - th[None, :]
    y = M.ravel()
    X = np.stack([np.ones_like(y), np.cos(d).ravel()], -1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / sst) if sst > 0 else float("nan")

res = {}
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = g['type_count'].astype(np.int64)
gen_type = np.repeat(np.arange(len(tc), dtype=np.int32), tc)
toff = np.zeros(len(tc), np.int64); toff[1:] = np.cumsum(tc)[:-1]

for tag, path in [("v4.1", "/tmp/flydata/gen/genv4_S1"), ("v4.2", "/tmp/flydata/gen/genv42_S1")]:
    pre = np.fromfile(path + "_pre.i32", dtype=np.int32)
    post = np.fromfile(path + "_post.i32", dtype=np.int32)
    W, cells, ctname = build_W(pre, post, gen_type)
    cz = np.load(path + "_cxtheta.npz")
    th_node = np.full(len(gen_type), np.nan)
    p = 0
    for k, ty in enumerate(cz["tidx"]):
        n = int(tc[ty])
        th_node[toff[ty]:toff[ty] + n] = cz["theta"][p:p + n]
        p += n
    th = th_node[cells]
    epg = np.where(ctname == "EPG")[0]
    th_e = th[epg]
    terms = {}
    Ksum = W[np.ix_(epg, epg)].copy()
    terms["direct_EPG_EPG"] = r2_cos(Ksum, th_e)
    for x in HOPS:
        mid = np.where(ctname == x)[0]
        if not len(mid):
            continue
        T2 = W[np.ix_(epg, mid)] @ W[mid][:, epg]
        Ksum += T2
        terms[f"2hop_via_{x}"] = r2_cos(T2, th_e)
    terms["K_total"] = r2_cos(Ksum, th_e)
    # 额外: 互反程度
    M = W[np.ix_(epg, epg)]
    terms["_recip_direct"] = float((np.minimum(M, M.T) > 0).sum() / max((M > 0).sum(), 1))
    res[tag] = terms
    print(tag, json.dumps(terms, indent=1), flush=True)
    del pre, post, W; gc.collect()

json.dump(res, open(ROOT + "cxterms.json", "w"), indent=1)
print("SAVED cxterms.json")
