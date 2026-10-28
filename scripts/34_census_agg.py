#!/usr/bin/env python3
"""Stage 34 v2: 普查聚合 —— 784 通道地图 + 三锚点 + 形态分类
记录字段: s,t,m,n,E,A,lift,phi,z_A,Rth_pre,Rth_post,r_side,sig2
hit      : z_A>4 且 lift>0.05  (置换检验 + 效应量)
ring_like: hit 且两侧相位在圆周上分散 (Rθ<0.5) —— 真环形/连续坐标
block    : hit 且相位聚集 —— 块/双侧等离散坐标
bilateral: hit 且 |corr(cosθ, side)|>0.5 —— 左右半球结构
"""
import numpy as np
import pandas as pd
import json, re

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
NF, NSC = 13, 28

def log(*a): print(*a, flush=True)

R = np.load(OUT + "pair_records.npy").astype(np.float32).reshape(-1, NF)
log("records:", len(R))
s = R[:, 0].astype(np.int32); t = R[:, 1].astype(np.int32)
m, n, E = R[:, 2], R[:, 3], R[:, 4]
A, lift, phi, zA = R[:, 5], R[:, 6], R[:, 7], R[:, 8]
Rp, Rq, rside, sig2 = R[:, 9], R[:, 10], R[:, 11], R[:, 12]

ann = pd.read_feather(ROOT + "data/body-annotations.feather", columns=["type", "superclass"])
g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
sc_types = list(g["sc_types"]); sc2i = {x: i for i, x in enumerate(sc_types)}
tn = np.load(OUT + "type_names.npy", allow_pickle=True)
tn_str = [str(x) for x in tn]; T = len(tn)
ann2 = ann.dropna(subset=["type"]).copy()
ann2["sc"] = ann2["superclass"].fillna("UNK").map(sc2i)
type_sc = np.full(T, sc2i["UNK"], dtype=np.int32)
for ti, name in enumerate(tn_str):
    sub = ann2.loc[ann2["type"] == name, "sc"]
    if len(sub): type_sc[ti] = sub.mode().iloc[0]
del ann, ann2

ch_pre, ch_post = type_sc[s], type_sc[t]
ch_key = ch_pre * NSC + ch_post

hit = (zA > 4) & (lift > 0.05)
ring = hit & (Rp < 0.5) & (Rq < 0.5)
block = hit & ~((Rp < 0.5) & (Rq < 0.5))
bilat = hit & (np.abs(rside) > 0.5)
log(f"tested={len(R)}  edge_cov={100*E.sum()/25135616:.1f}%  "
    f"hits={int(hit.sum())} ({100*hit.mean():.1f}%)  ring={int(ring.sum())} block={int(block.sum())} bilat={int(bilat.sum())}")

np.savez(OUT + "census_pairs.npz", s=s, t=t, m=m, n=n, E=E, A=A, lift=lift, phi=phi,
         zA=zA, Rp=Rp, Rq=Rq, rside=rside, sig2=sig2,
         hit=hit, ring=ring, block=block, bilat=bilat,
         ch_pre=ch_pre, ch_post=ch_post)

# ---------- 784 通道 ----------
M = {k: np.zeros(NSC * NSC) for k in ["lift", "A"]}
C = {k: np.zeros(NSC * NSC, np.int64) for k in ["tested", "edges", "hits", "ring", "bilat"]}
for k in range(NSC * NSC):
    mk = ch_key == k
    if not mk.any(): continue
    w = E[mk]
    C["tested"][k] = mk.sum(); C["edges"][k] = int(w.sum())
    C["hits"][k] = hit[mk].sum(); C["ring"][k] = ring[mk].sum(); C["bilat"][k] = bilat[mk].sum()
    M["lift"][k] = (lift[mk] * w).sum() / w.sum()
    M["A"][k] = (A[mk] * w).sum() / w.sum()
np.savez(OUT + "census_channels.npz",
         mean_lift=M["lift"].reshape(NSC, NSC), mean_A=M["A"].reshape(NSC, NSC),
         tested=C["tested"].reshape(NSC, NSC), edges=C["edges"].reshape(NSC, NSC),
         nhits=C["hits"].reshape(NSC, NSC), nring=C["ring"].reshape(NSC, NSC),
         nbilat=C["bilat"].reshape(NSC, NSC), sc_types=np.array(sc_types, dtype=object))

# ---------- 锚点 ----------
CX = {"EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"}
is_cx = np.array([x in CX for x in tn_str])
is_kc = np.array([x.startswith("KC") for x in tn_str])
is_t45 = np.array([re.fullmatch(r"T[45][a-d]", x) is not None for x in tn_str])
is_pn = np.array([x.endswith("PN") for x in tn_str])
is_mbon = np.array([x.startswith("MBON") for x in tn_str])
ALL = np.ones(T, bool)

def anchor(ms, mt, name, topn=10):
    mk = ms[s] & mt[t]
    sub = np.where(mk)[0]
    if not len(sub): return {"name": name, "n_pairs": 0}
    w = E[sub]
    o = sub[np.argsort(lift[sub])[::-1]]
    return {"name": name, "n_pairs": int(len(sub)), "edges": int(w.sum()),
            "wmean_lift": float((lift[sub] * w).sum() / w.sum()),
            "wmean_A": float((A[sub] * w).sum() / w.sum()),
            "hit_rate": float(hit[sub].mean()), "ring_rate": float(ring[sub].mean()),
            "top": [{"pair": f"{tn_str[s[i]]} -> {tn_str[t[i]]}", "E": int(E[i]),
                     "A": round(float(A[i]), 3), "lift": round(float(lift[i]), 3),
                     "z": round(float(zA[i]), 1),
                     "form": "ring" if ring[i] else ("block" if block[i] else "-")}
                    for i in o[:topn]]}

anchors = [anchor(is_cx, is_cx, "CX 环形吸引子 (预期高, ring)"),
           anchor(is_t45, ALL, "T4/T5 出边 (视觉视网膜定位, 预期高)"),
           anchor(ALL, is_t45, "T4/T5 入边 (视觉, 预期高)"),
           anchor(is_pn, is_kc, "PN -> KC (随机投影, 预期≈0)"),
           anchor(is_kc, is_kc, "KC -> KC (脚纤维束分层, 预期高)"),
           anchor(is_kc, is_mbon, "KC -> MBON (预期低)")]
json.dump({"n_records": int(len(R)), "n_hits": int(hit.sum()),
           "n_ring": int(ring.sum()), "n_bilat": int(bilat.sum()),
           "anchors": anchors},
          open(OUT + "census_anchors.json", "w"), ensure_ascii=False, indent=1)
for a in anchors:
    log(f"{a['name']}: pairs={a.get('n_pairs')}, lift={a.get('wmean_lift', 0):.3f}, "
        f"A={a.get('wmean_A', 0):.3f}, hit={a.get('hit_rate', 0):.2f}, ring={a.get('ring_rate', 0):.2f}")
log("AGG DONE")
