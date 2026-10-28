#!/usr/bin/env python3
# 69_cellcv.py — 逐细胞度配额评估 (§23): 真实 vs gen 逐型度 CV, EPG CX度 CV
# 用法: python3 69_cellcv.py <gen_prefix> <out_json>
import sys, json, time
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器/"
t0 = time.time()
prefix, out_json = sys.argv[1], sys.argv[2]

gd = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = gd["type_count"].astype(np.int64); T = len(tc)
toff = np.concatenate([[0], np.cumsum(tc)])
gtype = np.repeat(np.arange(T), tc)
N = int(tc.sum())

# gen 度 (基因组序 == gen 节点序)
pre = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
post = np.fromfile(prefix + "_post.i32", dtype=np.int32)
odg = np.bincount(pre, minlength=N); idg = np.bincount(post, minlength=N)
del pre, post
import gc; gc.collect()

# 真实度 (紧凑序 -> 基因组序对齐)
nt_full = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
tm = nt_full >= 0
ntc = nt_full[tm].astype(np.int64)
remap = np.cumsum(tm, dtype=np.int64) - 1
ep = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
eq = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
odr_c = np.bincount(remap[ep], minlength=N); idr_c = np.bincount(remap[eq], minlength=N)
del ep, eq; gc.collect()
perm = np.argsort(ntc, kind="stable")   # 基因组序 -> 紧凑序
odr = odr_c[perm]; idr = idr_c[perm]
assert (np.bincount(ntc, minlength=T) == tc).all()

rows = []
for t in range(T):
    if tc[t] < 3: continue
    seg = slice(toff[t], toff[t + 1])
    for nm, dr, dgg in [("out", odr, odg), ("in", idr, idg)]:
        a = dr[seg].astype(float); b = dgg[seg].astype(float)
        if a.mean() > 5:
            rows.append((t, nm, float(a.mean()),
                         float(a.std() / a.mean()), float(b.std() / max(b.mean(), 1e-9))))
print(f"逐型 CV 表 {len(rows):,} 行 ({time.time()-t0:.0f}s)", flush=True)

cv_r = np.array([r[3] for r in rows]); cv_g = np.array([r[4] for r in rows])
mn = np.array([r[2] for r in rows]); poi = 1 / np.sqrt(mn)
res = {
    "n_rows": len(rows),
    "cv_real_median": float(np.median(cv_r)), "cv_gen_median": float(np.median(cv_g)),
    "cv_poisson_median": float(np.median(poi)),
    "ratio_real_median": float(np.median(cv_r / poi)),
    "ratio_gen_median": float(np.median(cv_g / poi)),
    "cv_corr": float(np.corrcoef(cv_r, cv_g)[0, 1]),
}

# EPG 专项: 全度 CV + CX 度 CV (对 CX 型目标的出度)
tn = np.load(ROOT + "data_census/type_names.npy", allow_pickle=True)
cxnames = ["Delta7", "EPG", "EPGt", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]
cxset = {i for i, n in enumerate(tn) if n in cxnames}
epg = [i for i, n in enumerate(tn) if n == "EPG"][0]
seg = slice(toff[epg], toff[epg + 1])
res["EPG"] = {"out_cv_real": float(odr[seg].std() / odr[seg].mean()),
              "out_cv_gen": float(odg[seg].std() / odg[seg].mean()),
              "n": int(tc[epg])}
# EPG -> CX 目标出度 (58 口径)
pre = np.fromfile(prefix + "_pre.i32", dtype=np.int32).astype(np.int64)
post = np.fromfile(prefix + "_post.i32", dtype=np.int32).astype(np.int64)
iscx_t = np.array([t in cxset for t in gtype])
mg = iscx_t[post] & (gtype[pre] == epg)
deg_cx_g = np.bincount(pre[mg] - toff[epg], minlength=int(tc[epg])).astype(float)
ep2 = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
eq2 = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
rp = remap[ep2].astype(np.int64); rq = remap[eq2].astype(np.int64)
mr = (ntc[rp] == epg) & np.isin(ntc[rq], list(cxset))
cells = np.flatnonzero(ntc == epg)
pos = np.searchsorted(cells, rp[mr])
deg_cx_r = np.bincount(pos, minlength=int(tc[epg])).astype(float)
res["EPG_CX"] = {"cv_real": float(deg_cx_r.std() / deg_cx_r.mean()),
                 "cv_gen": float(deg_cx_g.std() / deg_cx_g.mean()),
                 "mean_real": float(deg_cx_r.mean()), "mean_gen": float(deg_cx_g.mean())}
json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
print(f"CV 中位: 真实 {res['cv_real_median']:.3f} gen {res['cv_gen_median']:.3f} 泊松 {res['cv_poisson_median']:.3f}", flush=True)
print(f"比值: 真实/泊松 {res['ratio_real_median']:.2f} gen/泊松 {res['ratio_gen_median']:.2f}, CV 相关 {res['cv_corr']:.3f}", flush=True)
print(f"EPG 出度CV: 真实 {res['EPG']['out_cv_real']:.3f} gen {res['EPG']['out_cv_gen']:.3f}", flush=True)
print(f"EPG CX度CV: 真实 {res['EPG_CX']['cv_real']:.3f} gen {res['EPG_CX']['cv_gen']:.3f}", flush=True)
print("DONE 69", flush=True)
