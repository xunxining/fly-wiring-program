#!/usr/bin/env python3
# 67_dist_profile.py — 生成图的距离剖面评估 (§22 P1)
# 用法: python3 67_dist_profile.py <gen_prefix> <out_json>
# gen 节点序 = 基因组序 = xyz_gen.f32 序 (offsets 分组), 直接索引
import sys, json, time
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器/"
t0 = time.time()
prefix, out_json = sys.argv[1], sys.argv[2]

gd = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
tc = gd["type_count"].astype(np.int64)
N = int(tc.sum())
xyz = np.fromfile(ROOT + "xyz_gen.f32", dtype=np.float32).reshape(-1, 3)
cov = ~np.isnan(xyz[:, 0])
print(f"gen 节点 {N:,}, 坐标覆盖 {cov.mean():.3f}", flush=True)

pre = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
post = np.fromfile(prefix + "_post.i32", dtype=np.int32)
print(f"gen 边 {len(pre):,} ({time.time()-t0:.0f}s)", flush=True)

both = cov[pre] & cov[post]
d = np.linalg.norm(xyz[pre[both]].astype(np.float64) - xyz[post[both]].astype(np.float64), axis=1)
del pre, post, both, xyz, cov
import gc; gc.collect()

qs = [0.1, 0.25, 0.5, 0.75, 0.9]
rs = [1, 2, 5, 10, 20, 40, 80, 160]
qg = np.quantile(d, qs)
gen_prof = {r: float((d < r).mean()) for r in rs}
gen_med = float(np.median(d))
gen_sorted = np.sort(d[::7])
del d; gc.collect()
print(f"gen 剖面完成, 中位 {gen_med:.1f}um ({time.time()-t0:.0f}s)", flush=True)

# 真实剖面 (重算, 同口径)
sys.path.insert(0, ROOT + "vendor")
import pandas as pd
nt_full = np.fromfile(ROOT + "data_census/node_type.i32", dtype=np.int32)
tm = nt_full >= 0
remap = np.cumsum(tm, dtype=np.int64) - 1
epre = np.fromfile(ROOT + "data_census/e_pre.i32", dtype=np.int32)
epost = np.fromfile(ROOT + "data_census/e_post.i32", dtype=np.int32)
rp = remap[epre].astype(np.int64); rq = remap[epost].astype(np.int64)
ntc = nt_full[tm]
ann = pd.read_feather(ROOT + "data/body-annotations.feather", columns=["bodyId", "somaLocation"])
node_ids = np.fromfile(ROOT + "data_census/node_ids.i64", dtype=np.int64)
nid_valid = node_ids[tm]
sl = ann["somaLocation"]; has = sl.notna().to_numpy()
amp = pd.Series(np.arange(len(ann)), index=ann["bodyId"].values)
pos_idx = amp.reindex(nid_valid).to_numpy(); okn = ~np.isnan(pos_idx)
loc_arr = np.full((len(ann), 3), np.nan)
loc_arr[has] = np.vstack(sl[has].to_numpy())
xyzc = np.full((len(ntc), 3), np.nan)
xyzc[okn] = loc_arr[pos_idx[okn].astype(int)]
xyzc /= 1000.0
covc = ~np.isnan(xyzc[:, 0])
br = covc[rp] & covc[rq]
dr = np.linalg.norm(xyzc[rp[br]].astype(np.float64) - xyzc[rq[br]].astype(np.float64), axis=1)
del rp, rq, br, xyzc, covc, loc_arr, ann, sl, has, pos_idx, okn, amp, node_ids, nid_valid, nt_full, tm, remap
gc.collect()

prof = {"gen": gen_prof, "real": {}}
for r in rs:
    prof["real"][r] = float((dr < r).mean())
    print(f"  <{r}um: gen {prof['gen'][r]:.4f} 真实 {prof['real'][r]:.4f}", flush=True)

# KS 与分位数
qr = np.quantile(dr, qs)
grid = np.geomspace(0.5, 200, 200)
cr = np.searchsorted(np.sort(dr[::7]), grid) / len(dr[::7])
cg = np.searchsorted(gen_sorted, grid) / len(gen_sorted)
ks = float(np.abs(cg - cr).max())
res = {
    "gen_median": gen_med, "real_median": float(np.median(dr)),
    "gen_q": dict(zip(map(str, qs), qg.tolist())),
    "real_q": dict(zip(map(str, qs), qr.tolist())),
    "profile": prof, "dist_KS": ks,
    "gen_edges": int(len(pre)), "gen_both_coord": int(both.sum()),
}
json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
print(f"中位: gen {res['gen_median']:.1f} 真实 {res['real_median']:.1f} um, KS {ks:.4f}", flush=True)
print("DONE 67", flush=True)
