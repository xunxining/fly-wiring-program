#!/usr/bin/env python3
# 67_dist_gen.py — 生成图距离剖面 (仅 gen 侧, 真实侧读 real_dist.json)
# 用法: python3 67_dist_gen.py <gen_prefix> <out_json>
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

pre = np.fromfile(prefix + "_pre.i32", dtype=np.int32)
post = np.fromfile(prefix + "_post.i32", dtype=np.int32)
print(f"gen 边 {len(pre):,} ({time.time()-t0:.0f}s)", flush=True)
both = cov[pre] & cov[post]
d = np.linalg.norm(xyz[pre[both]].astype(np.float64) - xyz[post[both]].astype(np.float64), axis=1)
del pre, post, both, xyz, cov
import gc; gc.collect()

rs = [1, 2, 5, 10, 20, 40, 80, 160]
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
gen_prof = {str(r): float((d < r).mean()) for r in rs}
gen_q = dict(zip(map(str, qs), np.quantile(d, qs).tolist()))
gen_med = float(np.median(d))
gs = np.sort(d[::7])
del d; gc.collect()

real = json.load(open(ROOT + "real_dist.json"))
grid = np.geomspace(0.5, 200, 200)
cg = np.searchsorted(gs, grid) / len(gs)
cr = np.searchsorted(np.array(real["sorted_sub"]), grid) / len(real["sorted_sub"])
ks = float(np.abs(cg - cr).max())
res = {"gen_median": gen_med, "real_median": real["median"],
       "gen_q": gen_q, "real_q": real["q"],
       "profile": {"gen": gen_prof, "real": real["profile"]},
       "dist_KS": ks}
json.dump(res, open(out_json, "w"), ensure_ascii=False, indent=1)
for r in rs:
    print(f"  <{r}um: gen {gen_prof[str(r)]:.4f} 真实 {real['profile'][str(r)]:.4f}", flush=True)
print(f"中位: gen {gen_med:.1f} 真实 {real['median']:.1f} um, KS {ks:.4f}", flush=True)
print("DONE 67lite", flush=True)
