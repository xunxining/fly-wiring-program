#!/usr/bin/env python3
"""102_male_prep.py — 雄侧全栈移植 M3: 类型排序重排 + 全资产重建
雌性生成器约定: 节点按类型分组排序 (toff[t]+p 索引), 只含有类型节点。
雄性原始 (data_census): 211,577 节点按 bodyId 排序, 含 47,071 无类型节点。
产物 (data_male/):
  node_ids.i64 node_type.i32 node_sc.i32 e_pre/e_post/e_w.i32 (重映射)
  xyz.f32 (somaLocation 体素×0.008→µm, 缺=NaN)
  genome_m.npz cq_cell/cq_t/cq_cnt.i32 act_in.f32
  np_masks_m2.npz (3×uint64 位打包) dyad_m.npz dyad_cap_m.npz prep_m_diag.json
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, json, time, gc, os
import pandas as pd
R = "/mnt/agents/output/果蝇生成器/"
DC = R + "data_census/"; DM = "/tmp/data_male/"
os.makedirs(DM, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
def rss(): return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])//1024

node_ids = np.fromfile(DC+"node_ids.i64", dtype=np.int64)
node_type = np.fromfile(DC+"node_type.i32", dtype=np.int32)
node_sc = np.fromfile(DC+"node_sc.i32", dtype=np.int32)
ep = np.fromfile(DC+"e_pre.i32", dtype=np.int32)
eq = np.fromfile(DC+"e_post.i32", dtype=np.int32)
ew = np.fromfile(DC+"e_w.i32", dtype=np.int32)
N0 = len(node_ids); E = len(ep)
typed = np.where(node_type >= 0)[0]
N = len(typed)
log(f"原始 N0={N0:,} typed N={N:,} E={E:,}")

# ---- 类型排序重排 ----
tt = node_type[typed]
o = np.argsort(tt, kind="stable")
new_of_old = np.full(N0, -1, np.int32)
new_of_old[typed[o]] = np.arange(N)
nt_new = tt[o]
sc_new = node_sc[typed[o]]
nid_new = node_ids[typed[o]]
T = int(nt_new.max())+1
tc = np.bincount(nt_new, minlength=T)
assert (np.diff(nt_new) >= 0).all()
nt_new.tofile(DM+"node_type.i32"); sc_new.tofile(DM+"node_sc.i32"); nid_new.tofile(DM+"node_ids.i64")
np.save(DM+"sc_names_m.npy", np.array(list(np.load(R+"基因组_distribs.npz", allow_pickle=True)["sc_types"]), dtype=object))
log(f"T={T:,} 重排完成 rss={rss()}MB")

# ---- 边重映射 ----
epn = new_of_old[ep]; eqn = new_of_old[eq]
assert (epn >= 0).all() and (eqn >= 0).all()
epn.tofile(DM+"e_pre.i32"); eqn.tofile(DM+"e_post.i32"); ew.tofile(DM+"e_w.i32")
del ep, eq, new_of_old; gc.collect()

# ---- 基因组 ----
k = nt_new[epn].astype(np.int64)*T + nt_new[eqn]
uk, cnt = np.unique(k, return_counts=True)
np.savez(DM+"genome_m.npz", uk=uk, cnt=cnt, type_count=tc,
         sc_types=np.load(R+"基因组_distribs.npz", allow_pickle=True)["sc_types"])
log(f"基因组对 {len(uk):,} 覆盖 {cnt.sum()/E:.4f}"); del k; gc.collect()

# ---- CELLQ ----
kc = epn.astype(np.int64)*T + nt_new[eqn]
uq, cc = np.unique(kc, return_counts=True)
assert cc.sum() == E
(uq//T).astype(np.int32).tofile(DM+"cq_cell.i32")
(uq%T).astype(np.int32).tofile(DM+"cq_t.i32")
cc.astype(np.int32).tofile(DM+"cq_cnt.i32")
log(f"CELLQ {len(uq):,} rss={rss()}MB"); del kc, uq, cc; gc.collect()

# ---- ACT ----
odr = np.bincount(epn, minlength=N).astype(np.float64)
idr = np.bincount(eqn, minlength=N).astype(np.float64)
tm_o = np.maximum(np.bincount(nt_new, weights=odr, minlength=T)/np.maximum(tc,1), 1e-9)
tm_i = np.maximum(np.bincount(nt_new, weights=idr, minlength=T)/np.maximum(tc,1), 1e-9)
a_in = np.clip(idr/tm_i[nt_new], 0.1, 10.0)
a_in.astype(np.float32).tofile(DM+"act_in.f32")
log(f"ACT a_in 中位 {np.median(a_in):.2f}")
del odr, idr, tm_o, tm_i, a_in; gc.collect()

# ---- xyz: somaLocation 体素->µm ----
ann = pd.read_feather(R+"data/body-annotations.feather", columns=["bodyId","somaLocation"])
ann = ann.sort_values("bodyId").reset_index(drop=True)
assert (ann.bodyId.values == node_ids).all()
sl = ann["somaLocation"].to_numpy()
xyz0 = np.full((N0,3), np.nan, np.float32)
has = np.array([v is not None and not (isinstance(v,float) and np.isnan(v)) for v in sl])
xyz0[has] = np.stack([np.asarray(v, dtype=np.float32) for v in sl[has]]) * 0.008  # 8nm voxels
xyz_new = xyz0[typed[o]]
xyz_new.tofile(DM+"xyz.f32")
log(f"xyz 覆盖 {np.isfinite(xyz_new[:,0]).mean():.3f}, 范围 {np.nanmin(xyz_new):.0f}..{np.nanmax(xyz_new):.0f} µm")
del ann, sl, xyz0, xyz_new; gc.collect()

# ---- NPOV 掩码: 146 ROI -> 3×uint64 ----
nm = np.load(DC+"np_masks_male.npz")
OUTm = nm["OUT"][typed[o]]; INm = nm["IN"][typed[o]]
K = OUTm.shape[1]; assert K <= 192
def pack(M):
    W = np.zeros((len(M),3), np.uint64)
    for b in range(K):
        w = b//64; sh = b%64
        W[:,w] |= M[:,b].astype(np.uint64) << np.uint64(sh)
    return W
OW, IW = pack(OUTm), pack(INm)
np.savez(DM+"np_masks_m2.npz", OUT=OW, IN=IW, K=K)
log(f"NPOV 位打包 {OW.shape} K={K}"); del nm, OUTm, INm, OW, IW; gc.collect()

# ---- DYAD: dyad_m.npz (uk/E/R) ----
key = epn.astype(np.int64)*N + eqn
ks = np.sort(key)
pos = np.minimum(np.searchsorted(ks, eqn.astype(np.int64)*N+epn), E-1)
is_mut = ks[pos] == (eqn.astype(np.int64)*N+epn)
recip = float(is_mut.mean())
pair = nt_new[epn].astype(np.int64)*T + nt_new[eqn]
up, cntp = np.unique(pair, return_counts=True)
upm, cmut = np.unique(pair[is_mut], return_counts=True)
R_al = np.zeros(len(up), np.int64); R_al[np.searchsorted(up, upm)] = cmut
np.savez_compressed(DM+"dyad_m.npz", uk=up.astype(np.int64), E=cntp.astype(np.int32), R=R_al.astype(np.int32))
log(f"dyad_m: recip_real={recip:.4f} 对 {len(up):,}")
del key, ks, pos, is_mut, pair, up, cntp, upm, cmut, R_al; gc.collect()

# ---- DYAD_CAP: dyad_cap_m.npz (uk/E/cap) ----
cq_cell = np.fromfile(DM+"cq_cell.i32", dtype=np.int32).astype(np.int64)
cq_t = np.fromfile(DM+"cq_t.i32", dtype=np.int32).astype(np.int64)
ck = np.sort(cq_cell*T + cq_t)
del cq_cell, cq_t; gc.collect()
si_ = nt_new[epn].astype(np.int64); ti_ = nt_new[eqn].astype(np.int64)
key2 = eqn.astype(np.int64)*T + si_
pos = np.minimum(np.searchsorted(ck, key2), len(ck)-1)
capable = ck[pos] == key2
del key2, pos, ck; gc.collect()
pair = si_*T + ti_
up, cntp = np.unique(pair, return_counts=True)
upc, ccap = np.unique(pair[capable], return_counts=True)
cap_al = np.zeros(len(up), np.int64); cap_al[np.searchsorted(up, upc)] = ccap
np.savez_compressed(DM+"dyad_cap_m.npz", uk=up.astype(np.int64), E=cntp.astype(np.int32),
                    cap=cap_al.astype(np.int32))
log(f"dyad_cap_m: cap_frac 全局 {cap_al.sum()/E:.4f}")

diag = dict(N0=int(N0), N=int(N), T=int(T), E=int(E), recip_real=recip,
            genome_pairs=int(len(uk)), cap_frac=float(cap_al.sum()/E))
json.dump(diag, open(DM+"prep_m_diag.json","w"), indent=1)
log(f"DONE rss={rss()}MB")

# ---- 拷贝到门户 (健壮: 12 次重试 + 尺寸校验 + 截断重拷) ----
import shutil
PDM = "/mnt/agents/output/果蝇生成器/data_male/"
def portal_copy(f):
    src = DM+f; dst = PDM+f; want = os.path.getsize(src)
    for att in range(12):
        try:
            os.makedirs(PDM, exist_ok=True)
            if os.path.exists(dst) and os.path.getsize(dst) == want:
                print("portal ok:", f, want, flush=True); return True
            shutil.copyfile(src, dst)
            if os.path.getsize(dst) == want:
                print("portal ok:", f, want, flush=True); return True
        except OSError as e:
            print(f"retry{att}", f, str(e)[:60], flush=True)
        time.sleep(20)
    return False
# 小文件优先, 大文件最后
files = sorted(os.listdir(DM), key=lambda f: os.path.getsize(DM+f))
bad = [f for f in files if not portal_copy(f)]
print("ALL COPIED" if not bad else f"FAILED: {bad}", flush=True)
