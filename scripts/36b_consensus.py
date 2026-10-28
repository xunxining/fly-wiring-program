#!/usr/bin/env python3
"""Stage 36b: 类型级相位共识 —— 让同一实例在所有核对的相位一致 (修复跨对相位乱序)
流程: 每个类型按边数降序处理其相位池; 每池与当前共识在共享实例上做圆对齐
      (旋转 ± 可选翻转取 |M| 大者); |M|<0.3 的池判定坐标系不兼容, 保持对特异
输出 data_census/consensus.npz:
  ctype/t_off/t_ids/t_theta/t_deg    类型共识池 (实例id, 相位, 度倾向)
  pair_use_cons/pair_phi_new/pair_q  每核对: 是否用共识, 共识系下的 φ, 对齐质量
"""
import numpy as np
import json
from collections import defaultdict

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
ALIGN_MIN = 0.3

def log(*a): print(*a, flush=True)

_pk = np.load(OUT + "poskernel_cal.npz")
pk = {key: _pk[key] for key in _pk.files}   # 一次性物化, 避免 npz 惰性读取钉住全数组
K = len(pk['s'])
E_map = {}
d = np.load(OUT + "census_pairs.npz")
for s, t, e in zip(d['s'][d['hit']], d['t'][d['hit']], d['E'][d['hit']]):
    E_map[(int(s), int(t))] = float(e)

# 收集每个类型的池: (pair_k, side 0=pre/1=post, ids, theta, deg)
pools = defaultdict(list)
for k in range(K):
    o0, o1 = int(pk['th_off'][k]), int(pk['th_off'][k + 1])
    ns = int(pk['ns'][k])
    ids = pk['ids'][o0:o1]; th = pk['theta'][o0:o1]; dg = pk['deg'][o0:o1]
    s, t = int(pk['s'][k]), int(pk['t'][k])
    pools[s].append([k, 0, ids[:ns], th[:ns], dg[:ns]])
    if pk['is_self'][k]:
        pass                      # 自对单池已计
    else:
        pools[t].append([k, 1, ids[ns:], th[ns:], dg[ns:]])

ctype = sorted(pools.keys())
t_ids_l, t_th_l, t_dg_l, t_off = [], [], [], [0]
pair_rot = {}        # (k, side) -> (flip, rot, |M|)
for ty in ctype:
    pl = sorted(pools[ty], key=lambda x: -E_map[(int(pk['s'][x[0]]), int(pk['t'][x[0]]))])
    cs = {}          # node -> [sx, sy, cnt, degsum]
    for k, side, ids, th, dg in pl:
        shared = [i for i, nid in enumerate(ids) if nid in cs]
        flip, rot, Mq = 0, 0.0, 0.0
        if not cs:
            Mq = 1.0               # 参考池: 共识本身
        elif len(shared) >= 8:
            z = np.array([complex(*cs[ids[i]][:2]) for i in shared])
            z = z / np.maximum(np.abs(z), 1e-9)
            zp = np.exp(1j * th[shared])
            M0 = (zp * np.conj(z)).mean()          # 同向: θ+ρ
            M1 = (np.conj(zp) * np.conj(z)).mean() # 翻转: -θ+ρ
            if abs(M1) > abs(M0):
                flip, rot, Mq = 1, float(-np.angle(M1)), abs(M1)
            else:
                flip, rot, Mq = 0, float(-np.angle(M0)), abs(M0)
        pair_rot[(k, side)] = (flip, rot, Mq)
        thn = (-th if flip else th) + rot
        for i, nid in enumerate(ids):
            e = cs.setdefault(nid, [0.0, 0.0, 0, 0.0])
            e[0] += np.cos(thn[i]); e[1] += np.sin(thn[i]); e[2] += 1; e[3] += dg[i]
    nids = np.array(sorted(cs.keys()), np.int32)
    sx = np.array([cs[n][0] for n in nids]); sy = np.array([cs[n][1] for n in nids])
    tth = np.arctan2(sy, sx).astype(np.float32)
    tdg = np.array([cs[n][3] / cs[n][2] for n in nids], np.float32)
    tdg /= max(tdg.mean(), 1e-9)
    t_ids_l.append(nids); t_th_l.append(tth); t_dg_l.append(tdg.astype(np.float32))
    t_off.append(t_off[-1] + len(nids))

# 核对级判定
pair_use = np.zeros(K, bool)
pair_phi = pk['phi'].astype(np.float32).copy()
pair_q = np.zeros(K, np.float32)
for k in range(K):
    if pk['is_self'][k]:
        f0, r0, q0 = pair_rot[(k, 0)]
        ok = q0 >= ALIGN_MIN
        # 自对: 双侧同池同变换, Δ 不变 -> φ 不变
        pair_use[k] = ok; pair_q[k] = q0
    else:
        f0, r0, q0 = pair_rot[(k, 0)]
        f1, r1, q1 = pair_rot[(k, 1)]
        ok = (q0 >= ALIGN_MIN) and (q1 >= ALIGN_MIN)
        pair_use[k] = ok; pair_q[k] = min(q0, q1)
        if ok:
            # θ_pre^c = σ0 θ_pre + ρ0, θ_post^c = σ1 θ_post + ρ1 (σ=±1)
            # cos(θp-θq+φ) = cos(σ0θpc-σ1θqc + φ - σ0ρ0 + σ1ρ1)
            # 双侧同号 σ0=σ1: 可吸收进 φ_new; 异号: 核对几何变号, 保守弃用
            if f0 == f1:
                sgn = 1 if f0 == 0 else -1
                # cos(θp-θq+φ), θ→θ^c=σθ+ρ  =>  φ_new = σ·φ - ρ0 + ρ1
                pair_phi[k] = sgn * float(pk['phi'][k]) - r0 + r1
            else:
                pair_use[k] = False

E_all = sum(E_map.values())
E_cons = sum(E_map[(int(pk['s'][k]), int(pk['t'][k]))] for k in range(K) if pair_use[k])
np.savez(OUT + "consensus.npz",
         ctype=np.array(ctype, np.int32), t_off=np.array(t_off, np.int64),
         t_ids=np.concatenate(t_ids_l), t_theta=np.concatenate(t_th_l),
         t_deg=np.concatenate(t_dg_l),
         pair_use_cons=pair_use, pair_phi_new=pair_phi, pair_q=pair_q)
log(f"types with pools: {len(ctype)}, consensus pairs: {pair_use.sum()}/{K}, "
    f"edge coverage: {100*E_cons/E_all:.1f}% of hit-pair edges")

# CX 类型对齐质量抽查
tn = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]
CX = {"EPG", "EPGt", "PEN_a(PEN1)", "PEN_b(PEN2)", "PEG", "Delta7"}
for k in range(K):
    s, t = tn[pk['s'][k]], tn[pk['t'][k]]
    if s in CX and t in CX:
        log(f"  CX pair {s}->{t}: use={pair_use[k]}, q={pair_q[k]:.2f}")
