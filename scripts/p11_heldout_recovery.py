# P11 留边恢复实验(§33a 注册) — 4GB 内存精简版
import numpy as np, json, time, gc
t0 = time.time()
D = '/mnt/agents/output/果蝇生成器/data_female/'
N, T, E = 138327, 8772, 19656020
nt = np.fromfile(D+'node_type.i32', dtype=np.int32)
xyz = np.fromfile(D+'xyz.f32', dtype=np.float32).reshape(N, 3)
ep = np.fromfile(D+'e_pre.i32', dtype=np.int32)
eq = np.fromfile(D+'e_post.i32', dtype=np.int32)
rng = np.random.default_rng(20260928)
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

hide = rng.random(E) < 0.05
ep_o, eq_o = ep[~hide], eq[~hide]
E_obs = len(ep_o); E_hid = E - E_obs
log(f"观测 {E_obs:,} 隐藏 {E_hid:,}")

key_full = np.sort(ep.astype(np.int64)*N + eq)
key_obs = np.sort(ep_o.astype(np.int64)*N + eq_o)
def has(kf, u, v):
    k = u.astype(np.int64)*N + v
    i = np.clip(np.searchsorted(kf, k), 0, len(kf)-1)
    return kf[i] == k
log("键库完成")

tpk = nt[ep_o].astype(np.int64)*T + nt[eq_o]
tpk_u, cnt_i = np.unique(tpk, return_counts=True)
ntp = len(tpk_u); cnt = cnt_i.astype(np.float64); del cnt_i
tp_idx = np.searchsorted(tpk_u, tpk).astype(np.int32); del tpk; gc.collect()
rich = cnt >= 30
nr = int(rich.sum())
rid_of = np.full(ntp, -1, np.int32); rid_of[rich] = np.arange(nr)
log(f"tp={ntp:,} rich={nr:,} (覆盖边 {cnt[rich].sum()/E_obs:.1%})")

dvec = (xyz[eq_o] - xyz[ep_o]).astype(np.float32)
dist = np.sqrt((dvec**2).sum(1))
NB = 48; bins = np.geomspace(1.0, 1200.0, NB+1).astype(np.float32)
dbin = np.clip(np.searchsorted(bins, dist)-1, 0, NB-1).astype(np.int32)
rm = rich[tp_idx]
rid = rid_of[tp_idx[rm]]
H_r = np.bincount(rid*NB + dbin[rm], minlength=nr*NB).reshape(nr, NB).astype(np.float32)
H_gl = np.bincount(dbin, minlength=NB).astype(np.float32)
cntr = cnt[rich].astype(np.float32)
mu = np.zeros((nr,3), np.float32); var = np.zeros((nr,3), np.float32)
for c in range(3):
    w = dvec[:,c][rm]
    mu[:,c] = np.bincount(rid, weights=w, minlength=nr)/np.maximum(cntr,1)
    var[:,c] = np.bincount(rid, weights=w*w, minlength=nr)/np.maximum(cntr,1) - mu[:,c]**2
var = np.maximum(var, 4.0)
del rid, rm; gc.collect()
rev = has(key_obs, eq_o, ep_o)
dy = np.bincount(tp_idx, weights=rev.astype(np.float32), minlength=ntp)
p_rev = np.clip((dy+1)/(cnt+2), 1e-4, 0.9999).astype(np.float32)
p_rev_gl = float(rev.mean()); del dy, rev; gc.collect()
log(f"统计完成 全局dyad率 {p_rev_gl:.3f} 平均距离 {dist.mean():.0f}µm")

cdf = np.cumsum(H_r, 1)/np.maximum(cntr[:,None],1)
p99_r = bins[np.clip((cdf < 0.99).sum(1), 0, NB-1)+1]
p99_gl = float(bins[int((np.cumsum(H_gl)/H_gl.sum() < 0.99).sum())+1])

uq_keys, uq_cnts = np.unique(ep_o.astype(np.int64)*T + nt[eq_o], return_counts=True)
uq_cnts = uq_cnts.astype(np.int32)
def qcount(u, tq):
    k = u.astype(np.int64)*T + tq
    i = np.clip(np.searchsorted(uq_keys, k), 0, len(uq_keys)-1)
    return np.where(uq_keys[i] == k, uq_cnts[i], 0)
log(f"配额键 {len(uq_keys):,}")

order = np.argsort(nt, kind='stable'); nt_s = nt[order]
t_start = np.searchsorted(nt_s, np.arange(T+1))
def sample_of_type(tarr, m):
    tarr = tarr.astype(np.int64)
    s = t_start[tarr]; e2 = t_start[tarr+1]
    ln = e2 - s
    r = (rng.random(m)*np.maximum(ln,1)).astype(np.int64)
    return np.where(ln>0, order[s+r], -1).astype(np.int32)

QB = 6
def qbin(q): return np.clip(q, 0, QB-1)

# ---- 配额经验表: 观测边 vs 同tp非边 ----
m_q = 400000
idx_e = rng.integers(0, E_obs, m_q)
qe = qbin(np.maximum(qcount(ep_o[idx_e], nt[eq_o[idx_e]])-1, 0))
tps_q = tpk_u[tp_idx[idx_e]]
cu = sample_of_type(tps_q//T, m_q); cv = sample_of_type(tps_q%T, m_q)
alive = (cu>=0)&(cv>=0)&(cu!=cv)
ok = alive.copy()
ok[alive] = ~has(key_full, cu[alive], cv[alive])
qn = qbin(qcount(cu[ok], nt[cv[ok]]))
hq_e = np.bincount(qe, minlength=QB)+1.0; hq_n = np.bincount(qn, minlength=QB)+1.0
logLR_q = np.clip(np.log(hq_e/hq_e.sum()) - np.log(hq_n/hq_n.sum()), -3, 3).astype(np.float32)
log(f"配额表 logLR={np.round(logLR_q,2)}")

# ---- 3) 评估集 ----
M0 = 300000
hid_idx = np.where(hide)[0]
pos_i = hid_idx[rng.integers(0, E_hid, M0)]
pu, pv = ep[pos_i], eq[pos_i]
ttp = nt[pu].astype(np.int64)*T + nt[pv]
pi = np.clip(np.searchsorted(tpk_u, ttp), 0, ntp-1)
known = tpk_u[pi] == ttp
log(f"正例 {M0}, tp已知比例 {known.mean():.4f}")

nu = np.empty(M0, np.int32); nv = np.empty(M0, np.int32)
alive = np.ones(M0, bool)
for it in range(40):
    if not alive.any(): break
    ai = np.where(alive)[0]; m2 = len(ai)
    cu = sample_of_type(ttp[ai]//T, m2); cv = sample_of_type(ttp[ai]%T, m2)
    d2 = np.sqrt(((xyz[cv]-xyz[cu])**2).sum(1))
    pj = pi[ai]
    lim = np.where(rich[pj], p99_r[np.clip(rid_of[pj],0,nr-1)], np.float32(p99_gl))
    okk = (cu>=0)&(cv>=0)&(cu!=cv)&(d2<=lim)
    okk[okk] = ~has(key_full, cu[okk], cv[okk])
    nu[ai[okk]] = cu[okk]; nv[ai[okk]] = cv[okk]; alive[ai[okk]] = False
    if it % 10 == 9: log(f"  负例采样 iter{it+1} 剩 {alive.sum()}")
log(f"硬负例 {M0-alive.sum()}/{M0}")
keep = ~alive
pu, pv, nu, nv, pi, known, ttp = pu[keep], pv[keep], nu[keep], nv[keep], pi[keep], known[keep], ttp[keep]
M = len(pu)

# ---- 4) 打分 ----
def scores(u, v, pidx, kmask):
    d2v = xyz[v] - xyz[u]
    dd = np.sqrt((d2v**2).sum(1)).astype(np.float32)
    db = np.clip(np.searchsorted(bins, dd)-1, 0, NB-1)
    ris = rich[pidx]
    rid2 = np.clip(rid_of[pidx], 0, nr-1)
    hmix = np.where(ris[:,None], H_r[rid2], H_gl[None,:])
    cmix = np.where(ris, cntr[rid2], H_gl.sum())
    s_dist = np.log((hmix[np.arange(len(u)), db]+0.5)/(cmix+0.5*NB))
    s_off = np.zeros(len(u), np.float32)
    okm = ris & kmask
    if okm.any():
        r2 = rid2[okm]
        r = d2v[okm] - mu[r2]
        s_off[okm] = np.clip(-0.5*((r*r)/var[r2]).sum(1) - 0.5*np.log(2*np.pi*var[r2]).sum(1), -60, 0)
    r_obs = has(key_obs, v, u)
    pr = np.where(ris, p_rev[pidx], p_rev_gl)
    s_dyad = np.where(r_obs, np.log(pr/(1-pr)), np.float32(0))
    s_quota = logLR_q[qbin(qcount(u, nt[v]))]
    return s_dist, s_off, s_dyad.astype(np.float32), s_quota, dd

sp = scores(pu, pv, pi, known)
ni = np.clip(np.searchsorted(tpk_u, ttp), 0, ntp-1)
sn = scores(nu, nv, ni, np.ones(M, bool))

def auc(pos, neg):
    a = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(a)).astype(np.float64)
    n1 = len(pos); n0 = len(neg)
    return float((r[:n1].sum() - n1*(n1-1)/2)/(n1*n0))

res = {'M': int(M), 'E_obs': int(E_obs), 'E_hid': int(E_hid),
       'known_tp_frac': float(known.mean()), 'p_rev_global': p_rev_gl,
       'logLR_quota': logLR_q.tolist()}
for k, i in [('dist',0), ('off',1), ('dyad',2), ('quota',3)]:
    res[f'auc_{k}'] = auc(sp[i], sn[i])
res['auc_full'] = auc(sp[0]+sp[1]+sp[2]+sp[3], sn[0]+sn[1]+sn[2]+sn[3])
res['auc_dist_off'] = auc(sp[0]+sp[1], sn[0]+sn[1])
res['auc_dist_off_quota'] = auc(sp[0]+sp[1]+sp[3], sn[0]+sn[1]+sn[3])
res['auc_dyad_dist'] = auc(sp[0]+sp[2], sn[0]+sn[2])
res['auc_dyad_dist_off'] = auc(sp[0]+sp[1]+sp[2], sn[0]+sn[1]+sn[2])
res['auc_dyad_quota'] = auc(sp[2]+sp[3], sn[2]+sn[3])
res['auc_distonly_raw'] = auc(-sp[4], -sn[4])
deg_out = np.bincount(ep_o, minlength=N); deg_in = np.bincount(eq_o, minlength=N)
res['auc_degree'] = auc(np.log1p(deg_out[pu])+np.log1p(deg_in[pv]),
                        np.log1p(deg_out[nu])+np.log1p(deg_in[nv]))
# 门槛判决
best_base = max(res['auc_distonly_raw'], res['auc_degree'], 0.5)
res['best_baseline'] = best_base
res['gate_a'] = bool(res['auc_full'] >= 0.75 and res['auc_full'] >= best_base + 0.05)
log(json.dumps(res, indent=1))
json.dump(res, open('/tmp/p11/p11_res.json','w'), indent=1)
np.savez('/tmp/p11/p11_terms.npz', sp0=sp[0],sp1=sp[1],sp2=sp[2],sp3=sp[3],sp4=sp[4],
          sn0=sn[0],sn1=sn[1],sn2=sn[2],sn3=sn[3],sn4=sn[4])
log("DONE")
