#!/usr/bin/env python3
"""Stage 57: CX 低噪声实例化原型 (纯 CX 盒, 与 49/53 的 K 计算严格同构)
策略: iid=现有伯努利采样(应复现 v4.1 数值) / sys=系统化配额放置 / blend ρ
"""
import numpy as np
import pickle, json

ROOT = "/mnt/agents/output/果蝇生成器/"
OUT = ROOT + "data_census/"
rng = np.random.default_rng(0)

g = np.load(ROOT + "基因组_distribs.npz", allow_pickle=True)
T = 11751
uk, cnt = g['uk'].astype(np.int64), g['cnt'].astype(np.int64)
type_count = g['type_count'].astype(np.int64)
tn_str = [str(x) for x in np.load(OUT + "type_names.npy", allow_pickle=True)]

lp = pickle.load(open(ROOT + "latent_params.pkl", "rb"))
cx_types = list(lp['type_list'])
cx_tidx = [tn_str.index(x) for x in cx_types]
CXSET = set(cx_tidx)
import pandas as pd
cx_cells = pd.read_csv(ROOT + "data_cx/cx_cells.csv")
th_learned = np.load(ROOT + "theta_learned.npy")
cx_theta_pool = {x: th_learned[(cx_cells['type'] == x).to_numpy()] for x in cx_types}
a_par, phi_par = lp['a'], lp['phi']

HOPS = ["Delta7", "PEG", "PEN_a(PEN1)", "PEN_b(PEN2)"]

def instantiate(jitter=0.1, seed=0):
    r = np.random.default_rng(seed)
    theta = {}           # tidx -> per-cell θ
    cell_of = {}         # (tidx, idx) -> global cell row
    row = 0
    ctname = []
    for xs, t in enumerate(cx_tidx):
        n = int(type_count[t])
        pool = cx_theta_pool[cx_types[xs]]
        dd = r.integers(0, len(pool), n)
        theta[t] = pool[dd] + r.normal(0, jitter, n)
        cell_of[t] = (row, row + n)
        ctname += [cx_types[xs]] * n
        row += n
    return theta, cell_of, np.array(ctname)

def place_pair(thp, thq, a, phi, e, strategy, rho=0.0, r=None):
    """严格复刻 38 语义: 仅 a>0.05 的对走核; 其余(含负 a, 38 公式退化)均匀。
    sys 模式下均匀对也做低噪声(平衡轮转)放置。"""
    ns_, nt_ = len(thp), len(thq)
    ncell = ns_ * nt_
    if a <= 0.05:
        if strategy == "iid":
            return r.integers(0, ns_, e), r.integers(0, nt_, e)
        # 低噪声均匀: 随机起点轮转
        u0 = r.integers(0, ncell)
        idx = (u0 + np.arange(e)) % ncell
        if strategy == "blend":
            n_iid = int(round(e * rho))
            if n_iid:
                idx[n_iid:] = r.integers(0, ncell, e - n_iid)
        return idx // nt_, idx % nt_
    Wij = np.exp(a * (np.cos(thp[:, None] - thq[None, :] + phi) - 1.0))  # (ns, nt)
    flat = Wij.ravel()
    cdf = np.cumsum(flat); cdf /= cdf[-1]
    def sys_sample(m):
        u = r.random()
        pos = (u + np.arange(m)) / m
        return np.searchsorted(cdf, pos)
    n_iid = int(round(e * rho)) if strategy == "blend" else (e if strategy == "iid" else 0)
    idxs = []
    if n_iid:
        idxs.append(r.choice(ncell, size=n_iid, p=flat / flat.sum()))
    if e - n_iid:
        idxs.append(sys_sample(e - n_iid))
    idx = np.concatenate(idxs)
    return idx // nt_, idx % nt_

def build_K(theta, cell_of, ctname, strategy, rho, seed):
    r = np.random.default_rng(seed + 1000)
    n_cells = len(ctname)
    W = np.zeros((n_cells, n_cells), np.float32)
    for k in range(len(uk)):
        s, t = int(uk[k] // T), int(uk[k] % T)
        if s not in CXSET or t not in CXSET:
            continue
        e = int(cnt[k])
        if e == 0:
            continue
        xs, xt = cx_tidx.index(s), cx_tidx.index(t)
        thp = theta[s]
        thq = thp if s == t else theta[t]
        pi, qi = place_pair(thp, thq, float(a_par[xs, xt]), float(phi_par[xs, xt]),
                            e, strategy, rho, r)
        p0, _ = cell_of[s]; q0, _ = cell_of[t]
        np.add.at(W, (q0 + qi, p0 + pi), 1)
    epg = np.where(ctname == "EPG")[0]
    K = W[np.ix_(epg, epg)].copy()
    for x in HOPS:
        mid = np.where(ctname == x)[0]
        if len(mid):
            K += W[np.ix_(epg, mid)] @ W[mid][:, epg]
    return K, epg, ctname

def stats(K, th_e):
    d = th_e[:, None] - th_e[None, :]
    y = K.ravel()
    X = np.stack([np.ones_like(y), np.cos(d).ravel(), np.sin(d).ravel()], -1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    tot = ((y - y.mean()) ** 2).sum(); res_ = ((y - pred) ** 2).sum()
    return {"amp": float(np.hypot(coef[1], coef[2])),
            "resid_var": float(res_ / len(y)), "R2": float(1 - res_ / tot),
            "Kmax": float(K.max()), "Kmean": float(K.mean())}

print(f"{'strategy':10s} {'seed':>4s} {'R2':>6s} {'amp':>6s} {'resid':>8s} {'Kmax':>6s} {'Kmean':>6s}")
out = {}
for strat, rho in [("iid", 1.0), ("sys", 0.0), ("blend", 0.5), ("blend", 0.25), ("blend", 0.1)]:
    tag = f"{strat}{'' if strat=='iid' else f'_rho{rho:g}'}"
    rs = []
    for seed in range(3):
        theta, cell_of, ctname = instantiate(seed=seed)
        K, epg, ct = build_K(theta, cell_of, ctname, strat, rho, seed)
        epg_cells_theta = theta[cx_tidx[ctname[epg[0]] == cx_types and cx_tidx.index("EPG")]][0] if False else None
        th_e = theta[tn_str.index("EPG")][epg - cell_of[tn_str.index("EPG")][0]]
        st = stats(K, th_e)
        rs.append(st)
        print(f"{tag:10s} {seed:4d} {st['R2']:6.3f} {st['amp']:6.2f} {st['resid_var']:8.1f} {st['Kmax']:6.0f} {st['Kmean']:6.1f}")
    out[tag] = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
    print(f"{tag:10s} mean {out[tag]['R2']:6.3f} {out[tag]['amp']:6.2f} {out[tag]['resid_var']:8.1f} {out[tag]['Kmax']:6.0f}")
json.dump(out, open(ROOT + "lownoise_proto.json", "w"), indent=1)
print("参考: real R2=0.617 amp=5.8 resid=10.4 Kmax=34 | v4.1 全图 0.474/12.7/87.7/87")
