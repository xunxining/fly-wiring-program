#!/usr/bin/env python3
# 150_swap.py — §70 ⑧: 跨性别参数互换边级预测 — 分段版 (fit/eval 分进程, 防 OOM)
# usage: python3 150_swap.py fit female|male   /   python3 150_swap.py eval
import json, sys, time
import numpy as np

t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
ROOT = "/mnt/agents/output/果蝇生成器/"
BIG, NB, NTEST = 500, 24, 200_000
namesF = np.load(ROOT+"data_female/type_names.npy", allow_pickle=True).astype(str)
namesM = np.load(ROOT+"data_male/type_names.npy", allow_pickle=True).astype(str)
vocab = {n: i for i, n in enumerate(sorted(set(namesF) | set(namesM)))}
V = len(vocab)

def load_sex(sex):
    D = ROOT + f"data_{sex}/"
    class S: pass
    S.sex = sex
    S.ep = np.fromfile(D+"e_pre.i32", dtype=np.int32)
    S.eq = np.fromfile(D+"e_post.i32", dtype=np.int32)
    S.nty = np.fromfile(D+"node_type.i32", dtype=np.int32)
    names = namesF if sex == "female" else namesM
    S.tid = np.array([vocab[x] for x in names], dtype=np.int32)[S.nty]   # 逐细胞词表 id
    S.xyz = np.fromfile(D+"xyz.f32", dtype=np.float32).reshape(-1,3)
    S.N = len(S.nty); S.E = len(S.ep)
    rng = np.random.default_rng(11)
    S.is_test = rng.random(S.E) < 0.10
    log(f"{sex}: N={S.N:,} E={S.E:,} test={S.is_test.sum():,}")
    return S

def dist_chunk(S, a, b, chunk=4_000_000):
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        e = min(s+chunk, len(a))
        d = np.sqrt(((S.xyz[a[s:e]].astype(np.float64)-S.xyz[b[s:e]])**2).sum(1))
        bad = ~np.isfinite(d) | (S.xyz[a[s:e]]==0).all(1) | (S.xyz[b[s:e]]==0).all(1)
        d[bad] = -1
        out[s:e] = d
    return out

if sys.argv[1] == "fit":
    sex = sys.argv[2]; S = load_sex(sex)
    tr = ~S.is_test
    tp, tq = S.ep[tr], S.eq[tr]
    pid = S.tid[tp].astype(np.int64)*V + S.tid[tq]
    upt, cnt = np.unique(pid, return_counts=True)
    tcnt = np.bincount(S.tid, minlength=V).astype(np.float64)
    pre_i, post_i = upt//V, upt%V
    rate = cnt / np.maximum(tcnt[pre_i]*tcnt[post_i], 1) * 1e6
    d_tr = dist_chunk(S, tp, tq)
    ok = d_tr >= 0
    qe = np.quantile(d_tr[ok], np.linspace(0,1,NB+1)[1:-1])
    db_tr = np.searchsorted(qe, d_tr); db_tr[~ok] = NB
    del d_tr, tp, tq, pid
    import gc; gc.collect()
    rng2 = np.random.default_rng(21)
    sa = rng2.integers(0, S.N, 1_200_000); sb = rng2.integers(0, S.N, 1_200_000)
    m = sa != sb; sa, sb = sa[m][:1_000_000], sb[m][:1_000_000]
    dbg = np.searchsorted(qe, dist_chunk(S, sa, sb)); dbg[dist_chunk(S,sa,sb) < 0] = NB
    h_b = np.bincount(dbg, minlength=NB+1) + 0.5
    h_e = np.bincount(db_tr, minlength=NB+1) + 0.5
    LR_glob = (h_e/h_e.sum())/(h_b/h_b.sum())
    big_pid = upt[cnt >= BIG]
    # 大对直方图: 需要 pid+db_tr — 重建 pid (内存已释放)
    tp, tq = S.ep[tr], S.eq[tr]
    pid = S.tid[tp].astype(np.int64)*V + S.tid[tq]
    pos = np.searchsorted(big_pid, pid)
    inbig = (pos < len(big_pid)) & (big_pid[np.clip(pos,0,len(big_pid)-1)] == pid)
    hh = np.bincount(np.clip(pos,0,len(big_pid)-1)[inbig]*(NB+1) + db_tr[inbig],
                     minlength=len(big_pid)*(NB+1)).reshape(len(big_pid), NB+1) + 0.5
    LR_big = (hh/hh.sum(1,keepdims=True))/(h_b/h_b.sum())
    np.savez_compressed(ROOT+f"data_male/swap_params_{sex}.npz",
                        upt=upt, rate=rate, qe=qe, LR_glob=LR_glob,
                        big_pid=big_pid, LR_big=LR_big.astype(np.float32))
    log(f"{sex} 参数已存: 类型对 {len(upt):,} 大对 {len(big_pid):,}")

else:  # eval
    PF = dict(np.load(ROOT+"data_male/swap_params_female.npz"))
    PM = dict(np.load(ROOT+"data_male/swap_params_male.npz"))
    log(f"参数载入: ♀{len(PF['upt']):,} ♂{len(PM['upt']):,}; 共享对 {len(set(PF['upt'].tolist())&set(PM['upt'].tolist())):,}")
    def test_set(S):
        # §71 修正: 类型桶采样(147/150初版的全图随机候选在小类型上86-89%落空)
        te = np.where(S.is_test)[0]
        rng3 = np.random.default_rng(12)
        te = te[rng3.permutation(len(te))[:NTEST]]
        ti, tj = S.ep[te], S.eq[te]
        tt = S.nty[tj]
        key_all = np.unique(S.ep.astype(np.int64)*S.N + S.eq)
        order = np.argsort(S.nty, kind="stable")
        cnts = np.bincount(S.nty)
        starts = np.concatenate([[0], np.cumsum(cnts)[:-1]])
        neg = np.empty(len(te), np.int64)
        member_ok = cnts[tt] >= 2
        matched = np.zeros(len(te), bool)
        active = np.where(member_ok)[0]
        for _ in range(40):
            if not len(active): break
            t_n = tt[active]
            cand = order[starts[t_n] + rng3.integers(0, cnts[t_n])]
            k = ti[active].astype(np.int64)*S.N + cand
            h = np.searchsorted(key_all, k)
            free = (cand != ti[active]) & ((h >= len(key_all)) | (key_all[np.clip(h,0,len(key_all)-1)] != k))
            neg[active[free]] = cand[free]; matched[active[free]] = True
            active = active[~free]
        km = member_ok & matched
        log(f"{S.sex} 桶采样: 剔除 {int((~km).sum()):,} (单成员 {int((~member_ok).sum())}, 未配 {int((~matched&member_ok).sum())})")
        return ti[km], tj[km], neg[km]
    def score(S, P, a, b):
        pid = S.tid[a].astype(np.int64)*V + S.tid[b]
        pos = np.searchsorted(P["upt"], pid)
        hit = (pos < len(P["upt"])) & (P["upt"][np.clip(pos,0,len(P["upt"])-1)] == pid)
        q = np.where(hit, P["rate"][np.clip(pos,0,len(P["upt"])-1)], 1e-9)
        db = np.searchsorted(P["qe"], dist_chunk(S, a, b)); db[dist_chunk(S,a,b) < 0] = NB
        posb = np.searchsorted(P["big_pid"], pid)
        hb = (posb < len(P["big_pid"])) & (P["big_pid"][np.clip(posb,0,len(P["big_pid"])-1)] == pid)
        dl = P["LR_glob"][db].copy()
        dl[hb] = P["LR_big"][np.clip(posb,0,len(P["big_pid"])-1)[hb], db[hb]]
        return q, q*dl, hit
    def auc(p, n): return float((p > n).mean() + 0.5*(p == n).mean())
    OUT = {}
    for sex, Pself, Pdonor, dname in [("female", PF, PM, "male"), ("male", PM, PF, "female")]:
        S = load_sex(sex)
        ti, tj, neg = test_set(S)
        res = {}; cov = None
        for tag, P in [("within", Pself), ("cross", Pdonor)]:
            qp, qdp, hp = score(S, P, ti, tj)
            qn_, qdn, hn = score(S, P, ti, neg)
            res[tag] = dict(quota=auc(qp, qn_), q_dist=auc(qdp, qdn))
            if tag == "cross": cov = float(hp.mean())
        OUT[sex] = dict(donor=dname, coverage=cov, **res)
        log(f"{sex} (donor={dname}): cov={cov:.3f} within={res['within']} cross={res['cross']}")
        del S
        import gc; gc.collect()
    g1f = OUT["female"]["cross"]["q_dist"] >= OUT["female"]["within"]["q_dist"] - 0.02
    g1m = OUT["male"]["cross"]["q_dist"] >= OUT["male"]["within"]["q_dist"] - 0.02
    OUT["gates"] = dict(G1=bool(g1f and g1m), G1_female=bool(g1f), G1_male=bool(g1m),
                        G2=bool(OUT["female"]["coverage"]>=0.6 and OUT["male"]["coverage"]>=0.6))
    json.dump(OUT, open(ROOT+"data_male/p8_swap.json","w"), indent=1)
    log(json.dumps(OUT["gates"]))
    print(json.dumps(OUT, indent=1))
