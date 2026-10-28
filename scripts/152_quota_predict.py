#!/usr/bin/env python3
# 152_quota_predict.py — §140 P19: 逐细胞连接预算的可推导性 (类型/谱系/位置 → 配额, 留出预测)
# 注册门槛见 报告.md §140。usage: python3 152_quota_predict.py
import sys, json, time
sys.path.insert(0, "/mnt/agents/output/果蝇生成器/vendor")
import numpy as np, pandas as pd
import pyarrow.feather as pf
from scipy.spatial import cKDTree

R = "/mnt/agents/output/果蝇生成器/"
DM = R + "data_male/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

# ---------- 载入 ----------
node_type = np.fromfile(DM+"node_type.i32", dtype=np.int32)
node_ids  = np.fromfile(DM+"node_ids.i64", dtype=np.int64)
xyz       = np.fromfile(DM+"xyz.f32", dtype=np.float32).reshape(-1,3)
pre       = np.fromfile(DM+"e_pre.i32", dtype=np.int32)
N = len(node_type)
od = np.bincount(pre, minlength=N).astype(np.float64)
t = pf.read_table(R+"data/body-annotations.feather",
                  columns=["bodyId","itoleeHl","trumanHl"])
df = pd.DataFrame({"bodyId": t.column("bodyId").to_pylist(),
                   "ito": t.column("itoleeHl").to_pylist(),
                   "tru": t.column("trumanHl").to_pylist()}).set_index("bodyId")
ann = df.reindex(node_ids)
ito = ann["ito"].fillna("").values; tru = ann["tru"].fillna("").values
hl_raw = np.where(ito!="", ito, tru)
hl_lab, _ = pd.factorize(hl_raw); hl_lab = hl_lab.astype(np.int64)
hl_lab[hl_raw==""] = -1
fin = np.isfinite(xyz).all(axis=1)
A = np.where((hl_lab>=0) & fin)[0]
B = np.where(fin)[0]
log(f"N={N:,}  A={len(A):,}  B={len(B):,}")

rng = np.random.default_rng(13)
fold = np.arange(N) % 5
fold[rng.permutation(N)] = np.arange(N) % 5  # deterministic shuffle assignment

def gmean(keys_tr, val_tr):
    uk, inv = np.unique(keys_tr, return_inverse=True)
    c = np.bincount(inv); s = np.bincount(inv, weights=val_tr)
    return uk, s/np.maximum(c,1), c

def gapply(uk, means, cnts, keys_te, fb):
    pos = np.searchsorted(uk, keys_te)
    pos_clip = np.clip(pos, 0, len(uk)-1)
    hit = (uk[pos_clip]==keys_te) & (cnts[pos_clip]>=5)
    out = fb.copy()
    out[hit] = means[pos_clip[hit]]
    return out

def sse_r2(y, p):
    return ((y-p)**2).sum()

# ---------- 目标 1: od ----------
def eval_od(pop, use_hl, hl_override=None, do_m3=True):
    hl = hl_lab if hl_override is None else hl_override
    sse1=sse2=sse3=0.0; sst=0.0
    for f in range(5):
        te = pop[fold[pop]==f]; tr = pop[fold[pop]!=f]
        gm = od[tr].mean()
        uk,m,c = gmean(node_type[tr], od[tr])
        p1 = gapply(uk,m,c,node_type[te], np.full(len(te), gm))
        sse1 += sse_r2(od[te],p1)
        if use_hl:
            ktr = node_type[tr]*100003 + hl[tr]; kte = node_type[te]*100003 + hl[te]
            uk2,m2,c2 = gmean(ktr, od[tr])
            p2 = gapply(uk2,m2,c2,kte, p1)
            sse2 += sse_r2(od[te],p2)
            base = p2; grp_tr = ktr; grp_te = kte
        else:
            base = p1; grp_tr = node_type[tr].astype(np.int64); grp_te = node_type[te].astype(np.int64)
        # M3: 组内 8 近邻空间残差
        if not do_m3:
            sse3 += np.nan; sst += ((od[te]-od[te].mean())**2).sum(); sse1+=0; continue
        p3 = base.copy()
        res_tr = od[tr] - base[np.isin(pop,tr)] if False else None
        # base 与 te 对齐; 训练侧 base 需要重算: 用同口径组均值
        if use_hl:
            base_tr = gapply(uk2,m2,c2,grp_tr, gapply(uk,m,c,node_type[tr], np.full(len(tr),gm)))
        else:
            base_tr = gapply(uk,m,c,node_type[tr], np.full(len(tr),gm))
        res_tr = od[tr] - base_tr
        ug = np.unique(grp_tr)
        # 建组 -> 位置索引
        gpos = {g: np.where(grp_tr==g)[0] for g in ug}
        order_te = np.argsort(grp_te, kind="stable")
        gte_sorted = grp_te[order_te]
        bounds = np.searchsorted(gte_sorted, np.append(ug, ug[-1]+1))
        for gi,g in enumerate(ug):
            idxtr = gpos[g]
            if len(idxtr)<9: continue
            sel = order_te[bounds[gi]:bounds[gi+1]]
            if len(sel)==0: continue
            tree = cKDTree(xyz[tr[idxtr]])
            dd,ii = tree.query(xyz[te[sel]], k=8)
            w = 1.0/(dd+1.0)
            p3[sel] += (w*res_tr[idxtr][ii]).sum(1)/w.sum(1)
        sse3 += sse_r2(od[te],p3)
        sst  += ((od[te]-od[te].mean())**2).sum()
    return 1-sse1/sst, 1-sse2/sst, 1-sse3/sst

log("== od, 主总体 A ==")
rA = eval_od(A, True)
log(f"A: R2 M1={rA[0]:.4f} M2={rA[1]:.4f} M3={rA[2]:.4f}  dM2-M1={rA[1]-rA[0]:.4f} dM3-M1={rA[2]-rA[0]:.4f}")
log("== od, 次总体 B ==")
rB = eval_od(B, False)
log(f"B: R2 M1={rB[0]:.4f} M3'={rB[2]:.4f}  d={rB[2]-rB[0]:.4f}")

# ---------- 护栏: 谱系置换零分布 (50 次, od, A) ----------
perm_d = np.empty(50)
for r in range(50):
    hlp = hl_lab.copy()
    for ty in np.unique(node_type[A]):
        m = (node_type==ty)
        idx = np.where(m & (hl_lab>=0))[0]
        if len(idx)>1: hlp[idx] = hl_lab[idx[rng.permutation(len(idx))]]
    perm_d[r] = eval_od(A, True, hlp, False)[1] - rA[0]
    if r%10==9: log(f"perm {r+1}/50 done, d={perm_d[r]:.4f}")
p99 = np.percentile(perm_d, 99)
log(f"lineage null p99 = {p99:.4f}, observed dM2-M1 = {rA[1]-rA[0]:.4f}")


# ---------- 目标 2: 非零配额条目 (主总体 A, 向量化低内存版) ----------
cq_cell = np.fromfile(DM+"cq_cell.i32", dtype=np.int32).astype(np.int64)
cq_t    = np.fromfile(DM+"cq_t.i32",    dtype=np.int32).astype(np.int64)
cq_cnt  = np.fromfile(DM+"cq_cnt.i32",  dtype=np.int32).astype(np.float64)
inA = np.zeros(N, dtype=bool); inA[A]=True
emask = inA[cq_cell]
ec, et, eq = cq_cell[emask], cq_t[emask], cq_cnt[emask]
del cq_cell, cq_t, cq_cnt, emask
log(f"entries on A: {len(eq):,}")

qbar = eq.mean(); sst_q = ((eq-qbar)**2).sum()
sse1q=sse2q=sse3q=0.0
for f in range(5):
    teC = A[fold[A]==f]; trC = A[fold[A]!=f]
    trset = np.zeros(N,dtype=bool); trset[trC]=True
    mtr = trset[ec]; mte = ~mtr
    ec_tr, et_tr, q_tr = ec[mtr], et[mtr], eq[mtr]
    ec_te, et_te, q_te = ec[mte], et[mte], eq[mte]
    # M1: (type, partner_t) 组均值
    k1tr = node_type[ec_tr]*1000037 + et_tr
    k1te = node_type[ec_te]*1000037 + et_te
    uk1,m1,c1 = gmean(k1tr, q_tr)
    p1 = gapply(uk1,m1,c1,k1te, np.full(len(q_te), qbar))
    # M2: (type, hl, partner_t)
    k2tr = (node_type[ec_tr]*100003 + hl_lab[ec_tr])*1000037 + et_tr
    k2te = (node_type[ec_te]*100003 + hl_lab[ec_te])*1000037 + et_te
    uk2,m2,c2 = gmean(k2tr, q_tr)
    p2 = gapply(uk2,m2,c2,k2te, p1)
    # M3: 同 (type,hl) 组 8 近邻空间残差
    grp_tr = node_type[trC]*100003 + hl_lab[trC]
    grp_te = node_type[teC]*100003 + hl_lab[teC]
    nbr = np.full((len(teC),8), -1, dtype=np.int64); wgt = np.zeros((len(teC),8))
    for g in np.unique(grp_tr):
        idxtr = np.where(grp_tr==g)[0]
        if len(idxtr)<9: continue
        sel = np.where(grp_te==g)[0]
        if len(sel)==0: continue
        tree = cKDTree(xyz[trC[idxtr]])
        dd,ii = tree.query(xyz[teC[sel]], k=8)
        nbr[sel] = trC[idxtr][ii]; wgt[sel] = 1.0/(dd+1.0)
    # 训练条目残差 (M2 口径)
    p2_tr = gapply(uk2,m2,c2,k2tr, gapply(uk1,m1,c1,k1tr, np.full(len(q_tr), qbar)))
    resid = q_tr - p2_tr
    # 训练条目按 (cell,t) 排序, 用 searchsorted 查邻居残差
    o = np.argsort(ec_tr*1000037+et_tr, kind="stable")
    key_sorted = ec_tr[o]*1000037 + et_tr[o]
    res_sorted = resid[o]
    # 邻居残差: 对每条测试条目, 8 个邻居逐个查 (向量化批量)
    nb_key_base = np.where(nbr>=0, nbr, 0)
    num = np.zeros(len(q_te)); den = np.zeros(len(q_te))
    # cell -> slot
    slot = np.full(N, -1, dtype=np.int64); slot[teC] = np.arange(len(teC))
    sl = slot[ec_te]
    has_nb = sl>=0
    idx_te = np.where(has_nb)[0]
    for k in range(8):
        nb = nbr[sl[idx_te], k]
        ok = nb>=0
        ii = idx_te[ok]
        if len(ii)==0: continue
        keys = nb[ok]*1000037 + et_te[ii]
        pos = np.searchsorted(key_sorted, keys)
        pos = np.clip(pos, 0, len(key_sorted)-1)
        hit = key_sorted[pos]==keys
        w = wgt[sl[ii], k]
        num[ii] += np.where(hit, w*res_sorted[pos], 0.0)
        den[ii] += np.where(hit, w, 0.0)
    p3 = p2.copy()
    okd = den>0
    p3[okd] += num[okd]/den[okd]
    sse1q += ((q_te-p1)**2).sum(); sse2q += ((q_te-p2)**2).sum(); sse3q += ((q_te-p3)**2).sum()
    log(f"fold {f}: te={len(q_te):,} R2 M1={1-sse1q/sst_q:.4f} M2={1-sse2q/sst_q:.4f} M3={1-sse3q/sst_q:.4f}")

q1,q2,q3 = 1-sse1q/sst_q, 1-sse2q/sst_q, 1-sse3q/sst_q
log(f"quota A: R2 M1={q1:.4f} M2={q2:.4f} M3={q3:.4f}  dM3-M1={q3-q1:.4f}")

out = dict(
  od_A=dict(M1=rA[0],M2=rA[1],M3=rA[2]),
  od_B=dict(M1=rB[0],M3p=rB[2]),
  quota_A=dict(M1=q1,M2=q2,M3=q3,dM3_M1=q3-q1),
  lineage_null_p99=float(p99), obs_dM2_M1_od=float(rA[1]-rA[0]),
  gates=dict(G1="od_A.M3>=0.80", G2="quota_A.M3>=0.50 and dM3_M1>=0.05",
             guard="obs_dM2_M1_od > null p99"),
  G1_pass=bool(rA[2]>=0.80),
  G2_pass=bool(q3>=0.50 and (q3-q1)>=0.05),
  guard_pass=bool((rA[1]-rA[0])>p99),
  n_A=int(len(A)), n_B=int(len(B)), n_entries_A=int(len(eq)))
json.dump(out, open(DM+"p19_quota_predict.json","w"), indent=1)
log("SAVED data_male/p19_quota_predict.json")
log("VERDICT: G1 %s G2 %s guard %s" % (out["G1_pass"], out["G2_pass"], out["guard_pass"]))
