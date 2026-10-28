#!/usr/bin/env python3
# 149_bio_anchor.py — §61: 活动因子 a_i = od_i/mean_od(type) 的生物学身份检验
# T1 雄性VNC血统 η²(200置换) / T2 血统内AP梯度 / T3 双性别Moran's I / T4 神经毡广度 / T5 类型大小
import json, re, time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = "/mnt/agents/output/果蝇生成器/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
rng = np.random.default_rng(13)
OUT = {}

def load(sex):
    D = ROOT + f"data_{sex}/"
    pre = np.fromfile(D + "e_pre.i32", dtype=np.int32)
    node_ty = np.fromfile(D + "node_type.i32", dtype=np.int32)
    xyz = np.fromfile(D + "xyz.f32", dtype=np.float32).reshape(-1, 3)
    N = len(node_ty)
    od = np.bincount(pre, minlength=N).astype(np.float64)
    mean_od = np.bincount(node_ty, weights=od) / np.maximum(np.bincount(node_ty), 1)
    a = od / np.maximum(mean_od[node_ty], 1e-9)
    return N, node_ty, xyz, od, a, D

def popcnt_masks(npz_path, key="OUT"):
    m = np.load(npz_path)
    M = m[key] if key in m.files else m[m.files[0]]
    M = np.ascontiguousarray(M)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
    b = M.view(np.uint8).reshape(M.shape[0], -1)
    return lut[b].sum(axis=1).astype(np.int32)

# ============ 雄性 ============
N, node_ty, xyz, od, a, D = load("male")
node_sc = np.fromfile(D + "node_sc.i32", dtype=np.int32)
tn = np.load(D + "type_names.npy", allow_pickle=True)
VNC = {15, 16, 17, 21, 22, 23, 24, 25, 26, 27}
is_vnc = np.isin(node_sc, list(VNC))
pat = re.compile(r"^(?:IN|AN|EN|TN)(\d{2}[AB])")
hl_of_type = {t: pat.match(str(tn[t])).group(1) for t in range(len(tn)) if pat.match(str(tn[t]))}
hl_ids = {h: i for i, h in enumerate(sorted(set(hl_of_type.values())))}
HL = np.full(N, -1, dtype=np.int32)
for t, h in hl_of_type.items():
    HL[node_ty == t] = hl_ids[h]
sel = is_vnc & (HL >= 0) & np.isfinite(xyz).all(axis=1)
av, hl, ap = a[sel], HL[sel], xyz[sel, 2]   # 长轴 = axis2 (§50)
odv = od[sel]                                # T1 修正: 配额 od 的血统结构(a 的型间结构被归一化恒消)
log(f"male VNC HL cells: {len(av):,}, lineages: {len(np.unique(hl))}")

def eta2(x, g):
    ng = np.bincount(g).astype(np.float64)
    sums = np.bincount(g, weights=x)
    means = sums / np.maximum(ng, 1)
    ssb = (ng * (means - x.mean()) ** 2).sum()
    sst = ((x - x.mean()) ** 2).sum()
    return ssb / max(sst, 1e-12)

# T1 修正(§61 附注): a_i 的型间均值被归一化恒为1, η²(a|HL)≡0 退化;
# 改为配额 od_i 的血统 η²(配额本身是否血统结构化)
e_real = eta2(odv, hl)
null_e = np.array([eta2(odv, rng.permutation(hl)) for _ in range(200)])
p1 = (1 + (null_e >= e_real).sum()) / 201
OUT["T1_hl_eta2_quota"] = dict(eta2=float(e_real), null_mean=float(null_e.mean()),
                               null_max=float(null_e.max()), p=float(p1),
                               G1=bool(p1 < 0.01 and e_real >= 0.05),
                               note="a_i 型间结构被归一化恒消(退化), 改测配额 od_i")
log(f"T1(quota) eta2={e_real:.4f} null {null_e.mean():.4f}±{null_e.std():.4f} (max {null_e.max():.4f}) p={p1:.4f} G1={OUT['T1_hl_eta2_quota']['G1']}")

# T2 血统内 AP 梯度
rhos, lgs = [], []
for h in np.unique(hl):
    m = hl == h
    if m.sum() >= 50:
        r = spearmanr(av[m], ap[m]).statistic
        if np.isfinite(r): rhos.append(r); lgs.append(int(m.sum()))
rhos = np.array(rhos)
mean_rho = np.abs(rhos).mean()
null_mr = []
for _ in range(200):
    nr = []
    for h in np.unique(hl):
        m = hl == h
        if m.sum() >= 50:
            r = spearmanr(rng.permutation(av[m]), ap[m]).statistic
            if np.isfinite(r): nr.append(r)
    null_mr.append(np.abs(np.array(nr)).mean())
null_mr = np.array(null_mr)
p2 = (1 + (null_mr >= mean_rho).sum()) / 201
OUT["T2_ap_gradient"] = dict(mean_abs_rho=float(mean_rho), n_lineages=len(rhos),
                             null_mean=float(null_mr.mean()), p=float(p2),
                             G2=bool(p2 < 0.01 and mean_rho >= 0.15))
log(f"T2 mean|rho|={mean_rho:.3f} over {len(rhos)} lineages null {null_mr.mean():.3f} p={p2:.4f} G2={OUT['T2_ap_gradient']['G2']}")

def moran(sex, a_all, xyz_all):
    fin = np.where(np.isfinite(xyz_all).all(axis=1))[0]
    sub = fin[rng.choice(len(fin), size=min(20000, len(fin)), replace=False)]
    x = a_all[sub]; X = xyz_all[sub]
    tree = cKDTree(X)
    _, idx = tree.query(X, k=9)
    src = np.repeat(np.arange(len(sub)), 8); dst = idx[:, 1:].ravel()
    W = csr_matrix((np.ones(len(src)), (src, dst)), shape=(len(sub), len(sub)))
    W = ((W + W.T) > 0).astype(np.float64)
    S0 = W.sum()
    z = x - x.mean()
    def MI(zv):
        return (len(zv) / S0) * (zv @ (W @ zv)) / max(zv @ zv, 1e-12)
    I_real = MI(z)
    null_I = np.array([MI(rng.permutation(z)) for _ in range(200)])
    p = (1 + (null_I >= I_real).sum()) / 201
    return dict(I=float(I_real), null_mean=float(null_I.mean()), null_max=float(null_I.max()), p=float(p))

# T3 + T4 雄性
nneu_m = popcnt_masks(D + "np_masks_m2.npz")
res3m = moran("male", a, xyz)
rho4m = spearmanr(a, nneu_m).statistic
OUT["T3_moran_male"] = {**res3m, "G3m": bool(res3m["p"] < 0.01 and res3m["I"] >= 0.05)}
OUT["T4_arbor_male"] = dict(rho=float(rho4m), G4m=bool(abs(rho4m) >= 0.15))
log(f"T3 male Moran I={res3m['I']:.4f} null {res3m['null_mean']:.4f} (max {res3m['null_max']:.4f}) p={res3m['p']:.4f}; T4 rho={rho4m:.3f}")

# T5 雄性类型大小
cnt = np.bincount(node_ty); mo = np.bincount(node_ty, weights=od) / np.maximum(cnt, 1)
big = cnt >= 20
r5m = spearmanr(np.log(cnt[big]), mo[big]).statistic
OUT["T5_typesize_male"] = dict(rho=float(r5m), n_types=int(big.sum()))
del node_sc, HL, is_vnc, nneu_m, od, a, xyz
import gc; gc.collect()

# ============ 雌性 ============
N, node_ty, xyz, od, a, D = load("female")
res3f = moran("female", a, xyz)
nneu_f = popcnt_masks(D + "np_masks.npz")
rho4f = spearmanr(a, nneu_f).statistic
OUT["T3_moran_female"] = {**res3f, "G3f": bool(res3f["p"] < 0.01 and res3f["I"] >= 0.05)}
OUT["T4_arbor_female"] = dict(rho=float(rho4f), G4f=bool(abs(rho4f) >= 0.15))
cnt = np.bincount(node_ty); mo = np.bincount(node_ty, weights=od) / np.maximum(cnt, 1)
big = cnt >= 20
OUT["T5_typesize_female"] = dict(rho=float(spearmanr(np.log(cnt[big]), mo[big]).statistic), n_types=int(big.sum()))
log(f"T3 female Moran I={res3f['I']:.4f} p={res3f['p']:.4f}; T4 rho={rho4f:.3f}; T5 f rho={OUT['T5_typesize_female']['rho']:.3f}")

OUT["gates"] = dict(G1=OUT["T1_hl_eta2_quota"]["G1"], G2=OUT["T2_ap_gradient"]["G2"],
                    G3=bool(OUT["T3_moran_male"]["G3m"] and OUT["T3_moran_female"]["G3f"]),
                    G4=bool(OUT["T4_arbor_male"]["G4m"] and OUT["T4_arbor_female"]["G4f"]))
json.dump(OUT, open(ROOT + "data_male/bio_anchor.json", "w"), indent=1)
log("saved data_male/bio_anchor.json; gates:", OUT["gates"])
