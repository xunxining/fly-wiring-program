#!/usr/bin/env python3
"""P9 命题A试点: 接线程序机制生成的稀疏结构 vs 等边数基线。
用法: python3 p9.py --task T1|T2 --cond COND --seed S >> res.jsonl
条件: dense/erdos/ws/het/pruned/qprog/qprog_nd
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)
R_IN, H = 64, 256          # 受体数, 隐藏数
D_IH, D_HH = 0.125, 0.04   # 边预算密度
E_IH = int(R_IN * H * D_IH)   # 2048
E_HH = int(H * H * D_HH)      # 2621
KAPPA = 2.0                # 偏移统计峰度 (von Mises)
RHO = 0.30                 # qprog 互惠预算

# ---------- 掩码生成 ----------
def ring_dist(i, j, L):
    d = abs(i - j) % L
    return min(d, L - d)

def quota_degrees(n, total, rng):
    """逐节点出度配额 (lognormal, 最大余数法精确到 total)"""
    d = rng.lognormal(0.0, 0.6, n)
    q = d / d.sum() * total
    qi = np.floor(q).astype(int)
    if total >= n:
        qi = np.maximum(qi, 1)  # 预算充足时每节点至少 1 边
    diff = total - qi.sum()
    frac = q - np.floor(q)
    order = np.argsort(-frac)
    k = 0
    while diff > 0:
        qi[order[k % n]] += 1; diff -= 1; k += 1
    lo = 1 if total >= n else 0
    while diff < 0:
        idx = order[-(k % n) - 1]
        if qi[idx] > lo: qi[idx] -= 1; diff += 1
        k += 1
    return qi

def offset_targets(i, Li, Lj, quotas_ok, rng, kappa=KAPPA):
    """从 von Mises 偏移分布为源 i 选 distinct 目标"""
    ci = i * Lj / Li
    offs = rng.vonmises(0.0, kappa, len(quotas_ok) * 3)
    js = (np.round(ci + offs * Lj / (2 * np.pi)).astype(int) % Lj)
    out, seen = [], set()
    for j in js:
        if j in seen or not quotas_ok[j]: continue
        seen.add(j); out.append(j)
    return out

def mask_erdos(Li, Lj, E, rng):
    idx = rng.choice(Li * Lj, E, replace=False)
    m = np.zeros((Li, Lj), np.float32); m.flat[idx] = 1.0
    return m

def mask_ws(Li, Lj, E, rng, p=0.1):
    k = max(2, int(round(E / Li)))
    m = np.zeros((Li, Lj), np.float32)
    for i in range(Li):
        ci = i * Lj / Li
        js = [(int(round(ci)) + d) % Lj for d in range(-k // 2, k // 2 + 1)]
        js = list(dict.fromkeys(js))[:k]
        for t, j in enumerate(js):
            if rng.random() < p:
                j = rng.integers(Lj)
            m[i, j] = 1.0
    # 裁剪/补齐到精确 E
    flat = np.flatnonzero(m.ravel())
    if len(flat) > E:
        m.ravel()[rng.choice(flat, len(flat) - E, replace=False)] = 0
    elif len(flat) < E:
        z = np.flatnonzero(1 - m.ravel())
        m.ravel()[rng.choice(z, E - len(flat), replace=False)] = 1
    return m

def mask_het(Li, Lj, E, rng):
    """异配度配额 + 均匀目标 (无局部性无互惠)"""
    q = quota_degrees(Li, E, rng)
    m = np.zeros((Li, Lj), np.float32)
    for i in range(Li):
        js = rng.choice(Lj, min(q[i], Lj), replace=False)
        m[i, js] = 1.0
    return m

def mask_qprog(Li, Lj, E, rng, dyad=True):
    """三机制: 度配额 + von Mises 偏移统计 + 互惠预算 (仅方阵 dyad 生效)"""
    q = quota_degrees(Li, E, rng)
    m = np.zeros((Li, Lj), np.float32)
    budget = np.ones(Lj, bool)
    for i in range(Li):
        js = offset_targets(i, Li, Lj, budget, rng)
        if len(js) < q[i]:  # 配额兜底: 随机补足
            rest = [j for j in rng.permutation(Lj) if j not in js]
            js = js + rest[:q[i] - len(js)]
        m[i, js[:q[i]]] = 1.0
    if dyad and Li == Lj:
        flat = np.flatnonzero(m.ravel())
        n_mut = int(E * RHO / 2)
        src = rng.choice(flat, min(n_mut * 2, len(flat)), replace=False)
        made = 0
        for f in src[:n_mut]:
            i, j = divmod(f, Lj)
            if m[j, i] == 0:
                # 换入反向边: 在 j 行随机删一条非互惠边保持配额
                row = np.flatnonzero(m[j])
                cand = [t for t in row if m[i, t] == 0 and t != i]
                if cand:
                    m[j, rng.choice(cand)] = 0
                    m[j, i] = 1.0; made += 1
    return m

def mask_qprog2(Li, Lj, E, rng, dyad=True):
    """CX式: 4 类型 × 类型对模态偏移表 Δ_st ∈ {-2..2} + 配额 + 互惠"""
    if Lj < 8:  # 输入层太小, 退回通用 qprog
        return mask_qprog(Li, Lj, E, rng, False)
    NT = 4
    ti = np.arange(Li) * NT // Li  # 环上连续分块
    tj = np.arange(Lj) * NT // Lj
    DT = rng.integers(-2, 3, (NT, NT))
    q = quota_degrees(Li, E, rng)
    m = np.zeros((Li, Lj), np.float32)
    by_t = [np.flatnonzero(tj == t) for t in range(NT)]
    for i in range(Li):
        s_ = ti[i]; got = set(); k = 0
        while len(got) < q[i] and k < q[i] * 8:
            t_ = rng.integers(NT)
            c = i * Lj / Li + DT[s_, t_] + rng.vonmises(0, 4) * Lj / (2 * np.pi)
            cand = by_t[t_]
            d = np.abs(cand - c) % Lj; d = np.minimum(d, Lj - d)
            j = int(cand[np.argmin(d)])
            if j not in got: got.add(j)
            k += 1
        if len(got) < q[i]:
            for j in rng.permutation(Lj):
                if int(j) not in got: got.add(int(j))
                if len(got) >= q[i]: break
        m[i, list(got)[:q[i]]] = 1.0
    if dyad and Li == Lj:
        flat = np.flatnonzero(m.ravel()); n_mut = int(E * RHO / 2)
        for f in rng.choice(flat, min(n_mut, len(flat)), replace=False):
            i, j = divmod(f, Lj)
            if m[j, i] == 0:
                row = np.flatnonzero(m[j]); cand = [t for t in row if m[i, t] == 0 and t != i]
                if cand: m[j, rng.choice(cand)] = 0; m[j, i] = 1.0
    return m

def build_masks(cond, rng, n_in=R_IN):
    e_ih = int(n_in * H * D_IH)
    if cond == "dense" or cond == "pruned":
        return np.ones((H, n_in), np.float32), np.ones((H, H), np.float32)
    gen = {"erdos": mask_erdos, "ws": mask_ws, "het": mask_het}[cond] if cond in ("erdos", "ws", "het") else None
    if gen:
        return gen(H, n_in, e_ih, rng), gen(H, H, E_HH, rng)
    if cond == "qprog2":
        return mask_qprog2(H, n_in, e_ih, rng, False), mask_qprog2(H, H, E_HH, rng, True)
    dyad = (cond == "qprog")
    return mask_qprog(H, n_in, e_ih, rng, False), mask_qprog(H, H, E_HH, rng, dyad)

# ---------- 数据 ----------
def data_motion(n, rng):
    T, W = 16, 3
    X = np.zeros((n, T, R_IN), np.float32)
    y = rng.integers(0, 4, n)  # 0左慢 1左快 2右慢 3右快
    for k in range(n):
        v = [1, 3][y[k] % 2] * [-1, 1][y[k] // 2]
        x0 = rng.integers(W, R_IN - W)
        for t in range(T):
            c = int(x0 + v * t) % R_IN
            X[k, t, c] = 1.0
            X[k, t, (c + 1) % R_IN] = 0.5
        # 干扰条 (杂波)
        cd = rng.integers(0, R_IN); vd = rng.choice([-2, -1, 1, 2])
        for t in range(T):
            X[k, t, (cd + vd * t) % R_IN] = 0.35
        X[k] += rng.normal(0, 0.12, X[k].shape).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

def data_heading(n, rng):
    T = 40
    v = rng.normal(0, 0.08, (n, T)).astype(np.float32).clip(-0.3, 0.3)
    th0 = rng.uniform(-np.pi, np.pi, n).astype(np.float32)
    th = th0[:, None] + np.cumsum(v, 1)
    X = np.stack([v * 10.0, np.sin(th), np.cos(th)], -1).astype(np.float32)
    # 地标只在 t<5 可见 (提示初始朝向), 之后仅速度输入
    X[:, 8:, 1:] = 0.0
    X[:, ::12, 1:] = np.stack([np.sin(th), np.cos(th)], -1)[:, ::12, :]  # 周期地标
    Y = np.stack([np.sin(th), np.cos(th)], -1).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y)

# ---------- 模型 ----------
class SparseRNN(nn.Module):
    def __init__(self, n_in, n_out, mih, mhh):
        super().__init__()
        self.mih = torch.tensor(mih); self.mhh = torch.tensor(mhh)
        self.Wih = nn.Parameter(torch.randn(H, n_in) * 0.1)
        self.Whh = nn.Parameter(torch.randn(H, H) * 0.05)
        self.b = nn.Parameter(torch.zeros(H))
        self.Wo = nn.Parameter(torch.randn(n_out, H) * 0.1)
        self.bo = nn.Parameter(torch.zeros(n_out))
        with torch.no_grad():  # 谱半径归一: 所有条件动力稳定性公平
            sr = np.abs(np.linalg.eigvals((self.Whh * self.mhh).numpy())).max()
            self.Whh.data *= 0.95 / max(sr, 1e-6)
    def forward(self, X):
        h = torch.zeros(X.shape[0], H)
        W1 = self.Wih * self.mih; W2 = self.Whh * self.mhh
        ys = []
        for t in range(X.shape[1]):
            a = 0.5 if X.shape[2] == 3 else 0.0  # T2 泄漏 α=0.5, T1 无泄漏
            h = (1 - a) * torch.tanh(X[:, t] @ W1.T + h @ W2.T + self.b) + a * h
            ys.append(h @ self.Wo.T + self.bo)
        return torch.stack(ys, 1)

def prune_to_budget(model):
    with torch.no_grad():
        for W, m, E in ((model.Wih, model.mih, E_IH), (model.Whh, model.mhh, E_HH)):
            thr = torch.kthvalue(W.abs().flatten(), W.numel() - E).values
            m.copy_((W.abs() >= thr).float())

def run(task, cond, seed):
    rng = np.random.default_rng(seed * 1000 + hash(cond) % 997)
    torch.manual_seed(seed * 77 + 3)
    n_in = R_IN if task == "T1" else 3
    mih, mhh = build_masks(cond, rng, n_in)
    if task == "T1":
        Xtr, ytr = data_motion(1024, rng); Xte, yte = data_motion(1024, rng)
        model = SparseRNN(R_IN, 4, mih, mhh); lossf = nn.CrossEntropyLoss()
        epochs, prune_at = 18, 7
    else:
        Xtr, ytr = data_heading(1024, rng); Xte, yte = data_heading(256, rng)
        model = SparseRNN(3, 2, mih, mhh); lossf = nn.MSELoss()
        epochs, prune_at = 12, 5
    opt = torch.optim.Adam(model.parameters(), 5e-3 if task=='T2' else 3e-3)
    for ep in range(epochs):
        if cond == "pruned" and ep == prune_at:
            prune_to_budget(model)
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 128):
            idx = perm[s:s + 128]
            out = model(Xtr[idx])
            loss = lossf(out[:, -1] if task == "T1" else out, ytr[idx] if task == "T1" else ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = model(Xte)
        if task == "T1":
            acc = (out[:, -1].argmax(1) == yte).float().mean().item()
            return {"task": task, "cond": cond, "seed": seed, "acc": round(acc, 4),
                    "edges": int(model.mih.sum() + model.mhh.sum())}
        pred = out[:, -12:, :]; tgt = yte[:, -12:, :]
        ang_p = torch.atan2(pred[:, :, 0], pred[:, :, 1])
        ang_t = torch.atan2(tgt[:, :, 0], tgt[:, :, 1])
        d = (ang_p - ang_t + math.pi) % (2 * math.pi) - math.pi
        rms = d.pow(2).mean().sqrt().item()
        return {"task": task, "cond": cond, "seed": seed, "rms": round(rms, 4),
                "edges": int(model.mih.sum() + model.mhh.sum())}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["T1", "T2"], required=True)
    ap.add_argument("--cond", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    r = run(a.task, a.cond, a.seed)
    r["sec"] = round(time.time() - t0, 1)
    print(json.dumps(r, ensure_ascii=False), flush=True)
