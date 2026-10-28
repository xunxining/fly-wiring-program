#!/usr/bin/env python3
"""P10: 真实果蝇环路掩码 vs 等边数基线 (复用 p9 的任务与训练)。
用法: python3 p10.py --task T1|T2 --arm real|erdos_m|ws_m|qprog_m --seed S
T1 用 realmed 掩码 (视叶运动通路), T2 用 realcx 掩码 (雄性 CX 环, CC-BY)。
"""
import argparse, json, math, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '/tmp/p9')
import p9  # 复用 mask_erdos/mask_ws/mask_qprog/data_motion/data_heading

torch.set_num_threads(2)
D_IH = 0.125

class SparseRNN(nn.Module):
    def __init__(self, n_in, n_out, mih, mhh):
        super().__init__()
        Hh = mhh.shape[0]
        self.mih = torch.tensor(mih); self.mhh = torch.tensor(mhh)
        self.Wih = nn.Parameter(torch.randn(Hh, n_in) * 0.1)
        self.Whh = nn.Parameter(torch.randn(Hh, Hh) * 0.05)
        self.b = nn.Parameter(torch.zeros(Hh))
        self.Wo = nn.Parameter(torch.randn(n_out, Hh) * 0.1)
        self.bo = nn.Parameter(torch.zeros(n_out))
        with torch.no_grad():
            sr = np.abs(np.linalg.eigvals((self.Whh * self.mhh).numpy())).max()
            self.Whh.data *= 0.95 / max(sr, 1e-6)
    def forward(self, X):
        h = torch.zeros(X.shape[0], self.mhh.shape[0])
        W1 = self.Wih * self.mih; W2 = self.Whh * self.mhh
        ys = []
        for t in range(X.shape[1]):
            a = 0.5 if X.shape[2] == 3 else 0.0
            h = (1 - a) * torch.tanh(X[:, t] @ W1.T + h @ W2.T + self.b) + a * h
            ys.append(h @ self.Wo.T + self.bo)
        return torch.stack(ys, 1)

def build(task, arm, rng):
    name = 'realmed' if task == 'T1' else 'realcx'
    real = np.load(f'/tmp/p9/mask_{name}.npy')
    Hh = real.shape[0]; E = int(real.sum())
    n_in = 64 if task == 'T1' else 3
    e_ih = int(n_in * Hh * D_IH)
    mih = p9.mask_qprog(Hh, n_in, e_ih, rng, False)
    if arm == 'real':
        return mih, real
    gen = {'erdos_m': p9.mask_erdos, 'ws_m': p9.mask_ws}[arm] if arm != 'qprog_m' else None
    mhh = gen(Hh, Hh, E, rng) if gen else p9.mask_qprog(Hh, Hh, E, rng, True)
    return mih, mhh

def run(task, arm, seed):
    rng = np.random.default_rng(seed * 1000 + hash(arm) % 997)
    torch.manual_seed(seed * 77 + 3)
    mih, mhh = build(task, arm, rng)
    if task == 'T1':
        Xtr, ytr = p9.data_motion(1024, rng); Xte, yte = p9.data_motion(1024, rng)
        model = SparseRNN(64, 4, mih, mhh); lossf = nn.CrossEntropyLoss()
        epochs, lr = 12, 3e-3
    else:
        Xtr, ytr = p9.data_heading(1024, rng); Xte, yte = p9.data_heading(256, rng)
        model = SparseRNN(3, 2, mih, mhh); lossf = nn.MSELoss()
        epochs, lr = 30, 5e-3
    opt = torch.optim.Adam(model.parameters(), lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), 128):
            idx = perm[s:s + 128]
            out = model(Xtr[idx])
            loss = lossf(out[:, -1] if task == 'T1' else out, ytr[idx] if task == 'T1' else ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = model(Xte)
        if task == 'T1':
            acc = (out[:, -1].argmax(1) == yte).float().mean().item()
            return {"task": task, "arm": arm, "seed": seed, "acc": round(acc, 4),
                    "edges": int(model.mih.sum() + model.mhh.sum())}
        pred = out[:, -12:, :]; tgt = yte[:, -12:, :]
        d = (torch.atan2(pred[:, :, 0], pred[:, :, 1]) - torch.atan2(tgt[:, :, 0], tgt[:, :, 1]) + math.pi) % (2 * math.pi) - math.pi
        return {"task": task, "arm": arm, "seed": seed, "rms": round(d.pow(2).mean().sqrt().item(), 4),
                "edges": int(model.mih.sum() + model.mhh.sum())}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['T1', 'T2'], required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, required=True)
    a = ap.parse_args()
    t0 = time.time()
    r = run(a.task, a.arm, a.seed)
    r['sec'] = round(time.time() - t0, 1)
    print(json.dumps(r, ensure_ascii=False), flush=True)
