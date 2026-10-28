#!/usr/bin/env python3
"""Stage 2 (内存优化版): 拟合化学亲和力模型"""
import numpy as np
import pandas as pd
import torch
import time, os, json, gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(2)

def log(*a): print(*a, flush=True)

d = np.load(D + "typepair.npz")
uk = d['uk'].astype(np.int64)
cnt = d['cnt'].astype(np.int64)
type_count = d['type_count'].astype(np.int64)
T = len(type_count)
tp = uk // T
tq = uk % T
possible = type_count[tp] * type_count[tq]
log("loaded:", len(uk), "type pairs")

nodes = pd.read_feather(D + "nodes.feather", columns=["type"])
types = sorted(nodes['type'].dropna().unique())
np.save(OUT + "type_names.npy", np.array(types, dtype=object))
del nodes; gc.collect()

rng = np.random.default_rng(42)
perm = rng.permutation(len(uk))
n_hold = int(0.1 * len(uk))
hold_idx, train_idx = perm[:n_hold], perm[n_hold:]

K = 32
NEG = 8_000_000
EPOCHS = 5
BS = 500_000
LR = 0.05
dev = "cpu"
log("setup done")

obs_keys = np.sort(uk)

def make_neg(n, seed):
    r = np.random.default_rng(seed)
    keep_s, keep_t = [], []
    done = 0
    while done < n:
        m = min(2_000_000, n - done)
        ns = r.integers(0, T, m)
        nt = r.integers(0, T, m)
        key = ns.astype(np.int64) * T + nt
        bad = np.isin(key, obs_keys)
        keep_s.append(ns[~bad]); keep_t.append(nt[~bad])
        done += m
    return np.concatenate(keep_s), np.concatenate(keep_t)

neg_s_np, neg_t_np = make_neg(NEG, 1)
neg_frac = T * T - len(uk)
corr = neg_frac / len(neg_s_np)
log("neg pairs:", len(neg_s_np), "corr:", round(float(corr), 1))

def T_(x, dt=torch.long): return torch.tensor(x, dtype=dt, device=dev)
tr_s, tr_t = T_(tp[train_idx]), T_(tq[train_idx])
tr_c = T_(cnt[train_idx], torch.float32)
tr_p = T_(possible[train_idx], torch.float32)
ng_s, ng_t = T_(neg_s_np), T_(neg_t_np)
del neg_s_np, neg_t_np, uk, cnt, possible, d; gc.collect()
log("tensors ready")

U = torch.nn.Parameter(torch.randn(T, K) * 0.05)
V = torch.nn.Parameter(torch.randn(T, K) * 0.05)
A = torch.nn.Parameter(torch.zeros(T))
Bp = torch.nn.Parameter(torch.zeros(T))
G = torch.nn.Parameter(torch.tensor(-3.0))
opt = torch.optim.Adam([U, V, A, Bp, G], lr=LR)

t0 = time.time()
for ep in range(EPOCHS):
    perm_t = torch.randperm(len(tr_s))
    l_acc, nb = 0.0, 0
    for i in range(0, len(tr_s), BS):
        idx = perm_t[i:i+BS]
        s, t = tr_s[idx], tr_t[idx]
        lg = (U[s] * V[t]).sum(1) + A[s] + Bp[t] + G
        p = torch.sigmoid(lg)
        c, ps = tr_c[idx], tr_p[idx]
        nll_pos = (c * torch.log(p + 1e-9) + (ps - c) * torch.log(1 - p + 1e-9)) / ps
        j = torch.randint(0, len(ng_s), (len(idx),))
        ns2, nt2 = ng_s[j], ng_t[j]
        nlg = (U[ns2] * V[nt2]).sum(1) + A[ns2] + Bp[nt2] + G
        nll_neg = torch.log(1 - torch.sigmoid(nlg) + 1e-9) * corr
        loss = -(nll_pos.mean() + nll_neg.mean())
        opt.zero_grad(); loss.backward(); opt.step()
        l_acc += loss.item(); nb += 1
    log(f"epoch {ep}: nll={l_acc/nb:.5f} ({time.time()-t0:.0f}s)")

with torch.no_grad():
    s_h, t_h = T_(tp[hold_idx]), T_(tq[hold_idx])
    score_pos = ((U[s_h] * V[t_h]).sum(1) + A[s_h] + Bp[t_h] + G).numpy()
    hn_s, hn_t = make_neg(len(hold_idx) * 2, 7)
    s_n, t_n = T_(hn_s), T_(hn_t)
    score_neg = ((U[s_n] * V[t_n]).sum(1) + A[s_n] + Bp[t_n] + G).numpy()

from sklearn.metrics import roc_auc_score
y = np.r_[np.ones(len(score_pos)), np.zeros(len(score_neg))]
auc = roc_auc_score(y, np.r_[score_pos, score_neg])
log(f"held-out type-pair AUC: {auc:.4f}")

torch.save({"U": U, "V": V, "A": A, "B": Bp, "G": G, "T": T, "K": K, "auc": float(auc)},
           OUT + "affinity.pt")
with open(OUT + "fit_meta.json", "w") as f:
    json.dump({"T": T, "K": K, "auc": float(auc), "epochs": EPOCHS}, f)
log("FIT DONE")
