#!/usr/bin/env python3
"""Stage 20: 功能对照实验核心 —— 蘑菇体式 few-shot 学习
模型: PN(输入) -> KC(稀疏扩展) -> MBON(可学习读出)
五组对照仅替换连接矩阵来源, 学习规则与协议完全一致
"""
import numpy as np
import gzip, struct, os, json

D = "/tmp/flydata/"

def log(*a): print(*a, flush=True)

# ---------- MNIST ----------
def load_mnist():
    def imgs(f):
        with gzip.open(f, "rb") as g:
            magic, n, r, c = struct.unpack(">IIII", g.read(16))
            return np.frombuffer(g.read(), np.uint8).reshape(n, r * c).astype(np.float32) / 255.0
    def lbls(f):
        with gzip.open(f, "rb") as g:
            magic, n = struct.unpack(">II", g.read(8))
            return np.frombuffer(g.read(), np.uint8)
    Xtr = imgs(D + "train-images-idx3-ubyte.gz")
    ytr = lbls(D + "train-labels-idx1-ubyte.gz")
    Xte = imgs(D + "t10k-images-idx3-ubyte.gz")
    yte = lbls(D + "t10k-labels-idx1-ubyte.gz")
    return Xtr, ytr, Xte, yte

# ---------- 电路模拟器 ----------
class OlfactoryCircuit:
    """rate 模型: KC 稀疏阈值激活, KC->MBON 可学 (perceptron 规则)
    conn_pn_kc: (n_kc, n_pn) 稀疏连接 (bool 或权重)
    w_kc_mb:    (n_cls, n_kc) 初始读出权重 (拓扑固定, 强度可学)
    """
    def __init__(self, pn_kc_weight, kc_mb_topo, n_cls=10, kc_frac=0.05, seed=0):
        self.W_pk = pn_kc_weight          # float32 (n_kc, n_pn)
        self.topo = kc_mb_topo            # bool (n_cls, n_kc) 允许学习的突触位置
        self.n_kc = self.W_pk.shape[0]
        self.n_cls = n_cls
        self.kc_frac = kc_frac
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((n_cls, self.n_kc), dtype=np.float32)

    def encode(self, X):
        """X: (n, n_pn) -> KC 稀疏编码 (n, n_kc) bool"""
        act = X @ self.W_pk.T                       # (n, n_kc)
        # 每样本取 top frac 为激活 (赢家抑制, 模拟 KC 的 APL 全局抑制)
        k = max(1, int(self.kc_frac * self.n_kc))
        thr = np.partition(act, self.n_kc - k, axis=1)[:, self.n_kc - k][:, None]
        return (act >= thr).astype(np.float32)

    def train_sample(self, kc, y, lr=1.0):
        """单样本 perceptron 更新, 只允许 topo 位置的突触"""
        scores = self.W @ kc                        # (n_cls,)
        pred = int(np.argmax(scores))
        if pred != y:
            self.W[y] += lr * kc * self.topo[y]
            self.W[pred] -= lr * kc * self.topo[pred]
            return 0
        return 1

    def predict(self, X, bs=2000):
        outs = []
        for i in range(0, len(X), bs):
            kc = self.encode(X[i:i + bs])
            outs.append(np.argmax(kc @ self.W.T, axis=1))
        return np.concatenate(outs)

# ---------- 对照组连接构造 ----------
def make_pn_kc_random(n_pn, n_kc, fanin=6, seed=0):
    """组5: 经典随机投影 —— 每 KC 随机采 fanin 个 PN, 均匀权重"""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_kc, n_pn), dtype=np.float32)
    for k in range(n_kc):
        idx = rng.choice(n_pn, min(fanin, n_pn), replace=False)
        W[k, idx] = 1.0
    return W

def rewire_deg_preserving(W_pk, n_swap=200000, seed=1):
    """组4: Maslov-Sneppen 度保持重连 (二部图)"""
    rng = np.random.default_rng(seed)
    W = (W_pk > 0).astype(np.float32)
    ks, ps = np.nonzero(W)
    E = len(ks)
    w_vals = W[ks, ps].copy()
    ks = ks.copy(); ps = ps.copy()
    for _ in range(n_swap):
        i, j = rng.integers(0, E, 2)
        ki, pi_, kj, pj = ks[i], ps[i], ks[j], ps[j]
        if ki == kj or pi_ == pj:
            continue
        if W[ki, pj] > 0 or W[kj, pi_] > 0:
            continue
        W[ki, pi_] = 0; W[kj, pj] = 0
        W[ki, pj] = w_vals[i]; W[kj, pi_] = w_vals[j]
        ps[i], ps[j] = pj, pi_
    return W

def make_kc_mb_topo_full(n_cls, n_kc):
    return np.ones((n_cls, n_kc), dtype=np.float32)

def make_kc_mb_topo_from_graph(kc_mb_weight, n_cls, kc_ids=None):
    """由真实/生成图的 KC->MBON 拓扑得到允许学习位置; MBON 映射到 n_cls 个读出槽"""
    return (kc_mb_weight > 0).astype(np.float32)

# ---------- 实验协议 ----------
def fewshot_eval(circuit, Xtr, ytr, Xte, yte, shots=(1, 5, 10), n_rep=5, seed=0):
    """每类随机取 k 张训练, 测测试集准确率; 重复 n_rep 次取均值±std
    返回 {k: (mean, std, synops)}; synops = 训练期突触更新总数(能耗代理)
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k in shots:
        accs, ops = [], []
        for rep in range(n_rep):
            circuit.W[:] = 0
            idx = []
            for c in range(10):
                pool = np.where(ytr == c)[0]
                idx.extend(rng.choice(pool, k, replace=False).tolist())
            rng.shuffle(idx)
            upd = 0
            for i in idx:
                kc = circuit.encode(Xtr[i:i + 1])[0]
                before = circuit.W.sum()
                correct = circuit.train_sample(kc, int(ytr[i]))
                if not correct:
                    upd += 1
            acc = (circuit.predict(Xte) == yte).mean()
            accs.append(acc); ops.append(upd)
        out[k] = (float(np.mean(accs)), float(np.std(accs)), float(np.mean(ops)))
        log(f"  {k}-shot: acc={out[k][0]:.4f}±{out[k][1]:.4f}, updates={out[k][2]:.1f}")
    return out
