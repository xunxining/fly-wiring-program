#!/usr/bin/env python3
"""Stage 11: 精确互惠性 —— 对称哈希分桶, 全 memmap 流式, 小内存
反向边 (b,a) 与 (a,b) 必落同一桶 (桶号由 (min,max) 决定), 桶内集合精确判定
"""
import numpy as np
import os, json, gc

D = "/tmp/flydata/"
OUT = "/tmp/flydata/gen/"
NB = 256
CH = 10_000_000

def log(*a): print(*a, flush=True)

def recip_exact(pre_path, post_path, tag):
    pre = np.memmap(pre_path, dtype=np.int32, mode="r")
    post = np.memmap(post_path, dtype=np.int32, mode="r")
    E = len(pre)
    Ng = int(max(pre.max(), post.max())) + 1 if E < 50_000_000 else None
    # 节点数上限: 用文件大小估算不够, 直接扫最大值太慢 -> 用 int64 key 即可, 桶号只依赖 (min,max)
    bdir = f"/tmp/flydata/rb_{tag}/"
    os.makedirs(bdir, exist_ok=True)
    fps = [open(bdir + f"{i}.i64", "wb") for i in range(NB)]
    n_noloop = 0
    for i0 in range(0, E, CH):
        i1 = min(i0 + CH, E)
        a = np.asarray(pre[i0:i1], dtype=np.int64)
        b = np.asarray(post[i0:i1], dtype=np.int64)
        keep = a != b
        a, b = a[keep], b[keep]
        n_noloop += len(a)
        mn = np.minimum(a, b); mx = np.maximum(a, b)
        bk = (mn * 2654435761 + mx * 40503) % NB
        order = np.argsort(bk, kind="stable")
        bk_s = bk[order]
        ka = a[order]; kb = b[order]
        bounds = np.searchsorted(bk_s, np.arange(NB + 1))
        for i in range(NB):
            s, e = bounds[i], bounds[i + 1]
            if e > s:
                (ka[s:e] * (1 << 31) + kb[s:e]).tofile(fps[i])
        if i0 % (CH * 5) == 0:
            log(f"  {tag} bucket pass {i0:,}/{E:,}")
    for f in fps:
        f.close()
    hits = 0
    for i in range(NB):
        fp = bdir + f"{i}.i64"
        if os.path.getsize(fp) == 0:
            continue
        keys = np.fromfile(fp, dtype=np.int64)
        s = set(keys.tolist())
        a = keys // (1 << 31); b = keys % (1 << 31)
        rev = b * (1 << 31) + a
        hits += sum(1 for r in rev.tolist() if r in s)
        del keys, s, rev
        gc.collect()
    for i in range(NB):
        os.remove(bdir + f"{i}.i64")
    os.rmdir(bdir)
    r = hits / n_noloop if n_noloop else 0.0
    log(f"{tag}: edges(no loop)={n_noloop:,}, reciprocal={hits:,}, reciprocity={r:.4f}")
    return r, n_noloop, hits

res = {}
res["real"] = recip_exact(D + "e_pre.i32", D + "e_post.i32", "real")
res["gen_S10"] = recip_exact(OUT + "gen_S10_pre.i32", OUT + "gen_S10_post.i32", "s10")
json.dump({"real_recip": res["real"][0], "real_edges": res["real"][1],
           "s10_recip": res["gen_S10"][0], "s10_edges": res["gen_S10"][1]},
          open(OUT + "recip_exact.json", "w"), indent=2)
log("RECIP DONE")
