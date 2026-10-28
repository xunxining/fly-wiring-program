#!/usr/bin/env python3
# 120b_sort_p9b.py — P9b 排序: gm_p9b → gm_p9bs (复用 120 的 sort_triple 逻辑)
import os, time
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器"
t0 = time.time()
def log(*a):
    try: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
    except OSError: pass

def cp_chunked(src, dst):
    with open(src, "rb") as f, open(dst, "wb") as g:
        while True:
            b = f.read(1 << 24)
            if not b: break
            g.write(b)

def atomic_write(arr, dst):
    att = 0
    while True:
        try:
            arr.tofile("/tmp/sort9b_stage.i32")
            cp_chunked("/tmp/sort9b_stage.i32", dst + ".tmp")
            if os.path.getsize(dst + ".tmp") == arr.size * 4:
                os.replace(dst + ".tmp", dst)
                return
        except OSError as e:
            att += 1
            if att % 10 == 1: log(f"write retry {att}: errno {e.errno}")
            time.sleep(3)

def size_of(p):
    for _ in range(20):
        try: return os.path.getsize(p)
        except OSError: time.sleep(3)
    return None

nm = np.fromfile(f"{ROOT}/data_male/node_type.i32", dtype=np.int32).size
op = [f"{ROOT}/gen_edges/gm_p9bs_{k}.i32" for k in ("pre", "post", "w")]
E = size_of(f"{ROOT}/gen_edges/gm_p9b_pre.i32") // 4
if all(size_of(p) == E * 4 for p in op):
    log("gm_p9bs already sorted, skip")
else:
    pre = np.fromfile(f"{ROOT}/gen_edges/gm_p9b_pre.i32", dtype=np.int32)
    post = np.fromfile(f"{ROOT}/gen_edges/gm_p9b_post.i32", dtype=np.int32)
    w = np.fromfile(f"{ROOT}/gen_edges/gm_p9b_w.i32", dtype=np.int32)
    kk = pre.astype(np.int64) * nm + post
    order = np.argsort(kk, kind="stable")
    log(f"gm_p9bs: sort key done ({E:,} edges)")
    atomic_write(pre[order], op[0]); log("pre written")
    atomic_write(post[order], op[1]); log("post written")
    atomic_write(w[order], op[2]); log("w written")
log("DONE")
