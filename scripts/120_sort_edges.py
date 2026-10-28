#!/usr/bin/env python3
# 120_sort_edges.py — 一次性排序边集(幂等+原子写), 供 117 csr 切片使用
# 输出: gen_edges/{g_dyad3s,gm_p8cs}_{pre,post,w}.i32  (按 pre*N+post 升序, w 跟随)
#       data_male/e_{pre,post}_s.i32                  (雄性真实边同样排序)
import os, time
import numpy as np

ROOT = "/mnt/agents/output/果蝇生成器"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

def cp_chunked(src, dst):
    with open(src, "rb") as f, open(dst, "wb") as g:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            g.write(b)

def atomic_write(arr, dst):
    att = 0
    while True:
        try:
            arr.tofile("/tmp/sort_stage.i32")
            cp_chunked("/tmp/sort_stage.i32", dst + ".tmp")
            if os.path.getsize(dst + ".tmp") == arr.size * 4:
                os.replace(dst + ".tmp", dst)
                return
        except OSError as e:
            att += 1
            if att % 10 == 1:
                log(f"write retry {att}: errno {e.errno}")
            time.sleep(3)

def size_of(p):
    for _ in range(20):
        try:
            return os.path.getsize(p)
        except OSError:
            time.sleep(3)
    return None

def sort_triple(pre_path, post_path, w_path, out_prefix, N):
    op = [f"{out_prefix}_{k}.i32" for k in ("pre", "post", "w")]
    E = size_of(pre_path) // 4
    if all(size_of(p) == E * 4 for p in op):
        log(f"{out_prefix} already sorted, skip")
        return
    pre = np.fromfile(pre_path, dtype=np.int32)
    post = np.fromfile(post_path, dtype=np.int32)
    w = np.fromfile(w_path, dtype=np.int32)
    kk = pre.astype(np.int64) * N + post
    order = np.argsort(kk, kind="stable")
    log(f"{out_prefix}: sort key done ({E:,} edges)")
    atomic_write(pre[order], op[0]); log("pre written")
    atomic_write(post[order], op[1]); log("post written")
    atomic_write(w[order], op[2]); log("w written")
    del pre, post, w, kk, order

def sort_pair(pre_path, post_path, opre, opost, N):
    E = size_of(pre_path) // 4
    if size_of(opre) == E * 4 and size_of(opost) == E * 4:
        log(f"{opre} already sorted, skip")
        return
    pre = np.fromfile(pre_path, dtype=np.int32)
    post = np.fromfile(post_path, dtype=np.int32)
    kk = pre.astype(np.int64) * N + post
    order = np.argsort(kk, kind="stable")
    log(f"{opre}: sort key done ({E:,} edges)")
    atomic_write(pre[order], opre); log("pre written")
    atomic_write(post[order], opost); log("post written")
    del pre, post, kk, order

# 雌性真实边已验证升序 ⇒ 跳过; 雄性真实边需要排序
nf = np.fromfile(f"{ROOT}/data_female/node_type.i32", dtype=np.int32).size
nm = np.fromfile(f"{ROOT}/data_male/node_type.i32", dtype=np.int32).size
sort_triple(f"{ROOT}/gen_edges/g_dyad3_pre.i32", f"{ROOT}/gen_edges/g_dyad3_post.i32",
            f"{ROOT}/gen_edges/g_dyad3_w.i32", f"{ROOT}/gen_edges/g_dyad3s", nf)
sort_triple(f"{ROOT}/gen_edges/gm_p8c_pre.i32", f"{ROOT}/gen_edges/gm_p8c_post.i32",
            f"{ROOT}/gen_edges/gm_p8c_w.i32", f"{ROOT}/gen_edges/gm_p8cs", nm)
sort_pair(f"{ROOT}/data_male/e_pre.i32", f"{ROOT}/data_male/e_post.i32",
          f"{ROOT}/data_male/e_pre_s.i32", f"{ROOT}/data_male/e_post_s.i32", nm)
open(f"{ROOT}/sort_edges_done", "w").write("ok")
log("ALL DONE")
