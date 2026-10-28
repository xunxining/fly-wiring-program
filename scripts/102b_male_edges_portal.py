#!/usr/bin/env python3
"""102b — 雄侧边表直写门户: 4MB 分块 + fsync + 断点重试
挂载对 100MB 连续写不稳定, 但小块+校验可通过。重算置换(~5s)后直写 e_post/e_w。
"""
import sys; sys.path.insert(0, "/mnt/agents/pylibs")
import numpy as np, os, time
R = "/mnt/agents/output/果蝇生成器/"
DC = R + "data_census/"; PDM = R + "data_male/"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

node_type = np.fromfile(DC+"node_type.i32", dtype=np.int32)
typed = np.where(node_type >= 0)[0]
o = np.argsort(node_type[typed], kind="stable")
new_of_old = np.full(len(node_type), -1, np.int32)
new_of_old[typed[o]] = np.arange(len(typed))
log("置换重建完成")

def write_chunked(name, arr):
    dst = PDM + name
    want = arr.nbytes
    CH = 4*1024*1024
    raw = arr.tobytes()
    for att in range(60):
        try:
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
            if have == want:
                # 校验尾块
                with open(dst, "rb") as f:
                    f.seek(want-64); tail = f.read()
                if tail == raw[-64:]:
                    log(f"{name} 已完整 {want:,}"); return True
            mode = "r+b" if 0 < have < want else "wb"
            with open(dst, mode) as f:
                if mode == "r+b": f.seek(have)
                pos = f.tell()
                while pos < want:
                    chunk = raw[pos:pos+CH]
                    f.write(chunk); f.flush(); os.fsync(f.fileno())
                    pos += len(chunk)
            if os.path.getsize(dst) == want:
                log(f"{name} 写入完成 {want:,} (att {att})"); return True
        except OSError as e:
            log(f"{name} retry{att}: {str(e)[:50]}")
            time.sleep(10)
    return False

for name, src in [("e_post.i32", "e_post.i32"), ("e_w.i32", "e_w.i32")]:
    a = np.fromfile(DC+src, dtype=np.int32)
    if name != "e_w.i32":
        a = new_of_old[a]
    ok = write_chunked(name, a)
    log(f"{name}: {'OK' if ok else 'FAIL'}")
    del a
print("DONE", flush=True)
