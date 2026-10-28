#!/usr/bin/env python3
"""工作队列式分块下载器：小块+多工人+逐块续传"""
import os, time, urllib.request, queue, threading

URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
TOTAL = 1051241946
DIR = "/tmp/flydata"
BLOCK = 8 * 1024 * 1024
WORKERS = 10

# 已完成的旧 part 文件直接当作前缀保留；把每个 part 剩余部分切成 block
q = queue.Queue()
tasks = []  # (block_file, start, end)
N = 8
CHUNK = TOTAL // N + 1
for part in range(N):
    pstart = part * CHUNK
    pend = min((part + 1) * CHUNK - 1, TOTAL - 1)
    pfile = os.path.join(DIR, f"part_{part}")
    have = os.path.getsize(pfile) if os.path.exists(pfile) else 0
    if pstart + have > pend:
        continue
    pos = pstart + have
    bi = 0
    # 已下载的 have 字节保留在 part_i 中作为第 0 段
    while pos <= pend:
        be = min(pos + BLOCK - 1, pend)
        bf = os.path.join(DIR, f"blk_{part}_{bi}")
        tasks.append((bf, pos, be))
        pos = be + 1
        bi += 1

for t in tasks:
    q.put(t)
print(f"blocks to fetch: {len(tasks)}", flush=True)

def worker():
    while True:
        try:
            bf, s, e = q.get_nowait()
        except queue.Empty:
            return
        target = e - s + 1
        while True:
            have = os.path.getsize(bf) if os.path.exists(bf) else 0
            if have >= target:
                break
            req = urllib.request.Request(URL, headers={"Range": f"bytes={s + have}-{e}"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r, open(bf, "ab") as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
            except Exception:
                time.sleep(2)
        q.task_done()

ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
[t.start() for t in ths]
[t.join() for t in ths]

# 按序合并 part_i 前缀 + blk_i_*
out = os.path.join(DIR, "connectome-weights.feather")
with open(out, "wb") as fo:
    for part in range(N):
        pstart = part * CHUNK
        pend = min((part + 1) * CHUNK - 1, TOTAL - 1)
        pfile = os.path.join(DIR, f"part_{part}")
        if os.path.exists(pfile):
            with open(pfile, "rb") as fi:
                fo.write(fi.read())
        bi = 0
        while True:
            bf = os.path.join(DIR, f"blk_{part}_{bi}")
            if not os.path.exists(bf):
                break
            with open(bf, "rb") as fi:
                fo.write(fi.read())
            bi += 1
size = os.path.getsize(out)
print("FINAL SIZE:", size, "OK" if size == TOTAL else "MISMATCH", flush=True)
if size == TOTAL:
    for f in os.listdir(DIR):
        if f.startswith("blk_") or f.startswith("part_"):
            os.remove(os.path.join(DIR, f))
print("ALLDONE", flush=True)
