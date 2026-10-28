#!/usr/bin/env python3
"""分块可断点续传下载 MaleCNS connectome-weights"""
import os, time, urllib.request, threading

URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
TOTAL = 1051241946
N = 8
DIR = "/tmp/flydata"
CHUNK = TOTAL // N + 1

def fetch(part):
    start = part * CHUNK
    end = min((part + 1) * CHUNK - 1, TOTAL - 1)
    path = os.path.join(DIR, f"part_{part}")
    target = end - start + 1
    while True:
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= target:
            return
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
        except Exception as e:
            print(f"part_{part} retry: {e}", flush=True)
            time.sleep(3)

threads = [threading.Thread(target=fetch, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in threads]
[t.join() for t in threads]

# 合并
out = os.path.join(DIR, "connectome-weights.feather")
with open(out, "wb") as fo:
    for i in range(N):
        p = os.path.join(DIR, f"part_{i}")
        with open(p, "rb") as fi:
            while True:
                b = fi.read(1 << 24)
                if not b:
                    break
                fo.write(b)
size = os.path.getsize(out)
print("FINAL SIZE:", size, "OK" if size == TOTAL else "MISMATCH", flush=True)
if size == TOTAL:
    for i in range(N):
        os.remove(os.path.join(DIR, f"part_{i}"))
print("ALLDONE", flush=True)
