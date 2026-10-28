#!/usr/bin/env python3
"""并行分块下载 pyarrow wheel"""
import os, time, urllib.request, threading

URL = "https://files.pythonhosted.org/packages/60/c9/711ca85d79f1ec98f29a5eae2b051e25b4ecec5de3e3c0e2d5c5dcb15664/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl"
OUT = "/tmp/pyarrow.whl"
N = 8

req = urllib.request.Request(URL, method="HEAD")
with urllib.request.urlopen(req, timeout=30) as r:
    total = int(r.headers["Content-Length"])
print("total:", total, flush=True)
chunk = total // N + 1

def fetch(i):
    s, e = i * chunk, min((i + 1) * chunk - 1, total - 1)
    p = f"/tmp/wheel_part_{i}"
    target = e - s + 1
    while True:
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if have >= target:
            return
        rq = urllib.request.Request(URL, headers={"Range": f"bytes={s + have}-{e}"})
        try:
            with urllib.request.urlopen(rq, timeout=30) as r, open(p, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
        except Exception:
            time.sleep(2)

ths = [threading.Thread(target=fetch, args=(i,), daemon=True) for i in range(N)]
[t.start() for t in ths]
[t.join() for t in ths]

with open(OUT, "wb") as fo:
    for i in range(N):
        with open(f"/tmp/wheel_part_{i}", "rb") as fi:
            fo.write(fi.read())
        os.remove(f"/tmp/wheel_part_{i}")
print("SIZE:", os.path.getsize(OUT), "OK" if os.path.getsize(OUT) == total else "BAD", flush=True)
