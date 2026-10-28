#!/usr/bin/env python3
import os, time, urllib.request, queue, threading
URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-stats-male-cns-v1.0-minconf-0.5.feather"
TOTAL = 778062826
DIR = "/tmp/flydata"
BLOCK = 8 * 1024 * 1024
WORKERS = 8
OUT = os.path.join(DIR, "body-stats.feather")

q = queue.Queue()
pos = 0
bi = 0
while pos < TOTAL:
    be = min(pos + BLOCK - 1, TOTAL - 1)
    q.put((os.path.join(DIR, f"sb_{bi}"), pos, be))
    pos = be + 1
    bi += 1
nblocks = bi
print("blocks:", nblocks, flush=True)

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
with open(OUT, "wb") as fo:
    for i in range(nblocks):
        bf = os.path.join(DIR, f"sb_{i}")
        with open(bf, "rb") as fi:
            fo.write(fi.read())
size = os.path.getsize(OUT)
print("FINAL:", size, "OK" if size == TOTAL else "MISMATCH", flush=True)
if size == TOTAL:
    for i in range(nblocks):
        os.remove(os.path.join(DIR, f"sb_{i}"))
print("ALLDONE", flush=True)
