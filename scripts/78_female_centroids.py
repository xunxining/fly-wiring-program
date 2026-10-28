#!/usr/bin/env python3
"""78_female_centroids.py — 从 flywire_synapses_783.feather 随机抽批计算逐神经元突触云质心
HTTP Range 随机访问 feather (pyarrow PythonFile), 6 线程并行, 每 3 批取 1 (~33% 突触)
输出: data_female/xyz_arbor.f32 (N,3 um, 基因组序, NaN=缺), centroids_diag.json
"""
import sys, os, json, time, urllib.request, threading
sys.path.insert(0, "/mnt/agents/output/果蝇生成器/vendor")
import numpy as np
import pyarrow as pa
from concurrent.futures import ThreadPoolExecutor

D = "/mnt/agents/output/果蝇生成器/data_female/"
URL = "https://zenodo.org/api/records/10676866/files/flywire_synapses_783.feather/content"
EVERY = 3          # 每 3 批取 1
NTHREAD = 6
def log(*a): print(time.strftime("[%H:%M:%S]"), *a, flush=True)

class HttpRangeFile:
    def __init__(self, url, chunk=4*1024*1024):
        self.url = url; self.pos = 0; self.chunk = chunk
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            self.size = int(r.headers["Content-Length"])
        self._cache_off = -1; self._cache = b""
    def tell(self): return self.pos
    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else (self.pos + pos if whence == 1 else self.size + pos)
        return self.pos
    def read(self, nbytes=-1):
        if nbytes < 0: nbytes = self.size - self.pos
        out = []
        while nbytes > 0 and self.pos < self.size:
            coff = (self.pos // self.chunk) * self.chunk
            if coff != self._cache_off:
                end = min(coff + self.chunk, self.size) - 1
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={coff}-{end}"})
                ok = False
                for _t in range(8):
                    try:
                        with urllib.request.urlopen(req, timeout=120) as r:
                            self._cache = r.read(); ok = True; break
                    except Exception:
                        time.sleep(2 + _t)
                if not ok: raise IOError("range fetch failed")
                self._cache_off = coff
            s = self.pos - coff
            take = min(nbytes, len(self._cache) - s)
            out.append(self._cache[s:s + take])
            self.pos += take; nbytes -= take
        return b"".join(out)
    def close(self): pass
    @property
    def closed(self): return False
    def readable(self): return True
    def seekable(self): return True

# census lookup
node_ids = np.fromfile(D + "node_ids.i64", dtype=np.int64)
N = len(node_ids)
srt = np.argsort(node_ids)
gids = node_ids[srt]
def map_ids(arr):
    p = np.clip(np.searchsorted(gids, arr), 0, N - 1)
    return np.where(gids[p] == arr, srt[p], -1)

# probe batch count
log("probing footer...")
f0 = HttpRangeFile(URL)
log(f"size={f0.size/1e9:.2f}GB, opening reader...")
r0 = pa.ipc.open_file(pa.PythonFile(f0, mode="r"))
NB = r0.num_record_batches
del r0, f0
log("probe done")
todo = list(range(0, NB, EVERY))
log(f"batches={NB}, processing {len(todo)} ({len(todo)/NB:.0%}) with {NTHREAD} threads")

CKPT = D + "centroid_ckpt.npz"
lock = threading.Lock()
done_batches = set()
G_sp = np.zeros((N, 3), np.float64); G_cp = np.zeros(N, np.int64)
G_sq = np.zeros((N, 3), np.float64); G_cq = np.zeros(N, np.int64)
if os.path.exists(CKPT):
    try:
        z = np.load(CKPT)
        G_sp, G_cp, G_sq, G_cq = z["sp"], z["cp"], z["sq"], z["cq"]
        done_batches = set(int(x) for x in z["done"])
        log(f"resume: {len(done_batches)} batches already done")
    except Exception as e:
        log("ckpt unreadable, starting fresh:", e)

def save_ckpt():
    with lock:
        tmp = D + "centroid_ckpt_tmp.npz"
        try:
            np.savez(tmp, sp=G_sp, cp=G_cp, sq=G_sq, cq=G_cq,
                     done=np.array(sorted(done_batches), dtype=np.int32))
            os.replace(tmp, CKPT)
        except Exception as e:
            log("ckpt save failed:", e)

def work(idxs, tid):
    f = HttpRangeFile(URL)
    rd = pa.ipc.open_file(pa.PythonFile(f, mode="r"))
    sp = np.zeros((N, 3), np.float64); cp = np.zeros(N, np.int64)
    sq = np.zeros((N, 3), np.float64); cq = np.zeros(N, np.int64)
    cols = ["pre_pt_root_id", "post_pt_root_id",
            "pre_pt_position_x", "pre_pt_position_y", "pre_pt_position_z",
            "post_pt_position_x", "post_pt_position_y", "post_pt_position_z"]
    done = 0
    for bi in idxs:
        if bi in done_batches: continue
        try:
            b = rd.get_batch(bi).select(cols)
        except Exception as e:
            log(f"  tid{tid} batch {bi} ERROR {e}"); continue
        pre = b.column(0).to_numpy(); post = b.column(1).to_numpy()
        xp = np.vstack([b.column(2).to_numpy(), b.column(3).to_numpy(), b.column(4).to_numpy()]).T
        xq = np.vstack([b.column(5).to_numpy(), b.column(6).to_numpy(), b.column(7).to_numpy()]).T
        pi_ = map_ids(pre); qi_ = map_ids(post)
        m = pi_ >= 0
        np.add.at(sp, pi_[m], xp[m]); np.add.at(cp, pi_[m], 1)
        m = qi_ >= 0
        np.add.at(sq, qi_[m], xq[m]); np.add.at(cq, qi_[m], 1)
        done += 1
        with lock:
            done_batches.add(bi)
        if done % 25 == 0:
            with lock:
                G_sp += sp; G_cp += cp; G_sq += sq; G_cq += cq
                sp[:] = 0; cp[:] = 0; sq[:] = 0; cq[:] = 0
            save_ckpt()
            log(f"  tid{tid}: {done}/{len(idxs)} batches (ckpt)")
    with lock:
        G_sp += sp; G_cp += cp; G_sq += sq; G_cq += cq
    save_ckpt()
    return None

t0 = time.time()
todo = [b for b in todo if b not in done_batches]
chunks = [todo[i::NTHREAD] for i in range(NTHREAD)]
with ThreadPoolExecutor(NTHREAD) as ex:
    list(ex.map(lambda a: work(a[1], a[0]), enumerate(chunks)))
sp, cp, sq, cq = G_sp, G_cp, G_sq, G_cq
log(f"fetch done in {time.time()-t0:.0f}s")

tot = sp + sq; cnt = cp + cq
xyz = np.full((N, 3), np.nan, np.float32)
ok = cnt > 0
xyz[ok] = (tot[ok] / cnt[ok, None] / 1000.0).astype(np.float32)   # nm -> um
xyz.tofile(D + "xyz_arbor.f32")
np.save(D + "arbor_syncnt.npy", cnt)

types = np.load(D + "type_names.npy", allow_pickle=True)
nt = np.fromfile(D + "node_type.i32", dtype=np.int32).astype(np.int64)
T = len(types)
toff = np.concatenate([[0], np.cumsum(np.bincount(nt, minlength=T))])
tn = {str(x): i for i, x in enumerate(types)}
diag = {"coverage": float(ok.mean()), "syn_sampled_total": int(cnt.sum()),
        "median_syn_per_cell": float(np.median(cnt[ok]))}
for t in ["EPG", "Delta7", "PFL3", "Tm3"]:
    if t in tn:
        ti = tn[t]; seg = slice(toff[ti], toff[ti+1])
        x = xyz[seg]; x = x[~np.isnan(x[:, 0])]
        diag[f"span_{t}"] = (x.max(0) - x.min(0)).round(1).tolist()
json.dump(diag, open(D + "centroids_diag.json", "w"), indent=1)
log("CENTROIDS DONE", diag)
