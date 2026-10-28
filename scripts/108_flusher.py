#!/usr/bin/env python3
# 108_flusher.py — /tmp 暂存 -> portal 异步同步器 (雄性生成配套)
# 顺序保证: 三个边文件同步到 >= ckpt.wstate 偏移后, 才复制 ckpt 到 portal
# 终局: 暂存 ckpt 消失(=DONE)且边文件完全同步后, 删除 portal ckpt 并退出
import os, sys, time, pickle, gzip

STAGE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/gm_stage"
BASE = sys.argv[1] if len(sys.argv) > 1 else "gm_p8c"
PORTAL = "/mnt/agents/output/果蝇生成器/gen_edges"
KINDS = ("pre", "post", "w")
CH = 4 * 1024 * 1024

def log(m):
    print(time.strftime("[%H:%M:%S]"), m, flush=True)

def ppath(k): return os.path.join(PORTAL, f"{BASE}_{k}.i32")
def spath(k): return os.path.join(STAGE, f"{BASE}_{k}.i32")
PCK = os.path.join(PORTAL, BASE + ".ckpt.pkl")
SCK = os.path.join(STAGE, BASE + ".ckpt.pkl")

def sync_file(k, upto=None):
    """把暂存文件追加同步到 portal (可选只到 upto 字节)。返回 portal 当前大小。"""
    sp, pp = spath(k), ppath(k)
    if not os.path.exists(sp):
        return os.path.getsize(pp) if os.path.exists(pp) else 0
    ssz = os.path.getsize(sp)
    tgt = ssz if upto is None else min(ssz, upto)
    for att in range(400):
        try:
            psz = os.path.getsize(pp) if os.path.exists(pp) else 0
            if psz >= tgt:
                return psz   # 只追加, 永不截断 (portal 超前部分与确定性流字节一致)
            with open(sp, "rb") as fr, open(pp, "r+b" if psz else "wb") as fw:
                fr.seek(psz); fw.seek(psz)
                cur = psz
                while cur < tgt:
                    b = fr.read(min(CH, tgt - cur))
                    if not b: break
                    fw.write(b); cur += len(b)
                fw.flush(); os.fsync(fw.fileno())
            return os.path.getsize(pp)
        except OSError as e:
            if att % 20 == 0: log(f"同步重试{att} {k}: {str(e)[:40]}")
            time.sleep(5)
    return -1

def read_wstate(path):
    raw = open(path, "rb").read()
    ck = pickle.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
    return ck.get("wstate", {})

log("flusher 启动")
idle = 0
while True:
    try:
        # 1) 边文件尽量同步(若暂存 ckpt 在, 先按其 wstate 同步到偏移, 再复制 ckpt)
        if os.path.exists(SCK):
            ws = None
            for att in range(12):
                try:
                    ws = read_wstate(SCK); break
                except Exception:
                    time.sleep(2)
            if ws:
                ok = all(sync_file(k) >= ws.get(k, 0) for k in KINDS)
                if ok:
                    # 2) ckpt 复制到 portal (tmp+replace)
                    for att in range(400):
                        try:
                            data = open(SCK, "rb").read()
                            with open(PCK + ".tmp", "wb") as f:
                                f.write(data); f.flush(); os.fsync(f.fileno())
                            os.replace(PCK + ".tmp", PCK)
                            break
                        except OSError as e:
                            if att % 20 == 0: log(f"ckpt 同步重试{att}: {str(e)[:40]}")
                            time.sleep(5)
        # 3) 边文件追平暂存(超过 ckpt 偏移的部分也同步, 减少终局等待)
        for k in KINDS: sync_file(k)
        # 4) 终局检测
        if not os.path.exists(SCK) and os.path.exists(PCK):
            done = False
            try:
                done = os.path.exists("/tmp/gm_seg.log") and "DONE:" in open("/tmp/gm_seg.log").read()
            except OSError: pass
            if done:
                sz_ok = all((os.path.getsize(ppath(k)) if os.path.exists(ppath(k)) else 0) ==
                            (os.path.getsize(spath(k)) if os.path.exists(spath(k)) else 0) for k in KINDS)
                if sz_ok:
                    for att in range(200):
                        try:
                            fb_s = os.path.join(STAGE, BASE + ".fallback.txt")
                            if os.path.exists(fb_s):
                                open(os.path.join(PORTAL, BASE + ".fallback.txt"), "wb").write(open(fb_s, "rb").read())
                            os.remove(PCK)
                            log("终局同步完成, portal ckpt 已删, flusher 退出")
                            sys.exit(0)
                        except OSError as e:
                            log(f"终局重试{att}: {str(e)[:40]}"); time.sleep(5)
        idle = 0
    except Exception as e:
        log(f"flusher 异常: {str(e)[:60]}")
    time.sleep(6)
