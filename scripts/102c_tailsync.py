#!/usr/bin/env python3
"""102c_tailsync.py — 追加流尾同步: /tmp 生成产物 -> 门户, 每 45s 增量同步
用法: python3 102c_tailsync.py <src_prefix> <dst_prefix>
幂等: 按字节偏移续传; 挂载断开自动重试; 生成结束后自动收尾退出。
"""
import os, sys, time
src_p, dst_p = sys.argv[1], sys.argv[2]
SUFFIX = ["_pre.i32", "_post.i32", "_w.i32"]
off = {s: 0 for s in SUFFIX}
# 前缀校验续传: 若门户已有部分文件, 验证与源前缀一致后从其长度续传
for s in SUFFIX:
    dst = dst_p + s; src = src_p + s
    try:
        if os.path.exists(dst) and os.path.getsize(dst) > 0 and os.path.exists(src):
            ds = os.path.getsize(dst); ss = os.path.getsize(src)
            n = min(ds, ss)
            if n > 0:
                with open(dst,'rb') as f: f.seek(max(0,n-65536)); dt = f.read()
                with open(src,'rb') as f: f.seek(max(0,n-65536)); st = f.read()
                if dt == st and ds <= ss:
                    off[s] = ds
                    print(f"前缀校验通过 {s}: 从 {ds:,}B 续传", flush=True)
                else:
                    print(f"前缀不一致 {s}: 从零重写", flush=True)
    except OSError as e:
        print(f"前缀校验跳过 {s}: {str(e)[:40]}", flush=True)
CH = 4 * 1024 * 1024
idle = 0
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

while True:
    prog = False
    for s in SUFFIX:
        src = src_p + s; dst = dst_p + s
        if not os.path.exists(src):
            continue
        try:
            sz = os.path.getsize(src)
        except OSError:
            continue
        if sz <= off[s]:
            continue
        want = sz - off[s]
        try:
            with open(src, "rb") as f:
                f.seek(off[s]); data = f.read()
        except OSError:
            continue
        # 分块写门户
        pos = 0; ok = True
        while pos < len(data):
            try:
                mode = "r+b" if os.path.exists(dst) else "wb"
                with open(dst, mode) as g:
                    g.seek(off[s] + pos)
                    g.write(data[pos:pos+CH]); g.flush(); os.fsync(g.fileno())
                pos += CH
            except OSError as e:
                log(f"{s} 挂载重试: {str(e)[:40]}")
                time.sleep(15)
                # 校验门户当前长度, 防止错位的重复写
                try:
                    cur = os.path.getsize(dst) if os.path.exists(dst) else 0
                    pos = max(0, cur - off[s])
                    if pos < 0 or pos > len(data): pos = 0 if cur == 0 else pos
                except OSError:
                    pass
                ok = True  # 继续内层循环直到写完
                continue
        off[s] += len(data)
        prog = True
        log(f"{s} 同步到 {off[s]:,}B")
    if prog:
        idle = 0
    else:
        idle += 1
    # 若源进程结束(无 done 标记可判), 连续 20 轮无进展且 log 出现 DONE 则退出
    try:
        logtxt = open("/tmp/gm_full.log").read()
        if "DONE:" in logtxt and idle >= 3:
            log("生成 DONE 且尾同步完成, 退出"); break
    except OSError:
        pass
    if idle > 400:
        log("超时退出"); break
    time.sleep(45)
