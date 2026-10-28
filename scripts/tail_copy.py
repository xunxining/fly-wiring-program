#!/usr/bin/env python3
"""tail_copy.py v2 — 跟随式分块拷贝 (抗挂载抖动版)。
- 快拷: 8MB 块, 不 sync, 0.2s 间隔, 紧跟生成进度;
- 抖动: portal 不可写时等待恢复后继续 (源在本地, 抖动不丢数据);
- 启动时以远端已有大小对齐 (断点续拷);
- 源消失则大声退出。
用法: python3 tail_copy.py <local_prefix> <remote_dir> <done_flag_file>"""
import os, sys, time, subprocess

loc, rdir, flag = sys.argv[1], sys.argv[2], sys.argv[3]
CH = 8 * 1024 * 1024
names = ["pre", "post", "w"]

def size(p):
    try: return os.path.getsize(p)
    except OSError: return -1

def portal_ok():
    try: os.listdir(rdir); return True
    except OSError: return False

copied = {}
for n in names:
    rp = os.path.join(rdir, f"{os.path.basename(loc)}_{n}.i32")
    copied[n] = max(size(rp), 0)

while True:
    try:
        done = os.path.exists(flag)
        src_gone = all(size(f"{loc}_{n}.i32") < 0 for n in names)
        if src_gone and not done:
            print("TAILCOPY_SRC_GONE", flush=True)
            sys.exit(2)
        for n in names:
            lp = f"{loc}_{n}.i32"
            rp = os.path.join(rdir, f"{os.path.basename(loc)}_{n}.i32")
            sz = size(lp)
            if sz < 0: continue
            limit = sz if done else sz - (sz % CH)
            while copied[n] < limit:
                if not portal_ok():
                    time.sleep(3); continue
                blk = min(CH, limit - copied[n])
                args = (["dd", f"if={lp}", f"of={rp}", f"bs={CH}",
                         f"skip={copied[n]//CH}", f"seek={copied[n]//CH}",
                         "count=1", "conv=notrunc", "status=none"]
                        if blk == CH and copied[n] % CH == 0 else
                        ["dd", f"if={lp}", f"of={rp}", "bs=4",
                         f"skip={copied[n]//4}", f"seek={copied[n]//4}",
                         f"count={blk//4}", "conv=notrunc", "status=none"])
                r = subprocess.run(args)
                if r.returncode != 0:
                    time.sleep(3); break
                copied[n] += blk
                time.sleep(0.2)
        if done:
            ok = True
            for n in names:
                ls_ = size(f"{loc}_{n}.i32")
                rs_ = size(os.path.join(rdir, f"{os.path.basename(loc)}_{n}.i32"))
                if ls_ < 0 or rs_ != ls_:
                    ok = False
            print("TAILCOPY_DONE", ok, flush=True)
            break
    except Exception as ex:
        print("TAILCOPY_ERR", repr(ex), flush=True)
        time.sleep(5)
    time.sleep(5)
