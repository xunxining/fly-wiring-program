#!/bin/bash
# 141_p9d_driver.sh — P9d 雄性旧坐标 shuffle 对照驱动 (§44): 预取/tmp + 暂存/tmp + flusher 异步冲刷
# 与 137 的区别: LATF 缺省(旧坐标+旧核), LATSHUFFLE=1, 输出 gm_p9d
# 注意: seg 日志必须用 /tmp/gm_seg.log (108_flusher 终局检测硬编码该名, §42 工程勘误)
cd /mnt/agents/output/果蝇生成器
export CELLQ=1 ACT=1 SPATIAL=1 NPOV=1 LAT=1 LATV=2 LATMODE=snap2 DYAD=2 SEGPAIRS=90000
export LATSHUFFLE=1
export MALE_D=/tmp/dm/ STAGE=/tmp/gm_stage9d
mkdir -p /tmp/gm_stage9d
bash 111_prefetch.sh > /tmp/gm9d_prefetch.log 2>&1
echo "$(date +%T) prefetch done" >> /tmp/gm9d_driver.log
for i in $(seq 1 200); do
  if [ ! -f gen_edges/gm_p9d.ckpt.pkl ] && [ -f gen_edges/gm_p9d_pre.i32 ]; then
    if grep -q "DONE:" /tmp/gm_seg.log 2>/dev/null; then break; fi
  fi
  pgrep -f "108_flusher.py gm_p9d" >/dev/null || nohup python3 -u 108_flusher.py gm_p9d /tmp/gm_stage9d > /tmp/gm9d_flush.log 2>&1 &
  python3 -u 105_male_generate.py gen_edges/gm_p9d > /tmp/gm_seg.log 2>&1
  RC=$?
  echo "$(date +%H:%M:%S) seg$i rc=$RC $(tail -1 /tmp/gm_seg.log 2>/dev/null)" >> /tmp/gm9d_driver.log
  # 终局三重确认 (flusher 重写 ckpt 有瞬时缺失窗口, 单次采样会误判, P9d seg1 踩坑)
  miss=0
  for c in 1 2 3; do
    [ ! -f gen_edges/gm_p9d.ckpt.pkl ] && miss=$((miss+1))
    sleep 2
  done
  if [ "$miss" -eq 3 ]; then
    echo "$(date +%H:%M:%S) ALL SEGMENTS DONE" >> /tmp/gm9d_driver.log
    break
  fi
  sleep 2
done
echo "$(date +%T) driver exit" >> /tmp/gm9d_driver.log
