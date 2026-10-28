#!/bin/bash
# 112_final_driver.sh — 雄侧最终段: 预取/tmp + 暂存/tmp + flusher 异步冲刷
cd /mnt/agents/output/果蝇生成器
export CELLQ=1 ACT=1 SPATIAL=1 NPOV=1 LAT=1 LATV=2 LATMODE=snap2 DYAD=2 SEGPAIRS=90000
export MALE_D=/tmp/dm/ STAGE=/tmp/gm_stage
mkdir -p /tmp/gm_stage
bash 111_prefetch.sh > /tmp/gm_prefetch.log 2>&1
for i in $(seq 1 30); do
  if [ ! -f gen_edges/gm_p8c.ckpt.pkl ] && [ -f gen_edges/gm_p8c_pre.i32 ]; then
    if grep -q "DONE:" /tmp/gm_seg.log 2>/dev/null; then break; fi
  fi
  pgrep -f "108_flusher.py gm_p8c" >/dev/null || nohup python3 -u 108_flusher.py gm_p8c /tmp/gm_stage > /tmp/gm_flush.log 2>&1 &
  python3 -u 105_male_generate.py gen_edges/gm_p8c > /tmp/gm_seg.log 2>&1
  RC=$?
  echo "$(date +%H:%M:%S) seg$i rc=$RC $(tail -1 /tmp/gm_seg.log 2>/dev/null)" >> /tmp/gm_driver.log
  if [ ! -f gen_edges/gm_p8c.ckpt.pkl ]; then
    echo "$(date +%H:%M:%S) ALL SEGMENTS DONE" >> /tmp/gm_driver.log
    break
  fi
  sleep 2
done
