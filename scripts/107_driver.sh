#!/bin/bash
# 107_driver.sh — 雄侧全栈分段生成驱动 (预取 /tmp + 直写 portal)
cd /mnt/agents/output/果蝇生成器
export CELLQ=1 ACT=1 SPATIAL=1 NPOV=1 LAT=1 LATV=2 LATMODE=snap2 DYAD=2 SEGPAIRS=5000
export MALE_D=/tmp/dm/
bash 111_prefetch.sh > /tmp/gm_prefetch.log 2>&1
for i in $(seq 1 20); do
  if [ ! -f gen_edges/gm_p8c.ckpt.pkl ] && [ -f gen_edges/gm_p8c_pre.i32 ]; then
    if grep -q "DONE:" /tmp/gm_seg.log 2>/dev/null; then break; fi
  fi
  python3 -u 105_male_generate.py gen_edges/gm_p8c > /tmp/gm_seg.log 2>&1
  RC=$?
  echo "$(date +%H:%M:%S) seg$i rc=$RC $(tail -1 /tmp/gm_seg.log 2>/dev/null)" >> /tmp/gm_driver.log
  if [ ! -f gen_edges/gm_p8c.ckpt.pkl ]; then
    echo "$(date +%H:%M:%S) ALL SEGMENTS DONE" >> /tmp/gm_driver.log
    break
  fi
  sleep 2
done
