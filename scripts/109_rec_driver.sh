#!/bin/bash
# 109_rec_driver.sh — gm_p8c_post 恢复: 重放前 3,595,000 对 (确定性字节流), 输出 gm_rec_*
cd /mnt/agents/output/果蝇生成器
export CELLQ=1 ACT=1 SPATIAL=1 NPOV=1 LAT=1 LATV=2 LATMODE=snap2 DYAD=2
export SEGPAIRS=60000 STARTP=0 MAXP=3595000
for i in $(seq 1 40); do
  if [ ! -f gen_edges/gm_rec.ckpt.pkl ] && grep -q "DONE:" /tmp/gm_rec_seg.log 2>/dev/null; then
    echo "$(date +%H:%M:%S) RECOVERY DONE" >> /tmp/gm_rec_driver.log; break
  fi
  python3 -u 105_male_generate.py gen_edges/gm_rec > /tmp/gm_rec_seg.log 2>&1
  echo "$(date +%H:%M:%S) rec$i rc=$? $(tail -1 /tmp/gm_rec_seg.log 2>/dev/null)" >> /tmp/gm_rec_driver.log
  sleep 2
done
