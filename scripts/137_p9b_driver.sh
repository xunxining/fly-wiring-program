#!/bin/bash
# 137_p9b_driver.sh — P9b 雄性 m2 全栈重生成驱动 (§41): 预取/tmp + 暂存/tmp + flusher 异步冲刷
# 与 112_final_driver.sh 的唯一区别: LATF=m2, 输出 gm_p9b, 预取 m2 点阵文件
cd /mnt/agents/output/果蝇生成器
export CELLQ=1 ACT=1 SPATIAL=1 NPOV=1 LAT=1 LATV=2 LATMODE=snap2 DYAD=2 SEGPAIRS=90000
export LATF=m2
export MALE_D=/tmp/dm/ STAGE=/tmp/gm_stage9b
mkdir -p /tmp/gm_stage9b
bash 111_prefetch.sh > /tmp/gm9b_prefetch.log 2>&1
# 补取 m2 点阵文件 (111 清单外)
for f in lat2q_m2.npz lat_snap_m2.npz; do
  for a in $(seq 1 100); do
    s_src=$(stat -c %s "data_male/$f" 2>/dev/null) || { sleep 3; continue; }
    s_dst=$(stat -c %s "/tmp/dm/$f" 2>/dev/null || echo 0)
    [ "$s_dst" = "$s_src" ] && break
    cp "data_male/$f" "/tmp/dm/$f.part" 2>/dev/null && mv "/tmp/dm/$f.part" "/tmp/dm/$f" && break
    sleep 3
  done
done
echo "$(date +%T) prefetch+m2 done" >> /tmp/gm9b_driver.log
for i in $(seq 1 200); do
  if [ ! -f gen_edges/gm_p9b.ckpt.pkl ] && [ -f gen_edges/gm_p9b_pre.i32 ]; then
    if grep -q "DONE:" /tmp/gm9b_seg.log 2>/dev/null; then break; fi
  fi
  pgrep -f "108_flusher.py gm_p9b" >/dev/null || nohup python3 -u 108_flusher.py gm_p9b /tmp/gm_stage9b > /tmp/gm9b_flush.log 2>&1 &
  python3 -u 105_male_generate.py gen_edges/gm_p9b > /tmp/gm9b_seg.log 2>&1
  RC=$?
  echo "$(date +%H:%M:%S) seg$i rc=$RC $(tail -1 /tmp/gm9b_seg.log 2>/dev/null)" >> /tmp/gm9b_driver.log
  if [ ! -f gen_edges/gm_p9b.ckpt.pkl ]; then
    echo "$(date +%H:%M:%S) ALL SEGMENTS DONE" >> /tmp/gm9b_driver.log
    break
  fi
  sleep 2
done
echo "$(date +%T) driver exit" >> /tmp/gm9b_driver.log
