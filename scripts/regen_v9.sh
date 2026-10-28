#!/bin/bash
W=/mnt/agents/output/果蝇生成器
L=$W/regen_v9.log
echo "regen $(date)" > $L
mkdir -p /home/kimi/scratch
SPATIAL=1 CXSYS=1 python3 $W/38_generate_v4.py 1 /home/kimi/scratch/genv9_S1 >> $L 2>&1 || { echo "GEN FAIL" >> $L; exit 1; }
ok=1
for f in pre post w; do
  cp /home/kimi/scratch/genv9_S1_$f.i32 $W/gen/genv9_S1_$f.i32 2>/dev/null
  a=$(stat -c%s /home/kimi/scratch/genv9_S1_$f.i32 2>/dev/null); b=$(stat -c%s $W/gen/genv9_S1_$f.i32 2>/dev/null)
  [ -n "$a" ] && [ "$a" = "$b" ] || ok=0
done
echo "copy ok=$ok $(date)" >> $L
if [ $ok = 1 ]; then
  python3 $W/67_dist_profile.py $W/gen/genv9_S1 $W/dist_v9.json >> $L 2>&1 || echo "DIST FAIL" >> $L
fi
echo "REGEN DONE $(date)" >> $L
