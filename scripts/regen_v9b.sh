#!/bin/bash
W=/mnt/agents/output/果蝇生成器
L=$W/regen_v9.log
S=/var/tmp/genv9
echo "regen-b $(date)" >> $L
mkdir -p $S
SPATIAL=1 CXSYS=1 python3 $W/38_generate_v4.py 1 $S/genv9_S1 >> $L 2>&1 || { echo "GEN FAIL" >> $L; exit 1; }
python3 $W/67_dist_gen.py $S/genv9_S1 $W/dist_v9.json >> $L 2>&1 || echo "DIST FAIL" >> $L
ok=1
for f in pre post w; do
  for try in 1 2 3 4 5 6 7 8; do
    timeout 120 cp $S/genv9_S1_$f.i32 $W/gen/genv9_S1_$f.i32 2>/dev/null
    a=$(stat -c%s $S/genv9_S1_$f.i32 2>/dev/null); b=$(stat -c%s $W/gen/genv9_S1_$f.i32 2>/dev/null)
    [ -n "$a" ] && [ "$a" = "$b" ] && break
    sleep 8
    [ $try = 8 ] && ok=0
  done
done
echo "copy ok=$ok $(date)" >> $L
echo "REGEN DONE $(date)" >> $L
