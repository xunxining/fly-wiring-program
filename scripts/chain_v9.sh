#!/bin/bash
W=/mnt/agents/output/果蝇生成器
L=$W/chain_v9.log
echo "=== stage0 gen SPATIAL=1 CXSYS=1 $(date) ===" >> $L
mkdir -p /tmp/groot
SPATIAL=1 CXSYS=1 python3 $W/38_generate_v4.py 1 /tmp/groot/genv9_S1 >> $L 2>&1 || { echo "GEN FAIL" >> $L; exit 1; }
echo "=== stage1 dist profile $(date) ===" >> $L
python3 $W/67_dist_profile.py /tmp/groot/genv9_S1 $W/dist_v9.json >> $L 2>&1 || echo "DIST FAIL" >> $L
echo "=== stage2 transitivity $(date) ===" >> $L
python3 $W/51_transitivity.py /tmp/groot/genv9_S1 >> $L 2>&1 || echo "T FAIL" >> $L
echo "=== stage3 suite 58 $(date) ===" >> $L
python3 $W/58_validate_v5.py /tmp/groot/genv9_S1 $W/v9_validation.json >> $L 2>&1 || echo "58 FAIL" >> $L
echo "=== stage4 copy $(date) ===" >> $L
ok=1
for f in pre post w; do
  for try in 1 2 3 4 5; do
    cp /tmp/groot/genv9_S1_$f.i32 $W/gen/genv9_S1_$f.i32 2>/dev/null
    a=$(stat -c%s /tmp/groot/genv9_S1_$f.i32 2>/dev/null); b=$(stat -c%s $W/gen/genv9_S1_$f.i32 2>/dev/null)
    [ -n "$a" ] && [ "$a" = "$b" ] && break; sleep 4; [ $try = 5 ] && ok=0
  done
done
echo "copy ok=$ok" >> $L
echo "=== CHAIN V9 DONE $(date) ===" >> $L
