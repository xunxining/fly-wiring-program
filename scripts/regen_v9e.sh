#!/bin/bash
W=/mnt/agents/output/果蝇生成器
L=$W/regen_v9d.log
S=/var/tmp/genv9d
echo "regen-e v9d KMAP+spatial $(date)" > $L
mkdir -p $S
SPATIAL=1 SPATIALOS=48 CXSYS=1 python3 $W/38_generate_v4.py 1 $S/genv9d_S1 >> $L 2>&1 || { echo "GEN FAIL" >> $L; exit 1; }
python3 $W/67_dist_gen.py $S/genv9d_S1 $W/dist_v9d.json >> $L 2>&1 || echo "DIST FAIL" >> $L
python3 $W/51_transitivity.py $S/genv9d_S1 >> $L 2>&1 || echo "T FAIL" >> $L
python3 $W/58_validate_v5.py $S/genv9d_S1 $W/v9d_validation.json >> $L 2>&1 || echo "58 FAIL" >> $L
echo "REGEN-D DONE $(date)" >> $L
