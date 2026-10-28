#!/bin/bash
W=/mnt/agents/output/果蝇生成器
L=$W/chain_v10.log
S=/var/tmp/genv10c
echo "=== v10c +目标活跃度 $(date) ===" > $L
mkdir -p $S
CELLQ=1 SPATIAL=1 SPATIALOS=48 ACT=1 CXSYS=1 python3 $W/38_generate_v4.py 1 $S/genv10c_S1 >> $L 2>&1 || { echo "GEN FAIL" >> $L; exit 1; }
python3 $W/69_cellcv.py $S/genv10c_S1 $W/cellcv_v10c.json >> $L 2>&1 || echo "CV FAIL" >> $L
python3 $W/67_dist_gen.py $S/genv10c_S1 $W/dist_v10c.json >> $L 2>&1 || echo "DIST FAIL" >> $L
python3 $W/51_transitivity.py $S/genv10c_S1 >> $L 2>&1 || echo "T FAIL" >> $L
python3 $W/58_validate_v5.py $S/genv10c_S1 $W/v10c_validation.json >> $L 2>&1 || echo "58 FAIL" >> $L
echo "=== CHAIN V10 DONE $(date) ===" >> $L
