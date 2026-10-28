#!/bin/bash
W=/mnt/agents/output/果蝇生成器
L=$W/regen_v9b.log
S=/var/tmp/genv9b
echo "regen-c v9b imputed $(date)" > $L
mkdir -p $S
SPATIAL=1 SPATIALXYZ=xyz_gen_imp.f32 CXSYS=1 python3 $W/38_generate_v4.py 1 $S/genv9b_S1 >> $L 2>&1 || { echo "GEN FAIL" >> $L; exit 1; }
python3 $W/67_dist_gen.py $S/genv9b_S1 $W/dist_v9b.json >> $L 2>&1 || echo "DIST FAIL" >> $L
python3 $W/51_transitivity.py $S/genv9b_S1 >> $L 2>&1 || echo "T FAIL" >> $L
echo "REGEN-B DONE $(date)" >> $L
