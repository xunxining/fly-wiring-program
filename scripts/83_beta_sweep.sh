#!/bin/bash
# NPOV beta sweep with fixed per-row rejection bound. Sequential (4GB memcg).
cd /mnt/agents/output/果蝇生成器
L=/mnt/agents/output/果蝇生成器/logs
mkdir -p "$L"
for B in 1.0 1.5 2.0; do
  TAG="npovB${B}"
  echo "=== $TAG start $(date) ===" >> "$L/83_sweep.log"
  CELLQ=1 ACT=1 SPATIAL=1 SPATIALOS=48 NPOV=1 NPB=$B \
    python3 75_female_generate.py /tmp/g_full_${TAG} >> "$L/83_${TAG}.log" 2>&1
  python3 82_quickT.py /tmp/g_full_${TAG} /tmp/q_full_${TAG}.json >> "$L/83_${TAG}.log" 2>&1
  cp /tmp/q_full_${TAG}.json "$L/q_full_${TAG}.json"
  echo "=== $TAG done $(date) ===" >> "$L/83_sweep.log"
done
echo "ALL DONE $(date)" >> "$L/83_sweep.log"
