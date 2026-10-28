#!/bin/bash
# 126_lataware_driver.sh — §38 P8f: 保 T 修剪(125) + 补填(117), 仅雌性单轮
ROOT="/mnt/agents/output/果蝇生成器"
LOG="$ROOT/lataware_driver.log"
D="$ROOT/data_female"
echo "=== taware driver start $(date) ===" >> "$LOG"
rm -f "$D/trim_result.json" "$D/trim_fill4.json"
until [ -f "$D/trim_result.json" ]; do
  echo "[taware] trim launch $(date)" >> "$LOG"
  python3 "$ROOT/127_trim_lataware.py" female >> "$LOG" 2>&1
  sleep 5
done
mv "$D/trim_result.json" "$D/trim_result_lataware.json"
echo "[taware] trim done, fill launch $(date)" >> "$LOG"
until [ -f "$D/trim_fill4.json" ]; do
  python3 "$ROOT/117_mutq_repair.py" female g_mutq4 g_mutq4 trim_fill4.json >> "$LOG" 2>&1
  sleep 5
done
echo "=== taware driver done $(date) ===" >> "$LOG"
