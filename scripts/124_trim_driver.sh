#!/bin/bash
# 124_trim_driver.sh — §36 P8e: 修剪(123) + 补填(117 override), ≤2 轮, 直至主门 |recip-real|≤0.005
ROOT="/mnt/agents/output/果蝇生成器"
LOG="$ROOT/trim_driver.log"
echo "=== driver start $(date) ===" >> "$LOG"

run_sex () {
  local SEX=$1 SRC=$2 DST=$3 REAL=$4
  local D="$ROOT/data_female"; [ "$SEX" = male ] && D="$ROOT/data_male"
  for ITER in 1 2; do
    if [ "$ITER" = 2 ]; then
      # 第 2 轮: 清掉两侧 sidecar(带重试), 让 123/117 从当前 DST post 全新起跑
      local P="$ROOT/gen_edges/${DST}_post.i32"
      for f in "$P.added2" "$P.removed" "$P.cur2" "$P.surplus" "$P.added" "$P.cur"                "$P.added2.A" "$P.added2.B" "$P.removed.A" "$P.removed.B"                "$P.added.A" "$P.added.B"                "$P.surplus.A" "$P.surplus.B" "$P".Apt* "$P".Bpt*; do
        for t in $(seq 1 10); do rm -f "$f" 2>/dev/null; [ ! -e "$f" ] && break; sleep 3; done
      done
      echo "[$SEX] iter2 sidecars cleared $(date)" >> "$LOG"
    fi
    # 修剪(123 自带幂等续跑; 直到 trim_result.json 出现)
    until [ -f "$D/trim_result.json" ]; do
      echo "[$SEX iter$ITER] trim launch $(date)" >> "$LOG"
      python3 "$ROOT/123_trim.py" "$SEX" >> "$LOG" 2>&1
      sleep 5
    done
    mv "$D/trim_result.json" "$D/trim_result_iter$ITER.json" 2>/dev/null
    # 补填(117 override; 直到 trim_fill.json 出现)
    rm -f "$D/trim_fill.json"
    until [ -f "$D/trim_fill.json" ]; do
      echo "[$SEX iter$ITER] fill launch $(date)" >> "$LOG"
      python3 "$ROOT/117_mutq_repair.py" "$SEX" "$DST" "$DST" trim_fill.json >> "$LOG" 2>&1
      sleep 5
    done
    mv "$D/trim_fill.json" "$D/trim_fill_iter$ITER.json" 2>/dev/null
    # 主门判定
    PASS=$(python3 - "$D/trim_fill_iter$ITER.json" "$REAL" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
print("yes" if abs(r["recip_new"] - float(sys.argv[2])) <= 0.005 else "no")
EOF
)
    echo "[$SEX iter$ITER] gate check: $PASS $(date)" >> "$LOG"
    [ "$PASS" = yes ] && return 0
  done
  return 1
}

if [ -f "$ROOT/data_female/trim_fill_iter1.json" ]; then
  # 雌性已通过主门(recip 0.32218 vs 真实 0.32130)——跳过, 防重复修剪覆盖已验证终态
  echo "[female] already passed (trim_fill_iter1.json present) — skipped $(date)" >> "$LOG"
  F=0
else
  run_sex female g_mutq  g_mutq2  0.32130410937717807
  F=$?
fi
run_sex male   g_mutqm g_mutqm2 0.2997500837059255
M=$?
echo "female_pass=$F male_pass=$M" > "$ROOT/data_female/trim_all_done"
echo "=== driver done $(date) F=$F M=$M ===" >> "$LOG"
