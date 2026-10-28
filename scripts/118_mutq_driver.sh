#!/bin/bash
# 118_mutq_driver.sh — §35 MUTQ 修复驱动: 收割机重启韧性(幂等续跑)
# 日志直接写门户(收割机清 /tmp 不丢进度)
cd "/mnt/agents/output/果蝇生成器" || exit 1
LOG=mutq_driver.log
DONE=data_female/mutq_repair.json
DONEM=data_male/mutq_repair.json
while [ ! -f "$DONE" ]; do
  echo "[driver $(date +%T)] female run start" >> $LOG
  python3 -u 117_mutq_repair.py female >> $LOG 2>&1
  echo "[driver $(date +%T)] female run exit" >> $LOG
  sleep 3
done
echo "[driver $(date +%T)] female DONE -> male" >> $LOG
while [ ! -f "$DONEM" ]; do
  echo "[driver $(date +%T)] male run start" >> $LOG
  python3 -u 117_mutq_repair.py male >> $LOG 2>&1
  echo "[driver $(date +%T)] male run exit" >> $LOG
  sleep 3
done
echo "[driver $(date +%T)] ALL DONE" >> $LOG
touch mutq_all_done
