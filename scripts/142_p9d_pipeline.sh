#!/bin/bash
# 142_p9d_pipeline.sh — P9d 后处理 (§44): 等生成 → 106 校验 → 113 T分解 (无需排序/MUTQ/P8e)
cd /mnt/agents/output/果蝇生成器 || exit 1
G=gen_edges
L=/tmp/p9d_pipe.log
echo "[pipe $(date +%T)] start" >> $L

# ---- 1. 等生成完成: ckpt 消失且三文件尺寸 == 25,135,616*4 ----
while :; do
  if [ ! -f $G/gm_p9d.ckpt.pkl ]; then
    s1=$(stat -c %s $G/gm_p9d_pre.i32 2>/dev/null || echo 0)
    s2=$(stat -c %s $G/gm_p9d_post.i32 2>/dev/null || echo 0)
    s3=$(stat -c %s $G/gm_p9d_w.i32 2>/dev/null || echo 0)
    if [ "$s1" = 100542464 ] && [ "$s2" = 100542464 ] && [ "$s3" = 100542464 ]; then
      echo "[pipe $(date +%T)] generation complete" >> $L; break
    fi
  fi
  sleep 60
done

# ---- 2. 106 全量校验 ----
until python3 -u 106_male_validate.py gen_edges/gm_p9d data_male/val_gm_p9d.json >> $L 2>&1 && [ -f data_male/val_gm_p9d.json ]; do
  echo "[pipe $(date +%T)] val retry" >> $L; sleep 10
done

# ---- 3. 113 T 分解 ----
until python3 -u 113_male_tgap.py gm_p9d data_male/tgap_p9d.json >> $L 2>&1 && [ -f data_male/tgap_p9d.json ]; do
  echo "[pipe $(date +%T)] tgap retry" >> $L; sleep 10
done
echo "[pipe $(date +%T)] ALL DONE" >> $L
