#!/bin/bash
# 138_p9b_pipeline.sh — P9b 后处理流水线 (§41): 排序 → MUTQ → P8e → 验证/T分解 → 汇总
# 全程收割机韧性(幂等续跑); 日志一律 /tmp (portal EIO 会卡死), 关键结果即写 portal 数据目录
cd /mnt/agents/output/果蝇生成器 || exit 1
G=gen_edges
L=/tmp/p9b_pipe.log
echo "[pipe $(date +%T)] start" >> $L

# ---- 1. 等生成完成: ckpt 消失且三文件尺寸 == 25,135,616*4 ----
while :; do
  if [ ! -f $G/gm_p9b.ckpt.pkl ]; then
    s1=$(stat -c %s $G/gm_p9b_pre.i32 2>/dev/null || echo 0)
    s2=$(stat -c %s $G/gm_p9b_post.i32 2>/dev/null || echo 0)
    s3=$(stat -c %s $G/gm_p9b_w.i32 2>/dev/null || echo 0)
    if [ "$s1" = 100542464 ] && [ "$s2" = 100542464 ] && [ "$s3" = 100542464 ]; then
      echo "[pipe $(date +%T)] generation complete" >> $L; break
    fi
  fi
  sleep 60
done

# ---- 2. 排序 ----
until python3 -u 120b_sort_p9b.py >> $L 2>&1 && [ -f $G/gm_p9bs_w.i32 ]; do
  echo "[pipe $(date +%T)] sort retry" >> $L; sleep 10
done

# ---- 3. 实测新基座 unique/self_loops, 生成 117b ----
python3 - <<'EOF' >> $L 2>&1
import numpy as np, json
R = "/mnt/agents/output/果蝇生成器/"
pre = np.fromfile(R+"gen_edges/gm_p9bs_pre.i32", dtype=np.int32)
post = np.fromfile(R+"gen_edges/gm_p9bs_post.i32", dtype=np.int32)
N = 164506
kk = pre.astype(np.int64)*N + post
u = int(len(np.unique(kk)))
sl = int((pre == post).sum())
json.dump({"unique": u, "self_loops": sl}, open(R+"data_male/p9b_base_sanity.json", "w"))
src = open(R+"117_mutq_repair.py").read()
old = 'BASE_U, BASE_SL = (18638016, 63765) if SEX == "female" else (23426144, 84444)'
new = f'BASE_U, BASE_SL = (18638016, 63765) if SEX == "female" else ({u}, {sl})'
assert old in src
open(R+"117b_p9b.py", "w").write(src.replace(old, new))
print(f"[pipe] 117b generated: BASE_U={u:,} BASE_SL={sl:,}")
EOF

# ---- 4. MUTQ 修复 (gm_p9bs → g_p9bm) ----
while [ ! -f data_male/p9b_mutq.json ]; do
  python3 -u 117b_p9b.py male gm_p9bs g_p9bm p9b_mutq.json >> $L 2>&1
  echo "[pipe $(date +%T)] mutq exit rc=$?" >> $L; sleep 5
done

# ---- 5. P8e 修剪 (g_p9bm → g_p9bm2) ----
while [ ! -f data_male/p9b_trim.json ]; do
  python3 -u 123_trim.py male g_p9bm g_p9bm2 p9b_trim.json >> $L 2>&1
  echo "[pipe $(date +%T)] trim exit rc=$?" >> $L; sleep 5
done

# ---- 6. 验证: 106 全量验证(基座+终态) + 113 T分解(终态) ----
until python3 -u 106_male_validate.py gen_edges/gm_p9b data_male/val_gm_p9b.json >> $L 2>&1 && [ -f data_male/val_gm_p9b.json ]; do
  echo "[pipe $(date +%T)] val base retry" >> $L; sleep 10
done
until python3 -u 106_male_validate.py gen_edges/g_p9bm2 data_male/val_g_p9bm2.json >> $L 2>&1 && [ -f data_male/val_g_p9bm2.json ]; do
  echo "[pipe $(date +%T)] val final retry" >> $L; sleep 10
done
until python3 -u 113_male_tgap.py g_p9bm2 data_male/tgap_p9b.json >> $L 2>&1 && [ -f data_male/tgap_p9b.json ]; do
  echo "[pipe $(date +%T)] tgap retry" >> $L; sleep 10
done
echo "[pipe $(date +%T)] ALL DONE" >> $L
