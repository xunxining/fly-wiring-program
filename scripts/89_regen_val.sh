#!/bin/bash
# 89_regen_val.sh — 再生成 全栈+门 并立刻全量校验+闭合曲线 (抗 /tmp 抹除: 链式执行, 结果落 portal)
cd /mnt/agents/output/果蝇生成器
L=logs
echo "=== regen start $(date) ===" >> $L/89.log
CELLQ=1 ACT=1 SPATIAL=1 SPATIALOS=48 NPOV=1 NPB=0 \
  python3 75_female_generate.py /tmp/g_npov0 >> $L/89_gen.log 2>&1
cp /tmp/g_npov0.fallback.txt $L/g_npov0.fallback.txt 2>/dev/null
python3 76_female_validate.py /tmp/g_npov0 data_female/valf_full_npov0.json >> $L/89.log 2>&1
python3 87_closslope.py /tmp/g_npov0 data_female/cls_npov0.json >> $L/89.log 2>&1
python3 82_quickT.py /tmp/g_npov0 $L/q_npov0_regen.json >> $L/89.log 2>&1
# 边表备份到 portal (76/87 已用完, 但留档防再丢)
mkdir -p gen_edges
cp /tmp/g_npov0_pre.i32 gen_edges/g_npov0_pre.i32
cp /tmp/g_npov0_post.i32 gen_edges/g_npov0_post.i32
cp /tmp/g_npov0_w.i32 gen_edges/g_npov0_w.i32
echo "=== regen+val done $(date) ===" >> $L/89.log
