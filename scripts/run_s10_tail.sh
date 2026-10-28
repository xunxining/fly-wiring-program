#!/bin/bash
# S=10 尾段续跑 v2: [3600000, 3830931) 拆 3 段, cat 并入(对齐安全的缓冲写), 成功才删临时文件
set -u
OUT=/tmp/flydata/gen
cd /mnt/agents/output/果蝇生成器

for s in 3600000 3680000 3760000; do
  e=$((s+80000)); [ $e -gt 3830931 ] && e=3830931
  ok=0
  for attempt in 1 2 3; do
    echo "=== tail segment [$s,$e) attempt $attempt ==="
    sync
    if python3 09_generate_s10.py $s $e 5000; then
      cat $OUT/seg${s}_pre.i32 >> $OUT/gen_S10_pre.i32 && \
      cat $OUT/seg${s}_post.i32 >> $OUT/gen_S10_post.i32 && \
      cat $OUT/seg${s}_w.i32 >> $OUT/gen_S10_w.i32 || { echo "!!! merge failed, keeping seg files"; exit 1; }
      rm -f $OUT/seg${s}_pre.i32 $OUT/seg${s}_post.i32 $OUT/seg${s}_w.i32
      sync
      ok=1
      break
    fi
    echo "!!! tail segment failed, cleaning and retrying"
    rm -f $OUT/seg${s}_pre.i32 $OUT/seg${s}_post.i32 $OUT/seg${s}_w.i32
  done
  [ $ok -eq 0 ] && { echo "!!! tail segment [$s,$e) failed 3 times, abort"; exit 1; }
  echo "=== merged through pair $e / 3830931 ==="
done
echo "=== TAIL DONE ==="
ls -la $OUT/gen_S10_*.i32
