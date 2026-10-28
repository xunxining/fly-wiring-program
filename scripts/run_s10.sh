#!/bin/bash
# S=10 分段驱动: 每段写临时文件, 成功后并入主文件; 失败删临时文件重试
set -u
TOTAL=3830931
SEG=300000
OUT=/tmp/flydata/gen
cd /mnt/agents/output/果蝇生成器

echo "=== prep: build w_emp/type_sc ==="
python3 09a_prep_s10.py || exit 1

echo "=== clean old output ==="
rm -f $OUT/gen_S10_pre.i32 $OUT/gen_S10_post.i32 $OUT/gen_S10_w.i32 $OUT/seg*_pre.i32 $OUT/seg*_post.i32 $OUT/seg*_w.i32
touch $OUT/gen_S10_pre.i32 $OUT/gen_S10_post.i32 $OUT/gen_S10_w.i32

for s in $(seq 0 $SEG $((TOTAL-1))); do
  e=$((s+SEG)); [ $e -gt $TOTAL ] && e=$TOTAL
  ok=0
  for attempt in 1 2 3; do
    echo "=== segment [$s,$e) attempt $attempt ==="
    if python3 09_generate_s10.py $s $e; then
      cat $OUT/seg${s}_pre.i32 >> $OUT/gen_S10_pre.i32
      cat $OUT/seg${s}_post.i32 >> $OUT/gen_S10_post.i32
      cat $OUT/seg${s}_w.i32 >> $OUT/gen_S10_w.i32
      rm -f $OUT/seg${s}_pre.i32 $OUT/seg${s}_post.i32 $OUT/seg${s}_w.i32
      ok=1
      break
    fi
    echo "!!! segment [$s,$e) failed, cleaning and retrying"
    rm -f $OUT/seg${s}_pre.i32 $OUT/seg${s}_post.i32 $OUT/seg${s}_w.i32
  done
  if [ $ok -eq 0 ]; then
    echo "!!! segment [$s,$e) failed 3 times, abort"
    exit 1
  fi
  echo "=== merged through pair $e / $TOTAL ==="
done
echo "=== ALL SEGMENTS DONE ==="
ls -la $OUT/gen_S10_*.i32
