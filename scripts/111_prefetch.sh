#!/bin/bash
# 111_prefetch.sh — 把雄性数据并行预取到 /tmp/dm (tmpfs), 带重试
SRC="/mnt/agents/output/果蝇生成器/data_male"
DST="/tmp/dm"
mkdir -p "$DST"
FILES="genome_m.npz lat_snap_m.npz lat2q_m.npz np_masks_m2.npz spatial_kernel_m.npz dyad_m.npz dyad_cap_m.npz node_sc.i32 xyz.f32 e_pre.i32 e_post.i32 e_w.i32 act_in.f32 cq_cell.i32 cq_t.i32 cq_cnt.i32"
for f in $FILES; do
  (
    for a in $(seq 1 100); do
      s_src=$(stat -c %s "$SRC/$f" 2>/dev/null) || { sleep 3; continue; }
      s_dst=$(stat -c %s "$DST/$f" 2>/dev/null || echo 0)
      [ "$s_dst" = "$s_src" ] && exit 0
      cp "$SRC/$f" "$DST/$f.part" 2>/dev/null && mv "$DST/$f.part" "$DST/$f" && exit 0
      sleep 3
    done
    echo "PREFETCH FAIL $f" >&2; exit 1
  ) &
done
wait
ls -la "$DST" | head -20
echo "PREFETCH DONE $(date +%H:%M:%S)"
