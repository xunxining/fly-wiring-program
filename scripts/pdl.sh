#!/bin/bash
URL="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
TOTAL=1051241946
N=8
CHUNK=$((TOTAL/N+1))
cd /mnt/agents/output/果蝇生成器/data
for i in $(seq 0 $((N-1))); do
  S=$((i*CHUNK)); E=$(( (i+1)*CHUNK-1 )); if [ $E -ge $TOTAL ]; then E=$((TOTAL-1)); fi
  if [ ! -f part_$i.done ]; then
    (curl -sL --retry 5 --retry-all-errors -r $S-$E -o part_$i "$URL" && touch part_$i.done) &
  fi
done
wait
cat part_0 part_1 part_2 part_3 part_4 part_5 part_6 part_7 > connectome-weights.feather
rm -f part_*.done part_0 part_1 part_2 part_3 part_4 part_5 part_6 part_7
echo ALLDONE
