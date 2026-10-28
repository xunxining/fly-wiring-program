#!/bin/bash
# 84b_dl_part.sh <i> — self-healing ranged download of syn-partners part i
set -u
I=$1
URL="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather"
SZ=2965702122; NP=8; CH=$((SZ / NP + 1))
S=$((I * CH)); E=$(( (I+1) * CH - 1 )); [ $E -ge $SZ ] && E=$((SZ-1))
EX=$((E - S + 1)); F="/tmp/synp_${I}"
cd /tmp
for att in $(seq 1 60); do
  CUR=0; [ -f "$F" ] && CUR=$(stat -c%s "$F")
  if [ "$CUR" -ge "$EX" ]; then echo "part $I complete: $CUR"; exit 0; fi
  RS=$((S + CUR))
  curl -sS --speed-time 30 --speed-limit 200000 -r ${RS}-${E} -o - "$URL" >> "$F"
  echo "part $I attempt $att ended, size $(stat -c%s "$F")/$EX"
  sleep 2
done
echo "part $I FAILED after 60 attempts"; exit 1
