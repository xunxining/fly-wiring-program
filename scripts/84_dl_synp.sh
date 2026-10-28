#!/bin/bash
# 84_dl_synp.sh — parallel ranged download of male syn-partners feather (2,965,702,122 B)
set -u
URL="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather"
SZ=2965702122
NP=8
CH=$((SZ / NP + 1))
cd /tmp
for i in $(seq 0 $((NP-1))); do
  S=$((i * CH)); E=$(( (i+1) * CH - 1 )); [ $E -ge $SZ ] && E=$((SZ-1))
  EX=$((E - S + 1))
  F="synp_${i}"
  CUR=0; [ -f "$F" ] && CUR=$(stat -c%s "$F")
  if [ "$CUR" -eq "$EX" ]; then echo "part $i done"; continue; fi
  RS=$((S + CUR))
  ( curl -sS --retry 5 -r ${RS}-${E} -o - "$URL" >> "$F" ) &
done
wait
# verify
OK=1
for i in $(seq 0 $((NP-1))); do
  S=$((i * CH)); E=$(( (i+1) * CH - 1 )); [ $E -ge $SZ ] && E=$((SZ-1))
  EX=$((E - S + 1)); CUR=$(stat -c%s "synp_${i}")
  echo "part $i: $CUR / $EX"
  [ "$CUR" -ne "$EX" ] && OK=0
done
if [ $OK -eq 1 ]; then
  cat synp_0 synp_1 synp_2 synp_3 synp_4 synp_5 synp_6 synp_7 > synp_male.feather
  TOT=$(stat -c%s synp_male.feather)
  echo "TOTAL: $TOT / $SZ"
  [ "$TOT" -eq "$SZ" ] && rm -f synp_0 synp_1 synp_2 synp_3 synp_4 synp_5 synp_6 synp_7 && echo "ASSEMBLED OK"
else
  echo "INCOMPLETE — rerun this script to resume"
fi
