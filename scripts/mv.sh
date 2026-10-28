#!/bin/bash
pkill -f "python3 dl.py" 2>/dev/null
sleep 1
mkdir -p /tmp/flydata
mv /mnt/agents/output/果蝇生成器/data/part_* /tmp/flydata/ 2>/dev/null
cp /mnt/agents/output/果蝇生成器/data/*.feather /tmp/flydata/ 2>/dev/null
ls -la /tmp/flydata/ | head -15
echo MOVED
