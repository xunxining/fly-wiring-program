#!/bin/bash
pkill -f "dl.py" 2>/dev/null
sleep 1
cd /tmp/flydata && nohup python3 /mnt/agents/output/果蝇生成器/dl2.py > /tmp/flydata/dl2.log 2>&1 &
echo RELAUNCHED
