#!/bin/bash
pkill -f dlw.py 2>/dev/null
sleep 2
rm -f /tmp/pyarrow.whl /tmp/wheel_part_*
echo CLEANED
