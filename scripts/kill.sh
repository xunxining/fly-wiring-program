#!/bin/bash
for p in $(pgrep -f "flyem-male-cns"); do kill "$p" 2>/dev/null; done
sleep 1
echo cleaned
