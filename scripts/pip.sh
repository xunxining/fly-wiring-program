#!/bin/bash
for i in 1 2 3 4 5; do
  pip install pyarrow && break
  sleep 5
done
pip list 2>/dev/null | grep -i pyarrow
echo PIP_DONE
