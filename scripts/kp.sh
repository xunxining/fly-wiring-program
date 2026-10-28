#!/bin/bash
pkill -f "pip install" 2>/dev/null
pkill -f "pip.sh" 2>/dev/null
sleep 1
echo KILLED
