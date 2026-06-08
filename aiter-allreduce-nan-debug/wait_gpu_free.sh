#!/usr/bin/env bash
# Poll 08-25 until mingzhi's UT container is gone AND all GPUs are idle (2x in a row).
free_streak=0
while true; do
  ts=$(date '+%H:%M:%S')
  mingzhi=$(docker ps --format '{{.Names}}' | grep -c 'mingzhi-fmoe-ut')
  maxbusy=$(rocm-smi --showuse 2>/dev/null | grep -oE 'GPU use \(%\): [0-9]+' | grep -oE '[0-9]+$' | sort -rn | head -1)
  maxbusy=${maxbusy:-0}
  if [[ "$mingzhi" == "0" && "$maxbusy" -lt 10 ]]; then
    free_streak=$((free_streak+1))
    echo "[$ts] candidate-free (mingzhi=$mingzhi maxbusy=${maxbusy}% streak=$free_streak)"
    if [[ "$free_streak" -ge 2 ]]; then
      echo "GPU_FREE_DETECTED at $ts (mingzhi gone, maxbusy=${maxbusy}%)"
      break
    fi
  else
    free_streak=0
    echo "[$ts] busy (mingzhi=$mingzhi maxbusy=${maxbusy}%)"
  fi
  sleep 120
done
