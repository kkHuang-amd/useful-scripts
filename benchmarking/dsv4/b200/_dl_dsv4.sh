#!/usr/bin/env bash
# Resumable retry wrapper for the DeepSeek-V4-Pro download onto the NFS mount.
# `hf download` is resumable, so each retry skips already-completed shards.
# Xet is disabled because its concurrent temp-file move races on this NFS mount.
set -uo pipefail
# Xet is much faster than plain HTTPS here (~430 vs ~30 MB/s). Its only issue is
# an occasional NFS temp-move race, which this retry loop absorbs via resume.
export HF_XET_HIGH_PERFORMANCE=1
DEST=/dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro
LOG=/dockerx/home/wunhuang/useful-scripts/benchmarking/dsv4/dsv4_download.log
for attempt in $(seq 1 40); do
    echo "=== attempt $attempt $(date --iso-8601=seconds) ===" >> "$LOG"
    hf download deepseek-ai/DeepSeek-V4-Pro --local-dir "$DEST" \
        --max-workers 4 >> "$LOG" 2>&1
    rc=$?
    n=$(ls "$DEST"/*.safetensors 2>/dev/null | wc -l)
    echo "=== attempt $attempt exit=$rc shards=$n/64 ===" >> "$LOG"
    if [ "$rc" -eq 0 ] && [ "$n" -eq 64 ]; then
        echo "=== DOWNLOAD COMPLETE ===" >> "$LOG"
        exit 0
    fi
    sleep 5
done
echo "=== GAVE UP after 40 attempts ===" >> "$LOG"
exit 1
