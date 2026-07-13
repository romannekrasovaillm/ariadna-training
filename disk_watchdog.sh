#!/bin/bash
# Disk watchdog for Vast AI — per vastai-checkpoint-management skill §3
CKPT_DIR="${1:-/workspace/checkpoints}"
KEEP_LAST="${2:-2}"
WARN_PCT=80
CRIT_PCT=92
INTERVAL=60
LOG="/workspace/watchdog.log"

log(){ echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
pct(){ df -P / | awk 'NR==2{gsub("%","",$5);print $5}'; }

while true; do
    used=$(pct)
    if [ "$used" -ge "$CRIT_PCT" ]; then
        log "CRIT: disk ${used}% — cleaning old checkpoints and caches"
        find "$CKPT_DIR" -maxdepth 1 -type d -name 'step-*' -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n -${KEEP_LAST} | awk '{print $2}' | while read d; do
            log "DELETE $d ($(du -sh "$d" 2>/dev/null | cut -f1))"
            rm -rf "$d"
        done
        pip cache purge 2>/dev/null
        rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B-Base/blobs 2>/dev/null
    elif [ "$used" -ge "$WARN_PCT" ]; then
        count=$(find "$CKPT_DIR" -maxdepth 1 -type d -name 'step-*' 2>/dev/null | wc -l)
        if [ "$count" -gt "$KEEP_LAST" ]; then
            log "WARN: disk ${used}%, ${count} checkpoints — rotating to ${KEEP_LAST}"
            find "$CKPT_DIR" -maxdepth 1 -type d -name 'step-*' -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n -${KEEP_LAST} | awk '{print $2}' | while read d; do
                log "DELETE $d"
                rm -rf "$d"
            done
        fi
    fi
    sleep "$INTERVAL"
done
