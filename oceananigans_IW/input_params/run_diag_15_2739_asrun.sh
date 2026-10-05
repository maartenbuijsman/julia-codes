#!/bin/bash
# run_diag_15_2739_asrun.sh
# Maarten Buijsman, USM DMS, 2026-9-14 (generated with Claude Code)
# Watches the series 15.27-39 (GM+tide) GPU batch log and, as soon as EACH
# run finishes, runs energetics + coarse-graining for just that one latitude.
# REWRITTEN as a polling loop (not tail -F | while read) after the pipe/read
# construct died silently with "read: 0: read error: Resource temporarily
# unavailable" under nohup -- polling is more robust for a long unattended
# run. Also idempotent: skips any runnm whose output .jld2 files already
# exist, so restarting after a crash doesn't waste time redoing finished work.

cd /home/mbui/Documents/julia-codes/oceananigans_IW
LOG=/home/mbui/ModelOutput/diagout/diag_15_2739_asrun.log
BATCH_LOG=/tmp/claude-1001/-home-mbui-ModelOutput-figs/series15_2739_batch.log
DIAGOUT=/home/mbui/ModelOutput/diagout
RUNNMS=(27 28 29 30 31 32 33 34 35 36 37 38 39)

echo "==================================================" >> "$LOG"
echo "per-run diagnostics watcher (polling, mainnm=15, runnm=27:39) started: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"

for idx in "${!RUNNMS[@]}"; do
    rn=${RUNNMS[$idx]}
    n=$((idx+1))

    if [ -f "$DIAGOUT/energetics_AMZexpt15.${rn}.jld2" ] && [ -f "$DIAGOUT/Etran_AMZexpt15.${rn}.jld2" ]; then
        echo "=== runnm=$rn already diagnosed (output files exist), skipping: $(date) ===" >> "$LOG"
        continue
    fi

    while [ "$(grep -c '^Finished run ' "$BATCH_LOG" 2>/dev/null)" -lt "$n" ]; do
        sleep 60
    done

    echo "--------------------------------------------------" >> "$LOG"
    echo "=== detected completion of GPU run $n/13 (runnm=$rn), starting diagnostics: $(date) ===" >> "$LOG"

    sed -i "s/^runnms  = collect([0-9:]*)/runnms  = collect($rn:$rn)/" IW_total_energetics_tile.jl
    sed -i "s/^runnms  = collect([0-9:]*)/runnms  = collect($rn:$rn)/" IW_coarsegraining_tile.jl

    echo "=== [energetics] runnm=$rn started: $(date) ===" >> "$LOG"
    julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
    echo "=== [energetics] runnm=$rn finished: $(date) ===" >> "$LOG"

    echo "=== [coarsegraining] runnm=$rn started: $(date) ===" >> "$LOG"
    julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
    echo "=== [coarsegraining] runnm=$rn finished: $(date) ===" >> "$LOG"
done

echo "==================================================" >> "$LOG"
echo "all 13 runs diagnosed, watcher exiting: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"
