#!/bin/bash
# run_diag_13_resume.sh
# Maarten Buijsman, USM DMS, 2026-9-4
# Resume run_diag_12_13.sh from stage 3/4 onward: mainnm=12 stages already
# completed and saved; mainnm is already toggled to 13 in both scripts.
# Retry after the first attempt hung on a pathologically slow surface-velocity
# read (1TB+ physically read for an 82GB file, no crash/error, just stalled I/O
# of unclear cause -- retrying fresh per Maarten's call).

cd /home/mbui/Documents/julia-codes/oceananigans_IW
LOG=/home/mbui/ModelOutput/diagout/diag_12_13_run.log

echo "=== [3/4] mainnm=13 energetics RETRY started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [3/4] mainnm=13 energetics finished: $(date) ===" >> "$LOG"

echo "=== [4/4] mainnm=13 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [4/4] mainnm=13 coarsegraining finished: $(date) ===" >> "$LOG"

echo "==================================================" >> "$LOG"
echo "ALL DIAGNOSTICS COMPLETE: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"
