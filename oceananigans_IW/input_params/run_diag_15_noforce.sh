#!/bin/bash
# run_diag_15_noforce.sh
# Maarten Buijsman, USM DMS, 2026-9-13 (generated with Claude Code)
# Energy + coarse-graining diagnostics for the mainnm=15 no-forcing (Flux=0,
# free-decay, redistribution-fix GM IC) series. All 13 runs already complete,
# so this just runs energetics then coarse-graining once each, serially, for
# runnms=1:13 (mainnm/runnms already set in both .jl files).

cd /home/mbui/Documents/julia-codes/oceananigans_IW
LOG=/home/mbui/ModelOutput/diagout/diag_15_noforce_run.log

echo "==================================================" >> "$LOG"
echo "diag pipeline (mainnm=15 no-forcing series) started: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"

echo "=== [1/2] runnm=1-13 energetics started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [1/2] runnm=1-13 energetics finished: $(date) ===" >> "$LOG"

echo "=== [2/2] runnm=1-13 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [2/2] runnm=1-13 coarsegraining finished: $(date) ===" >> "$LOG"

echo "==================================================" >> "$LOG"
echo "ALL mainnm=15 NO-FORCING DIAGNOSTICS COMPLETE: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"
