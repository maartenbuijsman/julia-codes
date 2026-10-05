#!/bin/bash
# run_diag_16_noforce.sh
# Maarten Buijsman, USM DMS, 2026-10-1 (generated with Claude Code)
# Energy + coarse-graining diagnostics for the mainnm=16 no-forcing (Flux=0,
# GM81 IC calibrated to 1x GM81 over days 10-20) series, runnm=1:12 (lat 0-45 N).
# Same as run_diag_15_noforce.sh: energetics then coarse-graining, serially
# (mainnm/runnms set in both .jl files). Can run while the GPU sims run.

export HDF5_USE_FILE_LOCKING=FALSE   # never lock the .nc files the GPU batch may touch
cd /home/mbui/Documents/julia-codes/oceananigans_IW
LOG=/home/mbui/ModelOutput/diagout/diag_16_noforce_run.log

echo "==================================================" >> "$LOG"
echo "diag pipeline (mainnm=16 no-forcing series) started: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"

echo "=== [1/2] runnm=1-12 energetics started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [1/2] runnm=1-12 energetics finished: $(date) ===" >> "$LOG"

echo "=== [2/2] runnm=1-12 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [2/2] runnm=1-12 coarsegraining finished: $(date) ===" >> "$LOG"

echo "==================================================" >> "$LOG"
echo "ALL mainnm=16 NO-FORCING DIAGNOSTICS COMPLETE: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"
