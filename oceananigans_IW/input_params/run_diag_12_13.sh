#!/bin/bash
# run_diag_12_13.sh
# Maarten Buijsman, USM DMS, 2026-9-4
# Sequential energetics + coarse-graining diagnostics for the GM-spectrum series:
# mainnm=12 (4km) first, then mainnm=13 (200m), runnm 27-39 both times.
# Toggles the hardcoded `mainnm` line in IW_total_energetics_tile.jl and
# IW_coarsegraining_tile.jl between stages (same sed-toggle pattern used for
# DX in the GM batch run).

cd /home/mbui/Documents/julia-codes/oceananigans_IW
LOG=/home/mbui/ModelOutput/diagout/diag_12_13_run.log

echo "==================================================" >> "$LOG"
echo "diag pipeline started: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"

echo "=== [1/4] mainnm=12 energetics started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [1/4] mainnm=12 energetics finished: $(date) ===" >> "$LOG"

echo "=== [2/4] mainnm=12 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [2/4] mainnm=12 coarsegraining finished: $(date) ===" >> "$LOG"

# toggle mainnm 12 -> 13 in both scripts
sed -i 's/^mainnm  = 12/mainnm  = 13/' IW_total_energetics_tile.jl
sed -i 's/^mainnm  = 12/mainnm  = 13/' IW_coarsegraining_tile.jl
echo "=== toggled mainnm 12 -> 13: $(date) ===" >> "$LOG"

echo "=== [3/4] mainnm=13 energetics started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [3/4] mainnm=13 energetics finished: $(date) ===" >> "$LOG"

echo "=== [4/4] mainnm=13 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [4/4] mainnm=13 coarsegraining finished: $(date) ===" >> "$LOG"

echo "==================================================" >> "$LOG"
echo "ALL DIAGNOSTICS COMPLETE: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"
