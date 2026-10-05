#!/bin/bash
# run_diag_13_noforce.sh
# Maarten Buijsman, USM DMS, 2026-9-5
# Energy + coarse-graining diagnostics for the mainnm=13 no-forcing (Flux=0,
# free-decay) series. Runs 1-9 are already complete (GPU batch, PID 3945630,
# params_13_noforce.jl); runs 10-13 are still running on the GPU when this is
# launched. Do 1-9 now, then wait for the GPU batch to finish, toggle runnms
# to 10-13 in both scripts, and do those too.

cd /home/mbui/Documents/julia-codes/oceananigans_IW
LOG=/home/mbui/ModelOutput/diagout/diag_13_noforce_run.log
GPU_WRAPPER_PID=3945630

echo "==================================================" >> "$LOG"
echo "diag pipeline (no-forcing series) started: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"

echo "=== [1/4] runnm=1-9 energetics started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [1/4] runnm=1-9 energetics finished: $(date) ===" >> "$LOG"

echo "=== [2/4] runnm=1-9 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [2/4] runnm=1-9 coarsegraining finished: $(date) ===" >> "$LOG"

echo "=== waiting for GPU batch (PID $GPU_WRAPPER_PID) to finish runs 10-13: $(date) ===" >> "$LOG"
while kill -0 "$GPU_WRAPPER_PID" 2>/dev/null; do
    sleep 120
done
echo "=== GPU batch finished, proceeding: $(date) ===" >> "$LOG"

# toggle runnms 1:9 -> 10:13 in both scripts
sed -i 's/^runnms  = collect(1:9)/runnms  = collect(10:13)/' IW_total_energetics_tile.jl
sed -i 's/^runnms  = collect(1:9)/runnms  = collect(10:13)/' IW_coarsegraining_tile.jl
echo "=== toggled runnms 1:9 -> 10:13: $(date) ===" >> "$LOG"

echo "=== [3/4] runnm=10-13 energetics started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_total_energetics_tile.jl >> "$LOG" 2>&1
echo "=== [3/4] runnm=10-13 energetics finished: $(date) ===" >> "$LOG"

echo "=== [4/4] runnm=10-13 coarsegraining started: $(date) ===" >> "$LOG"
julia --startup-file=no --threads=auto IW_coarsegraining_tile.jl >> "$LOG" 2>&1
echo "=== [4/4] runnm=10-13 coarsegraining finished: $(date) ===" >> "$LOG"

echo "==================================================" >> "$LOG"
echo "ALL NO-FORCING DIAGNOSTICS COMPLETE: $(date)" >> "$LOG"
echo "==================================================" >> "$LOG"
