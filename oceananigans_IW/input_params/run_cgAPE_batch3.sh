#!/bin/bash
# run_cgAPE_batch3.sh -- MCB, USM, 2026-10-4
# Replaces the remainder of run_cgAPE_batch.sh + run_cgAPE_batch2.sh with the series-15 GM sets
# moved ahead of the F and N2 sets. Waits for the running 16.01-12 julia process (PID passed as $1,
# verified before launch), then runs serially:
# 15.27-38 (old GM IC + 25 kW/m tide), 15.01-12 (old GM IC only),
# 11.01-12 (12.5 kW/m), 11.66-77 (50 kW/m), 11.40-51 (fixed N2 2.5N), 11.53-64 (fixed N2 50N); lat 0-45 N
export HDF5_USE_FILE_LOCKING=FALSE
SCR=/home/mbui/Documents/julia-codes/claudecodes/IW_coarsegraining_APE_tile.jl
cd /home/mbui/ModelOutput/diagout
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "=== [cgAPE] 16.01-12 finished (julia PID $1 exited): $(date) ===" >> cgAPE_batch.log
run_set () {   # $1 = log tag, rest = mainnm runnms
    tag=$1; shift
    echo "=== [cgAPE] $tag started: $(date) ===" >> cgAPE_batch.log
    julia -t auto $SCR "$@" > cgAPE_${tag}.log 2>&1
    echo "=== [cgAPE] $tag finished (exit $?): $(date) ===" >> cgAPE_batch.log
}
run_set 15.27-38 15 $(seq 27 38)
run_set 15.01-12 15 $(seq 1 12)
run_set 11.01-12 11 $(seq 1 12)
run_set 11.66-77 11 $(seq 66 77)
run_set 11.40-51 11 $(seq 40 51)
run_set 11.53-64 11 $(seq 53 64)
echo "=== [cgAPE] ALL SETS DONE: $(date) ===" >> cgAPE_batch.log
