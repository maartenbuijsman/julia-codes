#!/bin/bash
# run_cgAPE_batch2.sh -- MCB, USM, 2026-10-3
# N2-comparison sets for Π_K + Π_A coarse-graining (claudecodes/IW_coarsegraining_APE_tile.jl),
# started after run_cgAPE_batch.sh writes "[cgAPE] ALL DONE" to cgAPE_batch.log (log-line waiter, not pgrep)
# 11.40-51 (25 kW/m, fixed N2 2.5 N), 11.53-64 (25 kW/m, fixed N2 50 N); lat 0-45 N
export HDF5_USE_FILE_LOCKING=FALSE
SCR=/home/mbui/Documents/julia-codes/claudecodes/IW_coarsegraining_APE_tile.jl
cd /home/mbui/ModelOutput/diagout
until grep -q "\[cgAPE\] ALL DONE" cgAPE_batch.log; do sleep 60; done
run_set () {   # $1 = log tag, rest = mainnm runnms
    tag=$1; shift
    echo "=== [cgAPE] $tag started: $(date) ===" >> cgAPE_batch.log
    julia -t auto $SCR "$@" > cgAPE_${tag}.log 2>&1
    echo "=== [cgAPE] $tag finished (exit $?): $(date) ===" >> cgAPE_batch.log
}
run_set 11.40-51 11 $(seq 40 51)
run_set 11.53-64 11 $(seq 53 64)
echo "=== [cgAPE] N2 sets ALL DONE: $(date) ===" >> cgAPE_batch.log
