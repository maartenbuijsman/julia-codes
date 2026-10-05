#!/bin/bash
# run_cgAPE_batch.sh -- MCB, USM, 2026-10-3
# Serial Π_K + Π_A coarse-graining (claudecodes/IW_coarsegraining_APE_tile.jl)
# 16.27-38 (GM + 25 kW/m tide), 16.1-12 (GM only), 11.1-12 (12.5 kW/m), 11.66-77 (50 kW/m); lat 0-45 N
export HDF5_USE_FILE_LOCKING=FALSE
SCR=/home/mbui/Documents/julia-codes/claudecodes/IW_coarsegraining_APE_tile.jl
cd /home/mbui/ModelOutput/diagout
run_set () {   # $1 = log tag, rest = mainnm runnms
    tag=$1; shift
    echo "=== [cgAPE] $tag started: $(date) ===" >> cgAPE_batch.log
    julia -t auto $SCR "$@" > cgAPE_${tag}.log 2>&1
    echo "=== [cgAPE] $tag finished (exit $?): $(date) ===" >> cgAPE_batch.log
}
run_set 16.27-38 16 $(seq 27 38)
run_set 16.01-12 16 $(seq 1 12)
run_set 11.01-12 11 $(seq 1 12)
run_set 11.66-77 11 $(seq 66 77)
echo "=== [cgAPE] ALL DONE: $(date) ===" >> cgAPE_batch.log
