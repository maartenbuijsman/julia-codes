#!/bin/bash
# ============================================================
# run_batch.sh
#
# Reads params*.jl, then loops over the (lat, runnm) pairs,
# launching a fresh Julia process for each one so that the GPU
# is fully released between runs.
#
# Usage:
#   chmod +x run_batch.sh
#   ./run_batch.sh [params_file]          # defaults to params.jl
#
# SLURM example (one node, one GPU):
#   sbatch run_batch.sh
# ============================================================

#SBATCH --job-name=IW_Amz2
#SBATCH --output=logs/IW_Amz2_%j.log
#SBATCH --error=logs/IW_Amz2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# ---- configuration ----------------------------------------
PARAMS_FILE="${1:-params_IW_Amz_4km_2k_bash_cuda.jl}"          # first argument or default
JL_PTH="/home/mbui/Documents/julia-codes/oceananigans_IW"      # path of .jl file
JULIA_SCRIPT="${JL_PTH}/IW_Amz_4km_2k_bash_cuda.jl"            # path to simulation script
JULIA_BIN="${JULIA:-julia}"            # honour $JULIA env var if set
JULIA_THREADS="${JULIA_THREADS:-$(nproc)}"    # number of Julia threads (auto-detects number of available cores)
#JULIA_THREADS="${JULIA_THREADS:-4}"    # number of Julia threads
LOG_DIR="logs"
# -----------------------------------------------------------

mkdir -p "$LOG_DIR"

echo "=================================================="
echo "Batch run started: $(date)"
echo "Parameter file   : $PARAMS_FILE"
echo "Julia script     : $JULIA_SCRIPT"
echo "Julia binary     : $JULIA_BIN"
echo "Julia threads    : $JULIA_THREADS"
echo "=================================================="

# ---- parse params.jl with Julia ---------------------------
# Emit one line per run:  <mainnm> <runnm> <lat>
PARAM_LINES=$(
$JULIA_BIN --startup-file=no -e "
include(\"$PARAMS_FILE\")
for (r, l) in zip(runnm, lat)
    println(mainnm, \" \", r, \" \", l)
end
"
)

if [ -z "$PARAM_LINES" ]; then
    echo "ERROR: could not parse $PARAMS_FILE – exiting."
    exit 1
fi

NTOTAL=$(echo "$PARAM_LINES" | wc -l)
echo "Total runs to execute: $NTOTAL"
echo ""

# ---- main loop --------------------------------------------
RUN_IDX=0
while IFS=' ' read -r MAINNM RUNNM LAT; do
    RUN_IDX=$((RUN_IDX + 1))
    RUNLOG="${LOG_DIR}/run_${MAINNM}_${RUNNM}_lat${LAT}.log"

    echo "--------------------------------------------------"
    echo "Run ${RUN_IDX}/${NTOTAL}  |  mainnm=${MAINNM}  runnm=${RUNNM}  lat=${LAT}"
    echo "Log: $RUNLOG"
    echo "Start: $(date)"

    # Launch Julia, wait for it to finish, then continue.
    # Each invocation is a completely fresh process – GPU memory
    # is freed when Julia exits before the next run begins.
    $JULIA_BIN \
        --startup-file=no \
        --threads=$JULIA_THREADS \
        "$JULIA_SCRIPT" \
        "$MAINNM" "$RUNNM" "$LAT" \
        2>&1 | tee "$RUNLOG"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: Julia exited with code $EXIT_CODE for run ${MAINNM}.${RUNNM} (lat=${LAT})."
        echo "Check $RUNLOG for details."
        # Change 'continue' to 'exit 1' if you want the whole batch to stop on failure.
        continue
    fi

    echo "Finished run ${RUN_IDX}/${NTOTAL}  |  $(date)"

done <<< "$PARAM_LINES"

echo ""
echo "=================================================="
echo "All $NTOTAL runs completed: $(date)"
echo "=================================================="
