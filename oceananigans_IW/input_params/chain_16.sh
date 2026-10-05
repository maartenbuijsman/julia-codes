#!/bin/bash
# chain_16.sh -- start the series-16 GM + tide batch (params_16.jl) as soon as
# the GM-only batch (params_16_noforce.jl) has ended: waits for its
# "All ... runs completed" line and for the GPU sim process to be gone.
# Launch detached:  setsid nohup ./chain_16.sh > logs/batch16_tide_nohup.log 2>&1 < /dev/null &
cd /home/mbui/Documents/julia-codes/oceananigans_IW/input_params
until grep -q "All .* runs completed" logs/batch16_noforce_nohup.log; do sleep 60; done
while pgrep -f "julia.*IW_GM81_flux_LAT" > /dev/null; do sleep 60; done
echo "chain_16: GM-only batch ended $(date); starting params_16.jl"
exec ./run_batch_16.sh params_16.jl
