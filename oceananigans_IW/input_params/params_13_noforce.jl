# ============================================================
# params_13_noforce.jl – input parameters for IW_GM_flux_LAT_2000km_bash_cuda.jl batch run
#   forcing_metric = "flux"  ->  the two target columns are ENERGY FLUX [W/m]
#   N2source       = "zonalmean" (latitude-varying Mercator N2) — set in the .jl
#   GM-spectrum u,v initial condition ONLY, Flux=0 -> NO tidal forcing.
#   Free-decay/background GM-spectrum evolution, no external tidal energy input.
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 13

# 13 runs; DX=200m (already active in IW_GM_flux_LAT_2000km_bash_cuda.jl); GM
# spectrum initial condition, Flux=0 for both modes (no tidal forcing).
# runnm = 1:13, matching the 10-series "varying N2 MERCATOR" numbering slot.
lat   = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
runnm = collect(1:13)
Usur1 = fill(0.0, 13)  # mode-1 flux [W/m] -- zero: no tidal forcing
Usur2 = fill(0.0, 13)  # mode-2 flux [W/m] -- zero: no tidal forcing
numM  = fill("1", 13)
