# ============================================================
# params_15_noforce.jl – input parameters for IW_GM_flux_LAT_2000km_bash_cuda.jl batch run
#   forcing_metric = "flux"  ->  the two target columns are ENERGY FLUX [W/m]
#   N2source       = "zonalmean" (latitude-varying Mercator N2) — set in the .jl
#   GM-spectrum u,v,w,b initial condition ONLY (w,b-consistent + REDISTRIBUTION
#   domain-length-cutoff fix, see chat), Flux=0 -> NO tidal forcing.
#   Free-decay/background GM-spectrum evolution, no external tidal energy input.
#   mainnm=14's counterpart with the redistribution fix instead of k-clamp
#   (mainnm=14 was stopped early at runnm 1:5 after the k-clamp fix was found
#   to cause anomalously slow decay at low latitude) -- for comparison of
#   spectra and decay across the same 13 latitudes.
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 15

# 13 runs; DX=200m; GM spectrum initial condition (w,b-consistent +
# redistribution fix), Flux=0 for both modes (no tidal forcing). runnm = 1:13,
# matching params_13/14_noforce.jl's latitude ordering exactly.
lat   = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
runnm = collect(1:13)
Usur1 = fill(0.0, 13)  # mode-1 flux [W/m] -- zero: no tidal forcing
Usur2 = fill(0.0, 13)  # mode-2 flux [W/m] -- zero: no tidal forcing
numM  = fill("1", 13)
