# ============================================================
# params_15.jl – input parameters for IW_GM_flux_LAT_2000km_bash_cuda.jl batch run
#   forcing_metric = "flux"  ->  the two target columns are ENERGY FLUX [W/m]
#   N2source       = "zonalmean" (latitude-varying Mercator N2) — set in the .jl
#   Same M2-mode tidal forcing (flux=25 kW/m) as the 10-13 series, PLUS the
#   REDISTRIBUTION-fix Garrett-Munk u,v,w,b initial condition (current
#   IW_GM_flux_LAT_2000km_bash_cuda.jl) -- for direct comparison against the
#   10-13 series at matched flux.
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 15

# 13 runs; constant mode-1 flux = 25 kW/m, mode-2 off. runnm = 27:39, matching
# params_13.jl's "13.27-39" GM+tide series numbering exactly (same runnm slots,
# same latitudes, same 25 kW/m flux as the 10-13 series). DX=200m, GM spectrum + M2 tide.
lat   = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
runnm = collect(27:39)
Usur1 = fill(25.0e3, 13)  # mode-1 flux [W/m]
Usur2 = fill(0.0, 13)     # mode-2 flux [W/m]
numM  = fill("1", 13)
