# ============================================================
# params_10.jl – input parameters for IW_flux_LAT_2000km_bash_cuda.jl batch run
#   forcing_metric = "flux"  ->  the two target columns are ENERGY FLUX [W/m]
#   N2source       = "zonalmean" (latitude-varying Mercator N2) — set in the .jl
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 11

# 13 runs; constant mode-1 flux = 25 kW/m, mode-2 off (flux=0). runnm = 1:13.
# latitudes include 28.8 (M2 PSI critical latitude).
lat   = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
#runnm = collect(27:39)
#runnm = collect(40:52)
#runnm = collect(53:65)
runnm = collect(66:78)
Usur1 = fill(25e3,13)  # mode-1 flux [W/m]
Usur2 = fill(0.0,13)   # mode-2 flux [W/m]
numM  = fill("1",13)
