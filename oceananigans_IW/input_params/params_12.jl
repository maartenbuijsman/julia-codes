# ============================================================
# params_12.jl – input parameters for IW_GM_flux_LAT_2000km_bash_cuda.jl batch run
#   forcing_metric = "flux"  ->  the two target columns are ENERGY FLUX [W/m]
#   N2source       = "zonalmean" (latitude-varying Mercator N2) — set in the .jl
#   Same M2-mode forcing as the 10/11 series, PLUS a Garrett-Munk u,v initial
#   condition (see IW_GM_flux_LAT_2000km_bash_cuda.jl / claudecodes/GM_spectrum_init_2D.jl)
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 12

# 13 runs; constant mode-1 flux = 25 kW/m, mode-2 off. runnm = 27:39, matching
# the 10-series "varying N2 MERCATOR 25kW/m" numbering convention (test run
# 12.99 verified the pipeline first). DX=4km (10-series grid, active in
# IW_GM_flux_LAT_2000km_bash_cuda.jl), GM spectrum on.
lat   = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
runnm = collect(27:39)
Usur1 = fill(25.0e3, 13)  # mode-1 flux [W/m]
Usur2 = fill(0.0, 13)     # mode-2 flux [W/m]
numM  = fill("1", 13)
