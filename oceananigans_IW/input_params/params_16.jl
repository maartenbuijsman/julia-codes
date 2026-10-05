# ============================================================
# params_16.jl – input parameters for claudecodes/IW_GM81_flux_LAT_2000km_bash_cuda.jl
#   SERIES 16 production, GM + D2 tide (mode-1 flux 25 kW/m, as series 10-15), lat 0-45 N.
#   GM81 initial condition with the corrected amplitude (claudecodes/gm81_ic.jl),
#   rescaled to E(0) = GMs x E_GM81. GMs from the calibration runs 16.91-93
#   (claudecodes/IW_GM16_calibration_eval.jl, 2026-9-30): GMs = s0/c with
#   s0 = 1/R15 and c = R16/R15 = 1.018 (0 N), 0.947 (5 N), 0.939 (28.8 N),
#   linear in latitude in between, constant beyond 28.8 N. Target: day 10-20
#   mean = 1x GM81. Same seed and GMs as 16.1-12 (GM only) -> identical GM field.
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# GMs     : initial GM energy E(0) as a multiple of E_GM81
# outfine : 1 = production output (5-min, days 10-20), 0 = hourly only
# ============================================================

mainnm = 16

lat     = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0]
runnm   = collect(27:38)
Usur1   = fill(25.0e3, 12)  # mode-1 flux [W/m]
Usur2   = fill(0.0, 12)  # mode-2 flux [W/m]
numM    = fill("1", 12)
GMs     = [2.02, 2.20, 2.03, 2.25, 2.17, 2.48, 2.39, 2.54, 2.52, 2.44, 2.59, 2.68]
outfine = fill(1, 12)
