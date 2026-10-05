# ============================================================
# params_16_cal.jl – input parameters for claudecodes/IW_GM81_flux_LAT_2000km_bash_cuda.jl
#   SERIES 16 CALIBRATION runs: GM81 initial condition only (corrected
#   amplitude, see claudecodes/gm81_ic.jl), Flux=0 -> NO tidal forcing,
#   hourly output only (outfine=0). Purpose: measure the retention
#   R16 = E(day 10-20 mean)/E(0) at three latitudes and compare with series
#   15 (R15), to set the per-latitude initial level GMs = E(0)/E_GM81 that
#   puts the day 10-20 mean at 1x GM81 in the production runs 16.1-12 / 16.27-38.
#   First guess GMs = 1/R15 (0 N: inert v removed from series-15 numbers).
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]  (must have an N2_ZonalMeanAtl_lat*.jld2 file)
# runnm   : run number for each run (91-93 = calibration slots)
# Usur1   : mode-1 forcing target = ENERGY FLUX F1 [W/m]   (column name kept for run_batch parser)
# Usur2   : mode-2 forcing target = ENERGY FLUX F2 [W/m]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# GMs     : initial GM energy E(0) as a multiple of E_GM81
# outfine : 1 = production output (5-min, days 10-20), 0 = hourly only
# ============================================================

mainnm = 16

lat     = [ 0.0,  5.0, 28.8]
runnm   = [91, 92, 93]
Usur1   = fill(0.0, 3)   # mode-1 flux [W/m] -- zero: no tidal forcing
Usur2   = fill(0.0, 3)   # mode-2 flux [W/m]
numM    = fill("1", 3)
GMs     = [2.06, 1.92, 2.38]
outfine = fill(0, 3)
