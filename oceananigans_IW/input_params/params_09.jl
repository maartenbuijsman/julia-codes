# ============================================================
# params_09.jl – input parameters for IW_Amz_200m_2000km_bash_cuda.jl batch run
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]
# runnm   : run number for each run
# Usur1   : mode-1 surface velocity amplitude [m/s]
# Usur2   : mode-2 surface velocity amplitude [m/s]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 9

# 14 runs, lat = 0:60 (the 14 latitudes with an N2_ZonalMeanAtl_lat*.jld2 file),
# runnm = 15:28
lat   = [ 0.0,  2.5,  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
runnm = [  15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28]
Usur1 = [ 0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4]
Usur2 = [ 0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2,  0.2]
numM  = [ "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "1"]
