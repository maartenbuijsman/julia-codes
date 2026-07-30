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

#         run1   run2   run3  run4
lat   = [10.0,  15.0,  20.0,  25.0]
runnm = [   4,     5,     6,     7]
Usur1 = [ 0.4,   0.4,   0.4,   0.4]
Usur2 = [ 0.2,   0.2,   0.2,   0.2]
numM  = [ "1",   "1",   "1",   "1"]
