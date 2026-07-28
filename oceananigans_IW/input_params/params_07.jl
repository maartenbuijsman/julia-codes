# ============================================================
# params_06.jl – input parameters for IW_Amz_4km_700km_bash_cuda.jl batch run
#
# mainnm  : experiment number (single integer)
# lat     : latitude for each run [deg]
# runnm   : run number for each run
# Usur1   : mode-1 surface velocity amplitude [m/s]
# Usur2   : mode-2 surface velocity amplitude [m/s]
# numM    : mode selection string — "1", "2", or "1,2" for both modes
# ============================================================

mainnm = 7

#         run1   run2   run3
lat   = [ 20.0,  ]
runnm = [ 19,   ]
#Usur1 = [ 0.4, ]
Usur1 = [ 0.6,  ]
#Usur2 = [ 0.2, ]
Usur2 = [ 0.3,  ]
numM  = ["1",   ]
