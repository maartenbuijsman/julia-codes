# ============================================================
# params_IW_Amz_4km_2k_bash_cuda.jl –  input parameters for IW_Amz_4km_2k_bash_cuda.jl batch run
#
# mainnm  : experiment number (single integer)
# lat     : vector of latitudes to loop over
# runnm   : vector of run numbers, one per latitude entry
#           (must have the same length as lat)
# ============================================================

mainnm = 3

lat   = [0.0]
runnm = [16]

#lat   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 35.0, 40.0, 50.0]
#runnm = [3,   4,   5,   6,    7,    8,    9,    10,   11,   12,   13,   14]
