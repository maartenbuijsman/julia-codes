# ============================================================
# params.jl  –  input parameters for IW_Amz2_cuda batch run
#
# mainnm  : experiment number (single integer)
# lat     : vector of latitudes to loop over
# runnm   : vector of run numbers, one per latitude entry
#           (must have the same length as lat)
# ============================================================

mainnm = 1

lat   = [5.0, 10.0, 20.0, 30.0, 40.0]
runnm = [49,  50,   51,   52,   53  ]
