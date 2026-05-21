# ============================================================
# params_05.jl –  input parameters for IW_K1_4km_2k_bash_cuda.jl batch run
#
# mainnm  : experiment number (single integer)
# lat     : vector of latitudes to loop over
# runnm   : vector of run numbers, one per latitude entry
#           (must have the same length as lat)
# ============================================================

mainnm = 5

lat   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
runnm = collect(1:8)

# runnm = [1,   2,   3,   4,    5,    6,    7,    8]  # is the same
