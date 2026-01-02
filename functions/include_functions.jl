#= include_functions.jl
Maarten Buijsman, USM, 2025-8-8
This file includes all function files.
pathname is defined in the main file
NOTE: instead, this file path can also be included 
in ~/.julia/config/startup.jl
=#

using DSP          # for butter, tukey and hanning windows
using Statistics   # for mean
using LinearAlgebra
using FFTW

# filtering and fft
include(string(pathname,"butter_filters.jl"));                # band/low/highpass filter functions
include(string(pathname,"gridding_functions.jl"));            # includes meshgrid
include(string(pathname,"fft_spectra_vectorized.jl"));        # tested in using_DSP.jl
include(string(pathname,"sturm_liouville_noneqDZ_norm.jl"));  # tested in testing_sturmL.jl

# other
include(string(pathname,"coriolis.jl"));  # tested in testing_sturmL.jl

# simple functions
function stop()
    throw(error("stop here"))
end


# old
#include(string(pathname,"fft_spectra.jl"));                  # NOT TESTED!! 
