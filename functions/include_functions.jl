#= include_functions.jl
Maarten Buijsman, USM, 2025-7-30
This file includes all function files.
pathname is defined in the main file
NOTE: instead, this file can also be included 
in ~/.julia/config/startup.jl
=#

using DSP          # for butter, tukey and hanning windows
using Statistics   # for mean
using LinearAlgebra
using FFTW

# filtering and fft
include(string(pathname,"butter_filters.jl"))
include(string(pathname,"fft_spectra_vectorized.jl"))   # tested in using_DSP.jl
#include(string(pathname,"fft_spectra.jl"))             # NOT TESTED!! 

