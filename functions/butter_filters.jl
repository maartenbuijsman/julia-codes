#= butter_filters.jl
   Maarten Buijsman, USM, 2025-7-30
   This function file includes band, low, and highpass butter filters
=#

using DSP

function bandpass_butter(y,Tl,Th,dt,N)
#= bandpass filter
   y: time series that needs to be filtered
   Tl,Th: low and high cutoff times 
   dt: delta-t
   N: order of the filter
   Make sure all the time varibles have the same time units   
=# 

    f1=1/Th  
    f2=1/Tl 
    fs=1/dt
    b = digitalfilter(Bandpass(f1, f2), Butterworth(N); fs=fs)

    return filtfilt(b, y)
end