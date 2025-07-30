#= butter_filters.jl
   Maarten Buijsman, USM, 2025-7-30
   This function file includes band, low, and highpass butter filters
=#

using DSP

"""
bandpass_butter(y,Tl,Th,dt,N)

bandpass butterworth filter, returns filtered time series

# Arguments
y: time series that needs to be filtered.

Tl, Th: low and high cutoff periods. 

dt: delta-t.

N: order of the filter.

Make sure all the time variables have the same time units!   

# Example: 9-15 hour bandpass filter for 1 cph sampling (Δt=1hour)
yf = bandpass_butter(y,9,15,1,4)

"""
function bandpass_butter(y,Tl,Th,dt,N)
    f1=1/Th  
    f2=1/Tl 
    fs=1/dt
    b = digitalfilter(Bandpass(f1, f2), Butterworth(N); fs=fs)

    return filtfilt(b, y)
end