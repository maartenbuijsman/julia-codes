pathname = "C:/Users/w944461/Documents/JULIA/functions/"
include(string(pathname,"include_functions.jl"))

using CairoMakie
Threads.nthreads()

# test the filtering
t = 0:1:720

T1=12
T2=24
u(t) = 1*cos(2π/T1*t) + 0.5*cos(2π/T2*t)
uu = u.(t)

fig1 = Figure()
ax = Axis(fig1[1, 1])
lines!(ax,t,uu)
xlims!(ax, (0, 5*24))

fig1

Tl,Th,dt,N = 9,15,1,4
uuf = bandpass_butter(uu,Tl,Th,dt,N)

#limits!(ax, (min_x, max_x, min_y, max_y))

lines!(ax,t,uuf,color = :red)
lines!(ax,t,1*cos.(2π/T1*t),color = :green, linestyle = :dash)
lines!(ax,t,0.5*cos.(2π/T2*t),color = :orange, linestyle = :dash)
xlims!(ax, (0*24, 7*24))
fig1



#=
function bandpass_butter(y,Tl,Th,dt,N)
    f1=1/Th  
    f2=1/Tl 
    fs=1/dt
    b = digitalfilter(Bandpass(f1, f2), Butterworth(N); fs=fs)

    return filtfilt(b, y)
end
=#

#= Design a Butterworth bandpass filter
#f1, f2, fs = 1/30, 1/20, 1/1 # cph
f1, f2, fs = 1/15, 1/9, 1/1 # cph
b = digitalfilter(Bandpass(f1, f2), Butterworth(4); fs=fs)
uuf = filtfilt(b, uu)
=#


#=
fs = 1000 # Sampling frequency in Hz
f_low = 50 # Lower cutoff frequency in Hz
f_high = 200 # Upper cutoff frequency in Hz
filter_order = 4 # Filter order

# Design a Butterworth bandpass filter
%bandpass_filter = digitalfilter(Bandpass(f_low, f_high), Butterworth(filter_order))
bandpass_filter = digitalfilter(Butterworth(filter_order),Bandpass(f_low, f_high; fs=fs))


# Assuming 'signal' is your input data array
filtered_signal = filt(bandpass_filter, signal)


using DSP

# Digital filter
fs = 100
df = digitalfilter(Bandpass(5, 10; fs), Butterworth(2))
G = tf(df, 1/fs)
bodeplot(G, xscale=:identity, yscale=:identity, hz=true)
=#