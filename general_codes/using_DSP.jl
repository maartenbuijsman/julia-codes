pathname = "/home/mbui/Documents/julia-codes/functions/"
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

# create two column vector
uu2 = [uu uu]

Tl,Th,dt,N = 9,15,1,4
uuf = bandpass_butter(uu2,Tl,Th,dt,N)

#limits!(ax, (min_x, max_x, min_y, max_y))

lines!(ax,t,uuf[:,2],color = :red)
lines!(ax,t,1*cos.(2π/T1*t),color = :green, linestyle = :dash)
lines!(ax,t,0.5*cos.(2π/T2*t),color = :orange, linestyle = :dash)
xlims!(ax, (0*24, 7*24))
fig1

# fft ######################

# Test signal
T1, T2, T3 = 0.5, 1, 2  # days
t = collect(1:1447)/24
y = 1.0 .*  cos.(2π .* t ./ T1) .+ 
    0.5 .*  cos.(2π .* t ./ T2) .+ 
    0.25 .* cos.(2π .* t ./ T3)

fig1 = Figure()
ax1 = fig1[1, 1] 
lines(ax1, t, y)
fig1  

tukeycf=0.0
numwin=3
linfit=true
prewhit=false
period, freq, ppp = fft_spectra(t, y; tukeycf, numwin, linfit, prewhit);

# Plot
fig = Figure()
ax = Axis(fig[1, 1], title = "Power Spectrum",
    xlabel = "Frequency [1/unit]", ylabel = "Power",
     yscale = log10)

lines!(ax, freq, power, color = :green, linewidth = 2)

# Mark expected frequencies
expected_freqs = [1/T1, 1/T2, 1/T3]
expected_power = fill(mean(power), 3)
scatter!(ax, expected_freqs, expected_power, color = [:red, :blue, :orange], markersize = 14)
text!(ax, expected_freqs, expected_power .* 1.2;
    text = ["T1=$T1", "T2=$T2", "T3=$T3"],
    align = (:center, :bottom), fontsize = 14)
xlims!(ax, (0, 2.5))
ylims!(ax, (1e-4, 1e2))

fig

###############################################
# test the fft_spectra_vectorized function
# break down its parts

pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\"
include(string(pathname,"include_functions.jl"))

using CairoMakie

# Test signal
T1, T2, T3 = 0.5, 1, 2  # days
t = collect(1:1447)/24
y = 1.0 .*  cos.(2π .* t ./ T1) .+ 
    0.5 .*  cos.(2π .* t ./ T2) .+ 
    0.25 .* cos.(2π .* t ./ T3)

fig1 = Figure()
ax1 = fig1[1, 1] 
lines(ax1, t, y)
fig1  

numwin=3
tukeycf=0
linfit=true
#linfit=false
prewhit=false

# Convert to Float64
t = collect(float.(t))
y = collect(float.(y))

# Ensure even length
if isodd(length(t))
    t = t[1:end-1]
    y = y[1:end-1]
end

println(t[1],"; ",t[end],";",y[1],"; ",y[end])

dt = t[2] - t[1]
nt1 = length(t)

# Compute window size and create overlapping windows
inw = floor(Int, nt1/(numwin + 1))
windows = [(i*inw + 1):(i*inw + 2*inw) for i in 0:(numwin-1)]

# Window length for freq calculation
nt = length(windows[1])
df = 1 / (dt * nt)
freq = (1:nt/2) ./ (dt * nt)
period = 1.0 ./ freq

# Collect all windowed segments into a matrix
# access segments as: segments[1][[1 end]] 
segments = [y[w] for w in windows]

println("begin, end values = ", segments[1][[1 end]])

# Remove linear trend and mean if requested
if linfit
    segments = [begin
#        println("len segment: ", length(seg))
        x = 1:length(seg)
        X = hcat(x, ones(length(x)))
        β = X \ seg
        seg .- X * β
    end for seg in segments];
else
    segments = [begin 
#        println("mean segment: ", mean(seg))        
        seg .- mean(seg) 
    end for seg in segments];
end

# Precompute window function
#  tukey(dims, α; padding=0, zerophase=false)
#  For α == 0, the window is equivalent to a rectangular window. 
#  For α == 1, the window is a Hann window.
base_window = tukey(nt,tukeycf)

# Apply window
segments = [seg .* base_window for seg in segments]

println("begin, end values = ", segments[1][[1 end]])

# Prewhiten if needed
if prewhit
    segments = [begin
        d = diff(seg) ./ dt
        isodd(length(d)) ? d[1:end-1] : d
    end for seg in segments]
end

#= test fft
Y = fft(segments[1]) .* dt
Y[nt÷2:nt÷2+2]
n = length(segments[1])
#Y[1] = 0.0
deleteat!(Y, 1)
P2 = abs2.(Y)
P1 = 2 .* P2[1:n÷2]
prewhit ? (P1 .* df) ./ (freq .^ 2) : P1 .* df
=#

# Compute FFT and power for all windows
power_matrix = map(segments) do seg
    n = length(seg)
    Y = fft(seg) .* dt
    deleteat!(Y, 1)
    P2 = abs2.(Y)
    P1 = 2 .* P2[1:n÷2]
    println("if parseval theorem we get 1: ",sum(seg.^2*dt)/sum(P1*df))
    prewhit ? (P1 .* df) ./ (freq .^ 2) : P1 .* df
end


# Convert to matrix and average
power = reduce(+, power_matrix) ./ numwin

sum(y.^2*dt)/sum(P1*df) 

#return period, freq, power


# as in function
T1, T2, T3 = 0.5, 1, 2  # days
t = collect(1:1447)/24
y = 1.0 .*  cos.(2π .* t ./ T1) .+ 0.5 .*  cos.(2π .* t ./ T2) .+  0.25 .* cos.(2π .* t ./ T3)

fig1 = Figure(); ax1 = fig1[1, 1]; lines(ax1, t, y); fig1  

tukeycf=0.0; numwin=3; linfit=true; prewhit=false;
period, freq, power = fft_spectra(t, y; tukeycf, numwin, linfit, prewhit);

fig2 = Figure(); ax = Axis(fig2[1, 1], title = "Power Spectrum", xlabel = "Frequency [1/unit]", ylabel = "Power", yscale = log10)
lines!(ax, freq, power, color = :green, linewidth = 2)
expected_freqs = [1/T1, 1/T2, 1/T3]; expected_power = fill(mean(power), 3); # Mark expected frequencies
scatter!(ax, expected_freqs, expected_power, color = [:red, :blue, :orange], markersize = 14)
text!(ax, expected_freqs, expected_power .* 1.2; text = ["T1=$T1", "T2=$T2", "T3=$T3"], align = (:center, :bottom), fontsize = 14)
xlims!(ax, (0, 2.5)); ylims!(ax, (1e-4, 1e2)); fig2