# testing_sturmL.jl
# MCB, USM, 2025-8-8
# testing this chatgpt concoction 

#pathname = "C://Users//w944461//Documents//JULIA//functions/";
pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using CairoMakie
using NCDatasets
using GibbsSeaWater
using Interpolations
using Trapz

Threads.nthreads()

# load Gregg Jacobs' data set
filename = "/home/jacobs/Projects/WOD/amazon/cov_amazon_080_01.nc"
ds = NCDataset(filename,"r");

depth = ds["depth"][:];
double_depth1 = ds["double_depth1"][:];
latitude = ds["latitude"][:];
longitude = ds["longitude"][:];
TSmean = ds["mean"];

Nz = length(depth);

LON, LAT = meshgrid(longitude,latitude);

#=
fig = Figure()
ax = Axis(fig[1,1])
scatter!(LON[:],LAT[:])
scatter!(ax,LON[is,js],LAT[is,js], marker = '*', markersize = 50)
fig
=#

lonsel = 360-42; latsel = 7    # deep cast 4000 m
#lonsel = 360-45; latsel = 3   # shallow cast 2000 m
d,is = findmin(abs.(longitude.-lonsel));
d,js = findmin(abs.(latitude.-latsel));

TSsel = TSmean[:,is,js];
Ts = TSsel[1:Nz];
Ss = TSsel[Nz+1:end];
depths = ds["depth"][:]; 

# check for bad values and remove them
threshold = 1e4;
ibad = [i for i in eachindex(Ts) if Ts[i] > threshold]
if ~isempty(ibad)
    deleteat!(Ts, ibad)
    deleteat!(Ss, ibad)
    deleteat!(depths, ibad)
end

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Ts,-depths)

ax2 = Axis(fig[1,2])
lines!(ax2,Ss,-depths)
fig

Nz2 = length(depths);

# compute N2 using TEOS10
p  = gsw_p_from_z.(-depths,latsel)
SA = gsw_sa_from_sp.(Ss, p, lonsel, latsel)
CT = gsw_ct_from_t.(SA,Ts,p)
sig2 = gsw_sigma2.(SA,CT)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,CT,-p)

ax2 = Axis(fig[1,2])
lines!(ax2,SA,-p)
fig

N2   = zeros(Nz2-1) # Array for output of Brunt-Vaisalla values @ Pmid) 
Pmid = zeros(Nz2-1) # Array for outputs of pressure for N2
Lats = fill(latsel, Nz2) 
gsw_nsquared(SA, CT, p, Lats, Nz2, N2, Pmid)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,sig2,-p)
ylims!(ax1, -100, 0)

ax2 = Axis(fig[1,2])
lines!(ax2,N2,-Pmid)
ylims!(ax2, -100, 0)
fig

# find bad data points
N2b = N2
ibad = [i for i in eachindex(N2b) if N2b[i] < 0]
ii = ibad[end]
N2b[1:ii] .= 0.0

lines!(ax2,N2b,-Pmid,color=:red)
fig

# new vectors for EVP
itp = interpolate((p,), -depths, Gridded(Linear()))
zmid = itp.(Pmid)

# range is < mindepth
mindepth = -4000;
Iz = findall(item -> item > mindepth, zmid) 

# create final vectors
zz  = [0; zmid[Iz]; mindepth]
N2c = [0; N2b[Iz]; N2b[Iz[end]]] 

zeroval = 1e-12
Iz = findall(item -> item ==0, N2c) 
N2c[Iz] .= zeroval 

"""
    sturm_liouville_noneqDZ_norm(zf::Vector{Float64}, N2::Vector{Float64}, f::Float64, om::Float64, nonhyd::Int)
# Arguments    
zf: layer faces [m], can either surface to bottom (e.g., 0 to -H) or bottom to surface,
N2: Brunt-Väisälä frequency squared [rad^2/s^2] at layer faces zf,
f: Coriolis frequency [rad/s],
om: internal wave frequency [rad/s],
nonhyd: if -1, solve the non-hydrostatic Sturm-Liouville problem
"""

# make sure all values are from bottom to surface
# to comply with Oceananigans

zf = zz;
N2 = N2c;
Nzf = length(zf);

flipped = zf[1] < zf[end]  # if false, input is surface to bottom

# !flipped means if flipped is true, then !flipped is not true
if !flipped
    zf = reverse(zf)
    N2 = reverse(N2)
end

zc = zf[1:end-1]/2 + zf[2:end]/2;

# eigen value problemn ===================================== 
nonhyd = 1;
om = 2*π/(12.4*3600)
lat = 0
f = coriolis(lat)

k, L, C, Cg, Ce, Weig, Ueig, Ueig2 = 
    sturm_liouville_noneqDZ_norm(zf, N2, f, om, nonhyd);


Imod = 2;
fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Ueig[:,Imod],zc)
ylims!(ax1, mindepth, 0)

ax2 = Axis(fig[1,2])
lines!(ax2,Weig[:,Imod],zf)
ylims!(ax2, mindepth, 0)
fig

#lines(W2[:,1],zf)
#lines(Ueig2[:,1],zc)

# WKB scale z 
# pick a constant WKB scaled dz and convert it back
# this will used for getting a scaled dz

# depth-mean N 
H = abs(zf[1]);
Nave = trapz(zf,sqrt.(N2))/H

# WKB scaled z
# integrate from surface to bottom
zwkb = zeros(size(zf));
for i in Nzf-1:-1:1
    # reverse integrate, hence omit -
    zwkb[i] = trapz(zf[Nzf:-1:i],sqrt.(N2[Nzf:-1:i]))/Nave
    #println(zf[i])
end

dzkwb = diff(zwkb)
# scatter(zwkb,zf)
scatter(dzkwb,zc)

# now interpolate equidistant dzwkb
dzwkb2 = 40;
zwkb2 = collect(-H:dzwkb2:0)
#zwkb2 = vcat(collect(-H:dzwkb2:-40), collect(-35:5:0))

# extract new z values
#itz = interpolate((zwkb[end:-1:1],), zf[end:-1:1], Gridded(Linear()))
#zfw = itz.(zwkb2[2:end-1])

interp_linextr = linear_interpolation(zwkb, zf, extrapolation_bc=Line())
zfd = interp_linextr.(zwkb2)

zcd = zfd[1:end-1]/2 + zfd[2:end]/2;
dzd = diff(zfd)

# fix the dz near the surface
dzmin, Imin = findmin(dzd)
zfdadd = collect(range(zfd[Imin],0,length=Int(ceil(abs(zfd[Imin])/dzmin))))

zfw = vcat(zfd[1:Imin],zfdadd[2:end])
zcw = zfw[1:end-1]/2 + zfw[2:end]/2;
dzw = diff(zfw)

sum(dzw)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,zwkb, zf)
scatter!(ax1,zwkb2,zfd,color=:red)
#ylims!(ax1, -500, 0)
#xlims!(ax1, -2000, 10)
fig

fig = Figure()
ax1 = Axis(fig[1,1])
scatter!(dzw,zcw)
ylims!(ax1, -500, 0)
xlims!(ax1, 0, 40)
fig

# interpolate the eigenfunctions to zcw
# make sure the depth-mean = 0 for Ueig

Ueig2c = zeros(length(zcw),2);
Ueigc  = zeros(length(zcw),2);  #not normalized
Weigf  = zeros(length(zfw),2);  #not normalized 
Weigc = zeros(length(zcw),2);  #not normalized 

# loop over 2 modes
for i in 1:2
    intzc = linear_interpolation(zc, Ueig2[:,i], extrapolation_bc=Line())
    Ueig2c[:,i] = intzc.(zcw)

    # remove bias due to interpolation and rescale
    bias = sum(Ueig2c[:,i].*dzw)/H
    Ueig2c[:,i] = Ueig2c[:,i].-bias
    #sum(Ueig2c[:,i].*dzw)

    norm_factor = sqrt.(sum(Ueig2c[:,i].^2 .* dzw, dims=1) ./ H)
    Ueig2c[:,i] = Ueig2c[:,i]./norm_factor

    intzc = linear_interpolation(zc, Ueig[:,i], extrapolation_bc=Line())
    Ueigc[:,i] = intzc.(zcw)

    # for un-norm, only remove bias
    bias = sum(Ueigc[:,i].*dzw)/H
    Ueigc[:,i] = Ueigc[:,i].-bias
    #sum(Ueigc[:,i].*dzw)

    intzc = linear_interpolation(zf, Weig[:,i], extrapolation_bc=Line())
    Weigf[:,i] = intzc.(zfw)

    intzc = linear_interpolation(zf, Weig[:,i], extrapolation_bc=Line())
    Weigc[:,i] = intzc.(zcw)

end
mean(Ueig2c[:,2]./Ueigc[:,2])
sum((Ueig2c[:,1].^2).*dzw)
sum(Ueig2c[:,2].*dzw)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Ueig2[:,2],zc)
scatter!(ax1,Ueig2c[:,2],zcw,color=:red)
#ylims!(ax1, -500, 0)
#xlims!(ax1, 0, 40)
fig

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Weig[:,1],zf)
scatter!(ax1,Weigc[:,1],zcw,color=:red)
#ylims!(ax1, -500, 0)
#xlims!(ax1, 0, 40)
fig

# save these eigen functions (Ueigc @ c, Ueig2c @ c, Weigf @ f, Weigc) and dzw, zcw, and zfw
# then load in Oceananigans
# first use Ueigc

dirout = "/data3/mbui/ModelOutput/IW/forcingfiles/"
fnameAZ = "EIG_grid_amz1.jld2"

using JLD2
jldsave(string(dirout,fnameAZ); Ueigc, Ueig2c, Weigf, Weigc, dzw, zcw, zfw)

#using HDF5
#h5write(string(dirout,fnameAZ), )

# Open the JLD2 file
file = jldopen(string(dirout,fnameAZ), "r")

# List the keys (variables) in the file
println(keys(file))
data = file["dzw"]

# all EIGEN function testing is below ======================================

    flipped = zf[1] > zf[end]  # if true, input is surface to bottom
    
    if !flipped
        zf = reverse(zf)
        N2 = reverse(N2)
    end

    dz = -diff(zf)
    H = sum(dz)
    N = length(dz)

    # Handle hydrostatic / nonhydrostatic modes
    if nonhyd == 1
        NN = clamp.(N2 .- om^2, zeroval, Inf)
    else
        NN = clamp.(N2, zeroval, Inf)
    end

    # Construct B matrix
    B = Diagonal(-NN[2:end-1])

    # Construct A matrix
    A = zeros(N - 1, N - 1)
    A[1,1] = -2 / (dz[1] * dz[2])
    A[1,2] = 2 / (dz[2] * (dz[1] + dz[2]))

    for i in 2:N-2
        A[i,i-1] =  2 / (dz[i] * (dz[i] + dz[i+1]))
        A[i,i]   = -2 / (dz[i] * dz[i+1])
        A[i,i+1] =  2 / (dz[i+1] * (dz[i] + dz[i+1]))
    end

    i = N - 1
    A[i,i-1] =  2 / (dz[i] * (dz[i] + dz[i+1]))
    A[i,i]   = -2 / (dz[i] * dz[i+1])

    # Solve eigenvalue problem
    invCe2, W1 = eigen(A, B)
    Ce2 = 1 ./ invCe2
    Ce = sqrt.(Ce2)
    idx = sortperm(Ce, rev=true)
    Ce = Ce[idx]
    W1 = W1[:, idx]

    lines(W1[:,2],zf[2:end-1])


    k = abs.(sqrt(om^2 - f^2) ./ Ce)
    C = om ./ k
    L = 2π ./ k
    Cg = Ce.^2 .* k ./ om

    # Compute vertical structure functions (Weig: faces)
    W2 = vcat(zeros(1, size(W1,2)), W1, zeros(1, size(W1,2)))

    # Compute horizontal eigenfunctions (Ueig: centers)
    dW2 = W2[2:end, :] .- W2[1:end-1, :]
    dzu = repeat(dz, 1, size(W1,2))
    Ueig = dW2 ./ dzu

    zc = zf[1:end-1]/2 + zf[2:end]/2
    lines(Ueig[:,1],zc)
    sum(Ueig[:,2].*dz)

    norm_factor = sqrt.(sum(Ueig.^2 .* dzu, dims=1) ./ H)
    norm_factor[norm_factor .== 0] .= Inf
    Ueig2 = Ueig ./ norm_factor

        lines(Ueig2[:,2],zc)


    # set the correct sign
    for i in 1:size(Ueig2,2)
        if Ueig2[end,i] < 0
            Ueig[:,i] .*= -1
            Ueig2[:,i] .*= -1            
        end
    end

    # Reverse output structure functions if input was reversed
    if !flipped
        W2    = reverse(W2, dims=1)
        Ueig = reverse(Ueig, dims=1)
        Ueig2 = reverse(Ueig2, dims=1)        
    end







##-------------------------------------------------------------------------

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