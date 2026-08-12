#= IW_analysis_energy_2000km.jl
Maarten Buijsman, USM DMS, 2026-08-07
Load energy and fft results from various sims. and make figures
=#

println("number of threads is ",Threads.nthreads())

using Pkg, NCDatasets, Printf, CairoMakie, Statistics, 
JLD2, ColorSchemes, LaTeXStrings, Interpolations

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
    dirEIG = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirfig = string(pth0,"figs/");
    dirout = string(pth0,"diagout/");
    dirforce = string(pth0,"IW/forcingfiles/");
    dirEIG = string(pth0,"IW/forcingfiles/");
end

include(string(pathname,"include_functions.jl"))

# print figures
figflag = 1

const T2 = 12+25.2/60
const rho0=1020; 
const grav=9.81; 

# run names --------------------------------

# constant N2, NH 4km
#runnms  = collect(3:14)  #AMZ N2
#mainnms = fill(3, size(runnms))
#LATS    = [0, 2.5, 5, 10, 15, 20, 25, 28.8, 30, 35, 40, 50]

# constant N2, NH 4km test run with new forcing
#runnms  = collect(90:92)  #AMZ N2
#mainnms = fill(3, size(runnms))
#LATS    = [0, 20, 60]

#= constant N2, NH 200 m
runnms  = collect(1:14)  #AMZ N2
#runnms  = collect(15:28)  #MERC N2, zonally chnaging
mainnms = fill(9, size(runnms))
LATS    = vcat(collect(0:2.5:5), collect(10:5:60))
=#


# D2 NH flux forcing, 4 km
mainnm  = 10
#runnms  = collect(1:13) # varying  N2 MERCATOR
#runnms  = collect(14:26) # constant N2 MERCATOR 2.5N
runnms  = collect(27:39) # varying  N2 MERCATOR
#runnms  = collect(40:52) # constant N2 MERCATOR 2.5N
LATS    = vcat(collect(0:2.5:5), collect(10:5:25), 28.8, collect(30:5:50))


fnum = string(mainnm,".",runnms[1],"-",runnms[end])

if     mainnm == 3 && runnms[1] == 3; 
    titstr = string("Δx=4 km, const. N(z) (",fnum,")") 
elseif mainnm == 9 && runnms[1] == 1; 
    titstr = string("Δx=200 m, const. N(z) (",fnum,")")    
elseif mainnm == 9 && runnms[1] == 15; 
    titstr = string("Δx=200 m, Mercator N(z) (",fnum,")")    
elseif mainnm == 10 && runnms[1] == 1 || runnms[1] == 27; 
    titstr = string("Δx=4 km, Mercator N(z) (",fnum,")")    
elseif mainnm == 10 && runnms[1] == 14 || runnms[1] == 40; 
    titstr = string("Δx=4 km, const. 2.5N N(z) (",fnum,")")    
end


#= energetics file contents (loaded per run below) --------------------------
dnl, A0nl, alpnl, alpepshy, alpepsnh, Tbeatnh_days, Tbeathy_days, 
epsnh, epshy, epsomnh, epsomhy,
xc, freq, KEoma, KEommax, Fx, Fxt, Fxh, Fxs,    
FAx, FAxt, FAxh, FAxs, FKx, FKxt, FKxh, FKxs,
KE, KEt, KEh, KEs, APE, APEt, APEh, APEs    

# pre-allocate 
=#

# pre-allocate matrices (runs × x)
fnames0  = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
@load string(dirout, "energetics_", fnames0, ".jld2") xc freq
KEr  = zeros(length(runnms), length(xc))   # total KE
KEtr = zeros(length(runnms), length(xc))   # tidal-band KE
KEhr = zeros(length(runnms), length(xc))   # supertidal-band KE
KEOM    = zeros(length(runnms), length(freq))   # KE(ω) frequency spectrum
KEOMmax = zeros(length(runnms))                 # max KE(ω) per run (for later normalization)


for i=1:length(runnms)
    runnm = runnms[i]; LAT = LATS[i];

    # filename
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 
    println(fnames,"; lat=",LAT," -------------------") 

    fnameout = string("energetics_",fnames,".jld2")
    path_fname = string(dirout,fnameout)
    @load path_fname xc KE KEt KEh KEoma KEommax
    KEr[i,:]  = KE;
    KEtr[i,:] = KEt;
    KEhr[i,:] = KEh;
    KEOM[i,:]  = KEoma;
    KEOMmax[i] = KEommax;
end



## heatmaps of KE (total, tidal, supertidal) vs latitude -----------------
LdomH   = 2000e3;
titstrH = strip(replace(replace(titstr, "Δ" => "d"), r"[^A-Za-z0-9._-]+" => "_"), '_')   # filename-safe
cmapKE  = :thermal
fcKE    = 1e-3               # J/m² -> kJ/m² (KE depth-integrated: kg/m³·m²/s²·m = J/m²)

figKE = Figure(size=(700,950))
axKE1 = Axis(figKE[1, 1], title = string("(a) total KE — ",titstr), ylabel = "latitude [°]")
hmKE1 = heatmap!(axKE1, xc/1e3, LATS, KEr'*fcKE, colormap = cmapKE)
Colorbar(figKE[1, 2], hmKE1, label = "KE [kJ/m2]")
axKE1.xticklabelsvisible = false                     # shared x-axis with panels below

axKE2 = Axis(figKE[2, 1], title = "(b) tidal KE", ylabel = "latitude [°]")
hmKE2 = heatmap!(axKE2, xc/1e3, LATS, KEtr'*fcKE, colormap = cmapKE)
Colorbar(figKE[2, 2], hmKE2, label = "KEt [kJ/m2]")
axKE2.xticklabelsvisible = false

axKE3 = Axis(figKE[3, 1], title = "(c) supertidal KE", xlabel = "x [km]", ylabel = "latitude [°]")
hmKE3 = heatmap!(axKE3, xc/1e3, LATS, KEhr'*fcKE, colormap = cmapKE)
Colorbar(figKE[3, 2], hmKE3, label = "KEh [kJ/m2]")

xlims!(axKE1, 0, LdomH/1e3)
xlims!(axKE2, 0, LdomH/1e3)
xlims!(axKE3, 0, LdomH/1e3)
display(figKE)

if figflag==1; save(string(dirfig,"KE_KEt_KEh_heatmap_",titstrH,".png"), figKE)
end


## heatmaps of KE normalized by per-run max tidal KE (M2 input) [%] -----------------
titstrN = strip(replace(replace(titstr, "Δ" => "d"), r"[^A-Za-z0-9._-]+" => "_"), '_')   # filename-safe
cmapKEn = :thermal

Isrc   = findall(xc .<= 100e3)              # source window: first 100 km
KEtmax = maximum(KEtr[:, Isrc], dims=2)     # per-run max tidal KE in source region = M2 input
KErn   = KEr  ./ KEtmax .* 100          # total KE as % of max KEt
KEtrn  = KEtr ./ KEtmax .* 100          # tidal KE as % of max KEt (peaks at 100)
KEhrn  = KEhr ./ KEtmax .* 100          # supertidal KE as % of max KEt

figKEn = Figure(size=(700,950))
axKEn1 = Axis(figKEn[1, 1], title = string("(a) total KE / max(KEt) — ",titstr), ylabel = "latitude [°]")
hmKEn1 = heatmap!(axKEn1, xc/1e3, LATS, KErn', colormap = cmapKEn)
Colorbar(figKEn[1, 2], hmKEn1, label = "KE [%]")
axKEn1.xticklabelsvisible = false                    # shared x-axis with panels below

axKEn2 = Axis(figKEn[2, 1], title = "(b) tidal KE / max(KEt)", ylabel = "latitude [°]")
hmKEn2 = heatmap!(axKEn2, xc/1e3, LATS, KEtrn', colormap = cmapKEn, colorrange = (0, 100))
Colorbar(figKEn[2, 2], hmKEn2, label = "KEt [%]")
axKEn2.xticklabelsvisible = false

axKEn3 = Axis(figKEn[3, 1], title = "(c) supertidal KE / max(KEt)", xlabel = "x [km]", ylabel = "latitude [°]")
hmKEn3 = heatmap!(axKEn3, xc/1e3, LATS, KEhrn', colormap = cmapKEn, colorrange = (0, 100))
Colorbar(figKEn[3, 2], hmKEn3, label = "KEh [%]")

xlims!(axKEn1, 0, LdomH/1e3)
xlims!(axKEn2, 0, LdomH/1e3)
xlims!(axKEn3, 0, LdomH/1e3)
display(figKEn)

if figflag==1; save(string(dirfig,"KE_norm_maxKEt_heatmap_",titstrN,".png"), figKEn)
end


## FFT coefficients: KE(ω) frequency spectra, one line per run ------------------
titstrF = strip(replace(replace(titstr, "Δ" => "d"), r"[^A-Za-z0-9._-]+" => "_"), '_')   # filename-safe
colorsF = cgrad(:darktest, length(LATS), categorical = true)

figF = Figure(size=(700,500))
axF  = Axis(figF[1, 1], xscale = log10, yscale = log10,
    xticks = [1, 2, 4, 6, 8, 12, 24, 48],
    title = string("KE frequency spectra — ",titstr),
    xlabel = "frequency [cpd]", ylabel = "KE(ω)")
for i = 1:length(LATS)
    lw = LATS[i] == 0 ? 4 : 2                               # 0° line twice as thick
    ls = (25 <= LATS[i] <= 30) ? :dash : :solid            # 25°–30° (inclusive) dashed
    lines!(axF, freq, KEOM[i,:], color = colorsF[i], linewidth = lw, linestyle = ls)
end
xlims!(axF, 0.3, 48)
Colorbar(figF[1, 2], colormap = colorsF, limits = (0.5, length(LATS) + 0.5),
    ticks = (1:length(LATS), string.(LATS)), label = "latitude [°]")
display(figF)

if figflag==1; save(string(dirfig,"KEspec_freq_",titstrF,".png"), figF)
end


## heatmap of normalized KE(ω) spectra: latitude vs frequency -------------------
titstrFn = strip(replace(replace(titstr, "Δ" => "d"), r"[^A-Za-z0-9._-]+" => "_"), '_')   # filename-safe
KEOMn = KEOM ./ KEOMmax          # normalize each run by its own max KE(ω) (peaks at 1)
Ifr   = findall(freq .> 0)       # drop 0 cpd for log x-axis
#Zn    = max.((KEOMn[:, Ifr])', 1e-6)   # (freq × lat), floored for log colour scale
Zn    = (KEOMn[:, Ifr])'        # (freq × lat), floored for log colour scale

figFn = Figure(size=(750,500))
axFn  = Axis(figFn[1, 1], xscale = log10,
    xticks = [0.5, 1, 2, 4, 6, 8, 12, 24, 48],
    title = string("normalized KE frequency spectra — ",titstr),
    xlabel = "frequency [cpd]", ylabel = "latitude [°]")
hmFn = heatmap!(axFn, freq[Ifr], LATS, Zn,
    colormap =:thermal, colorscale = log10, colorrange = (1e-5, 1))
#    colormap = cgrad([:black, :purple, :red, :orange, :yellow, :white]), colorscale = log10, colorrange = (1e-6, 1))
Colorbar(figFn[1, 2], hmFn, label = "KE(ω) / max")
xlims!(axFn, 0.3, 48)
display(figFn)

if figflag==1; save(string(dirfig,"KEspec_heatmap_lat_freq_",titstrFn,".png"), figFn)
end


##
return
stop()

## plot energy spectra =======================

mainnms = [6, 7]; #4km-nh, 4k-h, 200m-nh, 200m-k
runnms  = [3, 3];

mainnms = [7, 7]; #200m-nh, weak, 200m-nh, strong
runnms  = [3, 15];

LAT = 0.0;

fnum1 = string(mainnms[1],".",runnms[1])
fnum2 = string(mainnms[2],".",runnms[2])

titstra = string("4-km hyd (",fnum1,")")
titstrb = string("200-m nonhyd (",fnum2,")")    

#= load energetics_AMZexpt06.01.jld2" data --------------------------
dnl, A0nl, alpnl, alpepshy, alpepsnh, Tbeatnh_days, Tbeathy_days, 
epsnh, epshy, epsomnh, epsomhy,
xc, freq, KEoma, KEommax, Fx, Fxt, Fxh, Fxs,    
FAx, FAxt, FAxh, FAxs, FKx, FKxt, FKxh, FKxs,
KE, KEt, KEh, KEs, APE, APEt, APEh, APEs    
=#

# pre-allocate 
fnames0  = @sprintf("AMZexpt%02i.%02i", mainnms[1], runnms[1])
@load string(dirout, "energetics_", fnames0, ".jld2") xc freq

KEOM = zeros(length(runnms), length(freq))
KEOMmax = zeros(length(runnms))

for i=1:length(runnms)
    mainnm = mainnms[i]; runnm = runnms[i]; 

    # filename
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 
    println(fnames," -------------------") 

    fnameout = string("energetics_",fnames,".jld2")
    path_fname = string(dirout,fnameout)
    @load path_fname KEoma KEommax
    KEOM[i,:] = KEoma;
    KEOMmax[i] = KEommax;
end


flim2 = [1 48]; 
Plims = [-14 1];

fig1 = Figure(size=(500,500))
ax1 = Axis(fig1[1, 1], xticks = [1, 2, 4, 6, 8, 12, 24, 48], xscale = log10, yscale = log10,
    title="transect-mean normalized spectral KE",ylabel="KE/KEmax");  
xlims!(ax1, flim2[1], flim2[2])
ylims!(ax1, 10.0^Plims[1], 10.0^Plims[2])
lines!(ax1, freq, KEOM[1,:]./KEOMmax[1], label = titstra, linestyle=:solid, color = :black, linewidth = 3)
lines!(ax1, freq, KEOM[2,:]./KEOMmax[2], label = titstrb, linestyle=:solid, color = :red, linewidth = 2)
axislegend(ax1, position = :lb)

# add coriolis rad/s => cpd
#fcpd = coriolis(LAT)/(2*pi)*24*3600
#lines!(vec([fcpd fcpd]),vec([minimum(log10.(KEoma)) maximum(log10.(KEoma))]), linestyle=:dash, color = :red, linewidth = 2)
#lines!(axb, vec([fcpd fcpd]), [10.0^Plims[1], 10.0^Plims[2]], linestyle=:dash, color = :red, linewidth = 2)

ax2 = Axis(fig1[2, 1], xticks = [1, 2, 4, 6, 8, 12, 24, 48], xscale = log10, 
    title=string(titstra," - ",titstrb),xlabel="frequency [cpd]",ylabel="ΔKE [m2/s2 1/cpd]")
xlims!(ax2, flim2[1], flim2[2])
#lines!(ax2,freq, KEOM[1,:]./KEOMmax[1] .- KEOM[2,:]./KEOMmax[2], linestyle=:solid, color = :blue, linewidth = 3)
lines!(ax2,freq, KEOM[1,:] .- KEOM[2,:], linestyle=:solid, color = :blue, linewidth = 2)

rowsize!(fig1.layout, 1, Relative(2/3))
rowsize!(fig1.layout, 2, Relative(1/3))

display(fig1)


# Save the figure as a PNG file
titstrc = replace(string(titstra,"_",titstrb), " " => "_", "(" => "", ")" => "")

if figflag==1; save(string(dirfig,"fft_KEsur_",titstrc,".png"), fig1)
end


##


stop()

##############################################
##############################################
##############################################

#= smooth some data ????
using Smoothers
NP = 200 # number of points to average over dx=200 m
NP = 10 #200*200/4000 for 4 km
Πxxa2 = convert(Vector{Float64}, coalesce.(sma(Πxxa, NP, true), 0.0))
Πzxa2 = convert(Vector{Float64}, coalesce.(sma(Πzxa, NP, true), 0.0))
Πnhxa2 = convert(Vector{Float64}, coalesce.(sma(Πnhxa, NP, true), 0.0))

xlim = 500

#fig = Figure(); 
fig = Figure(size = (600, 250));
 ax2 = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]", title=string(fname_short2,"; ",titlenm2))
lines!(ax2,xc/1e3,Πxxa2,color=:red, label="Πx")
lines!(ax2,xc/1e3,Πzxa2,color=:green, label="Πz")
lines!(ax2,xc/1e3,Πnhxa2,color=:black, label="Πnh")
lines!(ax2,xc/1e3,Πnhxa2+Πzxa2+Πxxa2,color=:orange, label="sum", linewidth = 3)
axislegend(ax2,position = :rt; framevisible = false )
xlims!(ax2, 0, xlim)
ylims!(ax2, -0.6e-4, 2.2e-4)
fig
=#

#= test
using Smoothers

# Generate some noisy data
x = 1:100
y = sin.(x .* 0.1) .+ randn(100) .* 0.5

# Apply LOESS smoothing
# The 'span' parameter (0.2 in this case) controls the degree of smoothing
#smoothed_y = loess(y, span=0.2)
ysmooth = sma(y, 10, true)
#Igood = findall(!ismissing, smoothed_y)
ysmooth2 = convert(Vector{Float64}, coalesce.(ysmooth, 0.0))


# Plot the original and smoothed data
fig = Figure(); 
ax = Axis(fig[1, 1])
lines!(ax,x, y)
lines!(ax,x, ysmooth2)
fig
=#

## load and compare the CG transects =======================================

#=
fnamal = ["AMZ1_lat0_8d_U1_0.25_U2_0.00",  # mode 1
          "AMZ1_lat0_8d_U1_0.00_U2_0.20",  # mode 2
          "AMZ1_lat0_8d_U1_0.25_U2_0.20"]  # mode 1+2
          =#

#fnamal = ["AMZ2_lat0_12d_U1_0.50_U2_0.00",  # mode 1
#          "AMZ2_lat0_12d_U1_0.00_U2_0.40",  # mode 2
#          "AMZ2_lat0_12d_U1_0.50_U2_0.40"]  # mode 1+2

fnamal = ["AMZ3_hvis_12d_U1_0.40_U2_0.00",  # mode 1
          "AMZ3_hvis_12d_U1_0.00_U2_0.30",  # mode 2
          "AMZ3_hvis_12d_U1_0.40_U2_0.30"]  # mode 1+2


# load simulations
Πsum = 0;
xc=0;  Πnhxa=0;  Πzxa=0;  Πxxa=0;
for i in 1:2
    path_fname = string(dirout,"Etran_",fnamal[i],".jld2")

    @load path_fname xc  Πnhxa  Πzxa  Πxxa    
    Πsum = Πsum .+ Πnhxa .+ Πzxa .+ Πxxa
end

# old stuff? new stuff is below
# Open the JLD2 file
path_fname = string(dirout,"Etran_",fnamal[3],".jld2");

fff = jldopen(path_fname, "r")
println(keys(fff))  # List the keys (variables) in the file
close(fff)

@load path_fname xc  Πnhxa  Πzxa  Πxxa    
Πsum2 = Πnhxa .+ Πzxa .+ Πxxa


fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]")
lines!(ax,xc/1e3,Πsum,color=:red, linewidth = 2, label="sim. mode 1 + sim. mode 2")
lines!(ax,xc/1e3,Πsum2,color=:green, linewidth = 2, label="sim. mode 1+2")
axislegend(position = :rb)
fig


# cumulative sum
Πcumsum  = cumtrapz(xc,Πsum);
Πcumsum2 = cumtrapz(xc,Πsum2);

fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Σ Π [W/kg*m2]", 
title = "cumulative tidal to supertidal energy transfer")
lines!(ax,xc/1e3,Πcumsum,color=:red, linewidth = 3, label="sim. mode 1 + sim. mode 2")
lines!(ax,xc/1e3,Πcumsum2,color=:green, linewidth = 3, label="sim. mode 1+2", linestyle = :dash)
axislegend(position = :rb)
xlims!(ax, 0, 500)
fig

if figflag==1; save(string(dirfig,"PI_cumsum.png"), fig)
end

#=
# test

Πsum2 = Πnhxa .+ Πzxa .+ Πxxa


fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]")
lines!(ax,xc/1e3,Πsum2,color=:green, linewidth = 2, label="sim. mode 1")
axislegend(position = :rb)
fig


# cumulative sum
Πcumsum2 = cumtrapz(xc,Πsum2);

fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Σ Π [W/kg*m2]", 
title = "cumulative tidal to supertidal energy transfer")
lines!(ax,xc/1e3,Πcumsum2,color=:green, linewidth = 3, label="sim. mode 1+2", linestyle = :dash)
axislegend(position = :rb)
xlims!(ax, 0, 500)
fig

stop()
=#

## load all latitudes and plot the cumsum ======================================
#runnms = [38 39 40 41 42 43 44 45 46 47 48    49];
#LATS =   [0 2.5 5  10 15 20 25 30 40 50 28.80 35];
runsel = [38 39 40 41 42 43 44 48 45 49 46 47];

# get xc
mainnm = 1
fnames = @sprintf("Etran_AMZexpt%02i.%02i.jld2",mainnm,runsel[1]) 
path_fname = string(dirout,fnames)

@load path_fname xc

DX = xc[2] - xc[1]

# use data away from forcing and sponges
#xlims = [75,500]*1e3;
xlims = [0,700]*1e3;
Ix = findall(item -> item >= xlims[1] && item<= xlims[2], xc);

Πtran = zeros(length(runsel),length(xc));
Πmin  = zeros(length(runsel));
Πmax  = zeros(length(runsel));
Πsum  = zeros(length(runsel));
LATSS = zeros(length(runsel));
for i in 1:length(runsel)

    fnames = @sprintf("Etran_AMZexpt%02i.%02i.jld2",mainnm,runsel[i]) 
    path_fname = string(dirout,fnames)

    @load path_fname Πnhxa Πzxa Πxxa LAT
    LATSS[i] = LAT
    Πtran[i,:] = Πnhxa .+ Πzxa .+ Πxxa

    Πsum[i] = sum(Πtran[i,Ix]*DX)/sum(length(Ix)*DX)
    Πmin[i],Πmax[i] = extrema(Πtran[i,Ix])
    #Πmax[i] = max(Πtran[i,Ix])
end

fig1 = Figure()
axa = Axis(fig1[1,1], title="Π for τ=9 hr",xlabel="Π [W/kg]",ylabel="latitude [°N]");  
scatterlines!(axa,Πsum,LATSS, linestyle=:solid, color = :black, linewidth=3,label="mean")
scatterlines!(axa,Πmax,LATSS, linestyle=:dash, color = :red, linewidth=3,label="max")
scatterlines!(axa,Πmin,LATSS, linestyle=:dash, color = :deepskyblue, linewidth=3,label="min")
axislegend(axa, position = :rt)
fig1


if figflag==1; save(string(dirfig,"max_min_ave_PI_lat.png"), fig1)
end