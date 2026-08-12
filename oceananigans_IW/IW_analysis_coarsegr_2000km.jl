#= IW_analysis_coarsegr.jl
Maarten Buijsman, USM DMS, 2026-08-10
Load coarse graining results from various sims. and make figures
=#

# TO DO:
# normalize CGE/KE!!

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

#= constant N2, NH 200 m
#runnms  = collect(1:14)  #AMZ N2
runnms  = collect(15:28)  #MERC N2, zonally chnaging
mainnms = fill(9, size(runnms))
LATS    = vcat(collect(0:2.5:5), collect(10:5:60))
=#

# D2 NH flux forcing, 4 km
mainnm  = 10
#runnms  = collect(1:14) # constant N2 WOCE AMZ
runnms  = collect(1:13) # varying  N2 MERCATOR        F=15kW/m
#runnms  = collect(14:26) # constant N2 MERCATOR 2.5N  F=15kW/m
runnms  = collect(27:39) # varying  N2 MERCATOR       F=25kW/m
runnms  = collect(40:52) # constant N2 MERCATOR 2.5N  F=25kW/m
#runnms  = collect(53:65) # constant N2 MERCATOR 50N   F=25kW/m
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
elseif mainnm == 10 && runnms[1] == 53; 
    titstr = string("Δx=4 km, const. 50N N(z) (",fnum,")")        
end


#= load energetics_AMZexpt06.01.jld2" data --------------------------
dnl, A0nl, alpnl, alpepshy, alpepsnh, Tbeatnh_days, Tbeathy_days, 
epsnh, epshy, epsomnh, epsomhy,
xc, freq, KEoma, KEommax, Fx, Fxt, Fxh, Fxs,    
FAx, FAxt, FAxh, FAxs, FKx, FKxt, FKxh, FKxs,
KE, KEt, KEh, KEs, APE, APEt, APEh, APEs    


# pre-allocate 
fnames0  = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
@load string(dirout, "energetics_", fnames0, ".jld2") xc
FXT = zeros(length(runnms), length(xc))
FXH = zeros(length(runnms), length(xc))

for i=1:length(runnms)
    mainnm = mainnms[i]; runnm = runnms[i]; LAT = LATS[i];

    # filename
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 
    println(fnames,"; lat=",LAT," -------------------") 

    fnameout = string("energetics_",fnames,".jld2")
    path_fname = string(dirout,fnameout)
    @load path_fname xc Fxt Fxh 
    FXT[i,:] = Fxt;
    FXH[i,:] = Fxh;
end
=#


## load cross-scale transfers Etran_AMZexptX.X.jld2" data --------------------------
#= LAT, xc, zc, 
Πnhxa, Πxxa, Πzxa (depend on x), 
Πnhza, Πxza, Πzza (depend on z),
Πxztot (depends on x,z)
=#

# pre-allocate 
fnames0  = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
@load string(dirout, "Etran_", fnames0, ".jld2") xc zc
CGE = zeros(length(runnms), length(xc))
CGEsum = copy(CGE)
Πxztot = nothing
dx = xc[2]-xc[1]
Lsmooth = 1600  # Gaussian σ [m] Lsmooth = 400 ≈ 2 grid cells; kills 2Δx grid noise, keeps ≳1 km structure

for i=1:length(runnms)
    runnm = runnms[i]; LAT = LATS[i];

    # filename
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 
    println(fnames,"; lat=",LAT," -------------------") 

    fnameout = string("Etran_",fnames,".jld2")
    path_fname = string(dirout,fnameout)
    @load path_fname Πnhxa  Πxxa  Πzxa
    pietot = Πnhxa+Πxxa+Πzxa;
    CGE[i,:] = copy(pietot)

    if dx < 500

        # Gaussian smoothing on physical scale Lsmooth (edge-renormalised)
        pietot = gaussfilt(xc, pietot, Lsmooth)

        CGE[i,:] = pietot
    end

    # cumtrapz 
    CGEsum[i,:] = cumtrapz(xc,CGE[i,:]) 

    if i==99; 
        global Πxztot  #xc, zc
        @load path_fname Πxztot; 

        if dx < 500

            # Gaussian smoothing in x, applied to each depth column
            for kz in 1:size(Πxztot, 2)
                Πxztot[:, kz] = gaussfilt(xc, Πxztot[:, kz], Lsmooth)
            end
        end

    end
end


## heatmaps of CGE and its cumulative integral CGEsum vs latitude -----------------
fcH     = 1e5;                                       # scale Π to 1e4 W/(kg m), as in panel (c)
fc5H    = 1;                                          # CGEsum already in W/(kg m^2), as in panel (d)
LdomH   = 2000e3;
titstrH = strip(replace(replace(titstr, "Δ" => "d"), r"[^A-Za-z0-9._-]+" => "_"), '_')   # filename-safe
#cmaxH   = maximum(abs.(CGE))*fcH                     # symmetric range about 0
cmaxH   = 1                    # symmetric range about 0
#cmaxsH  = maximum(abs.(CGEsum))*fc5H                  # symmetric range about 0
cmaxsH  = 10*fc5H 

figCGE = Figure(size=(700,850))
axCGE  = Axis(figCGE[1, 1], title = string("(a) cross-scale energy transfer — ",titstr),
    ylabel = "latitude [°]")
hmCGE  = heatmap!(axCGE, xc/1e3, LATS, CGE'*fcH,
    colormap = Reverse(:RdBu_5), colorrange = (-cmaxH, cmaxH))
Colorbar(figCGE[1, 2], hmCGE, label = @sprintf("Π [%.0e W/kg m]", 1/fcH))
axCGE.xticklabelsvisible = false                     # shared x-axis with panel below

axCGEs = Axis(figCGE[2, 1], title = "(b) cumulative cross-scale energy transfer",
    xlabel = "x [km]", ylabel = "latitude [°]")
hmCGEs = heatmap!(axCGEs, xc/1e3, LATS, CGEsum'*fc5H,
    colormap = Reverse(:RdBu_5), colorrange = (-cmaxsH, cmaxsH))
Colorbar(figCGE[2, 2], hmCGEs, label = "Σ Π dx [W/kg m2]")

xlims!(axCGE,  0, LdomH/1e3)
xlims!(axCGEs, 0, LdomH/1e3)
display(figCGE)

if figflag==1; save(string(dirfig,"CGE_CGEsum_heatmap_",titstrH,".png"), figCGE)
end

println("CGE min/max = ", @sprintf("%.2e", minimum(CGE)), " / ", @sprintf("%.2e", maximum(CGE)))

## line plots of CGE and cumulative CGEsum vs x, one line per latitude ----------
fcC     = 1e4;                                          # scale Π to 1e4 W/(kg m), as in panel (c)
LdomC   = 2000e3;
titstrC = strip(replace(replace(titstr, "Δ" => "d"), r"[^A-Za-z0-9._-]+" => "_"), '_')   # filename-safe
colorsC = cgrad(:darktest, length(LATS), categorical = true)   # distinct color per run

figCL = Figure(size=(700,800))
axCL1 = Axis(figCL[1, 1], title = string("(a) cross-scale energy transfer — ",titstr),
    ylabel = "Π [1e4 W/kg m]")
axCL2 = Axis(figCL[2, 1], title = "(b) cumulative cross-scale energy transfer",
    xlabel = "x [km]", ylabel = "Σ Π dx [W/kg m2]")
for i = 1:length(LATS)
    lw = LATS[i] == 0 ? 4 : 2                               # 0° line twice as thick
    ls = (25 <= LATS[i] <= 30) ? :dash : :solid            # 25°–30° (inclusive) dashed
    lines!(axCL1, xc/1e3, CGE[i,:]*fcC, color = colorsC[i], linewidth = lw, linestyle = ls)
    lines!(axCL2, xc/1e3, CGEsum[i,:],  color = colorsC[i], linewidth = lw, linestyle = ls)
end
axCL1.xticklabelsvisible = false                           # shared x-axis with panel below
xlims!(axCL1, 0, LdomC/1e3)
xlims!(axCL2, 0, LdomC/1e3)
Colorbar(figCL[1:2, 2], colormap = colorsC, limits = (0.5, length(LATS) + 0.5),
    ticks = (1:length(LATS), string.(LATS)), label = "latitude [°]")
display(figCL)

if figflag==1; save(string(dirfig,"CGE_CGEsum_lines_lat_",titstrC,".png"), figCL)
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