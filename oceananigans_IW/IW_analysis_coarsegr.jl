#= IW_analysis_coarsegr.jl
Maarten Buijsman, USM DMS, 2026-6-8
Load coarse graining results from various sims. and make figures
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

# only select data after the spinup time 
# => i.e., time it takes before waves reach the eastern boundary
const tspin = 4; #days

# run names --------------------------------
# D2, mode 1 + 2 interactions
#LATS    = [0, 0, 0];
#mainnms = [6, 6, 6]; #4km-nh
#runnms  = [1, 2, 3]; #centered v=1e-2
#runnms  = [4, 5, 6]; #weno v=1e-2 => too diffusive

#mainnms = [7, 7, 7]; #200m-nh
#runnms  = [1, 2, 3]; #centered v=1e-2
#runnms  = [4, 5, 6]; #weno v=1e-5 
#runnms  = [13, 14, 15]; #weno v=1e-5 

# constant N2 
runnms  = collect(1:14)
mainnms = fill(9, size(runnms))
LATS    = vcat(collect(0:2.5:5), collect(10:5:60))

fnum = string(mainnms[1],".",runnms[1],"-",runnms[end])

if     mainnms[1] == 6; titstr = string("4-km hydrostatic (",fnum,")")
elseif mainnms[1] == 7; titstr = string("200 m non-hydrostatic (",fnum,")")   
elseif mainnms[1] == 9 && runnms[1] == 1; 
    titstr = string("Const. N(z) (",fnum,")")    
end

#= load energetics_AMZexpt06.01.jld2" data --------------------------
dnl, A0nl, alpnl, alpepshy, alpepsnh, Tbeatnh_days, Tbeathy_days, 
epsnh, epshy, epsomnh, epsomhy,
xc, freq, KEoma, KEommax, Fx, Fxt, Fxh, Fxs,    
FAx, FAxt, FAxh, FAxs, FKx, FKxt, FKxh, FKxs,
KE, KEt, KEh, KEs, APE, APEt, APEh, APEs    
=#

# pre-allocate 
fnames0  = @sprintf("AMZexpt%02i.%02i", mainnms[1], runnms[1])
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

#= load cross-scale transfers Etran_AMZexptX.X.jld2" data --------------------------
LAT, xc, zc, 
Πnhxa, Πxxa, Πzxa (depend on x), 
Πnhza, Πxza, Πzza (depend on z),
Πxztot (depends on x,z)
=#

# pre-allocate 
@load string(dirout, "Etran_", fnames0, ".jld2") zc
CGE = zeros(length(runnms), length(xc))
CGEsum = copy(CGE)
Πxztot = nothing
dx = xc[2]-xc[1]

for i=1:length(runnms)
    mainnm = mainnms[i]; runnm = runnms[i]; LAT = LATS[i];

    # filename
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 
    println(fnames,"; lat=",LAT," -------------------") 

    fnameout = string("Etran_",fnames,".jld2")
    path_fname = string(dirout,fnameout)
    @load path_fname Πnhxa  Πxxa  Πzxa
    pietot = Πnhxa+Πxxa+Πzxa;
    CGE[i,:] = copy(pietot)

    if dx < 500
        # 3-pnt ave is enough to remove noise!
#        pietot[2:end-1]  = 1/3*pietot[1:end-2] +
#         1/3*pietot[2:end-1] + 1/3*pietot[3:end]; #best

        pietot[3:end-2]  = 1/5*pietot[1:end-4] + 1/5*pietot[2:end-3] + 1/5*pietot[3:end-2] +
         1/5*pietot[4:end-1] + 1/5*pietot[5:end]; 

        CGE[i,:] = pietot  
    end 

    # cumtrapz 
    CGEsum[i,:] = cumtrapz(xc,CGE[i,:]) 

    if i==3; 
        global Πxztot  #xc, zc
        @load path_fname Πxztot; 

        if dx < 500
            # 3-pnt ave is enough to remove noise!
#            Πxztot[2:end-1,:] = 1/3*Πxztot[1:end-2,:] +
#            1/3*Πxztot[2:end-1,:] + 1/3*Πxztot[3:end,:]; #best

            # 5 point average
            Πxztot[3:end-2,:]  = 1/5*Πxztot[1:end-4,:] + 1/5*Πxztot[2:end-3,:] + 1/5*Πxztot[3:end-2,:] +
             1/5*Πxztot[4:end-1,:] + 1/5*Πxztot[5:end,:];             
        end 

    end
end

##

# plot figure
ylim = [0 7]
ylim3 = [-0.1 1.7]
ylim5 = [-2 35]
Ldom = 700e3;

ylim = [0 15]
ylim3 = [-0.1 3.5]
ylim5 = [-2 70]
Ldom = 700e3;

fig = Figure(size=(600,925))
ax = Axis(fig[1, 1],title = string("(a) ",titstr,"; tidal flux"), ylabel = "flux [W/m]")
ylims!(ax, ylim[1], ylim[2])
trslc = 0.2; vspan!(ax, 500, 700, color = (:yellow, trslc))
vspan!(ax, 0, 56, color = (:lightblue, trslc))
lines!(ax, xc/1e3, (FXT[1,:]+FXT[2,:])/1e3, label = "sim. mode 1 + sim. mode 2", color = :red, linewidth = 3, linestyle = :dash)
lines!(ax, xc/1e3, FXT[3,:]/1e3, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :solid)
axislegend(ax, position = :rt)
#ax.xticklabelsvisible = false

ax2 = Axis(fig[2, 1],title = "(b) supertidal flux", ylabel = "flux [W/m]")
ylims!(ax2, ylim[1], ylim[2])
vspan!(ax2, 500, 700, color = (:yellow, trslc))
vspan!(ax2, 0, 56, color = (:lightblue, trslc))
lines!(ax2, xc/1e3, (FXH[1,:]+FXH[2,:])/1e3, label = "sim. mode 1 + sim. mode 2", color = :red, linewidth = 3, linestyle = :dash)
lines!(ax2, xc/1e3, FXH[3,:]/1e3, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :solid)

ax3 = Axis(fig[3, 1],title = "(c) cross-scale energy transfers (τ=9.3 h)", ylabel = "Π [1e4 W/kg m]")
fc = 1e4;
ylims!(ax3, ylim3[1], ylim3[2])
vspan!(ax3, 500, 700, color = (:yellow, trslc))
vspan!(ax3, 0, 56, color = (:lightblue, trslc))
lines!(ax3, xc/1e3, (CGE[1,:]+CGE[2,:])*fc, label = "sim. mode 1 + sim. mode 2", color = :red, linewidth = 3, linestyle = :dash)
lines!(ax3, xc/1e3, CGE[3,:]*fc, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :solid)

ax5 = Axis(fig[4, 1],title = "(d) cumulative cross-scale energy transfers", 
ylabel = "Σ Π dx [W/kg m2]")
fc5 = 1;
ylims!(ax5, ylim5[1], ylim5[2])
vspan!(ax5, 500, 700, color = (:yellow, trslc))
vspan!(ax5, 0, 56, color = (:lightblue, trslc))
lines!(ax5, xc/1e3, (CGEsum[1,:]+CGEsum[2,:])*fc5, label = "sim. mode 1 + sim. mode 2", color = :red, linewidth = 3, linestyle = :dash)
lines!(ax5, xc/1e3, CGEsum[3,:]*fc5, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :solid)

ylim4 = -500;
fc2 = 1e7;
clims = (-1e-6*fc2,1e-6*fc2)
ax4 = Axis(fig[5, 1], title="(e) Π sim. mode 1 + 2 [1e-7 W/kg]", xlabel="x [km]", ylabel="z [m]"); 
hm = heatmap!(ax4, xc/1e3, zc, Πxztot*fc2, colormap = Reverse(:RdBu_5), colorrange = clims); 
ylims!(ax4, ylim4, 0)
Colorbar(fig[5,2], hm); 

xlims!(ax, 0, Ldom/1e3)
xlims!(ax2, 0, Ldom/1e3)
xlims!(ax3, 0, Ldom/1e3)
xlims!(ax4, 0, Ldom/1e3)
xlims!(ax5, 0, Ldom/1e3)
display(fig)

# Save the figure as a PNG file
titstr2 = replace(titstr, " " => "_", "(" => "", ")" => "")

if figflag==1; save(string(dirfig,"flux_coarsgr_",titstr2,".png"), fig)
end

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