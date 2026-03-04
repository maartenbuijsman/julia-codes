#= IW_coarsegraining.jl
Maarten Buijsman, USM DMS, 2025-8-7
Load model runs and perform coarsegraining diagnostics
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2
using LaTeXStrings


WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    dirsim = "/data3/mbui/ModelOutput/IW/";
    dirfig = "/data3/mbui/ModelOutput/figs/";
    dirout = "/data3/mbui/ModelOutput/diagout/";
end

include(string(pathname,"include_functions.jl"))

# print figures
figflag = 1

const T2 = 12+25.2/60

# file name ===========================================

# 01 expts
mainnm = 1

#      38 39 40 41 42 43 44 45 46 47 48    49
LATS = [0 2.5 5 10 15 20 25 30 40 50 28.80 35];
runnms = [38 39 40 41 42 43 44 45 46 47 48 49];

# 03 expts
oldnm  = 0  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing

mainnm = 3    
runnm  = 1 
LAT    = 0

# file name ===========================================
#####for runnm in runnms
    #LAT = LATS[runnm-37];
    #println("lat is ",LAT,"------------------------------------") 
    #end

if oldnm==1
    # function of latitude
    LAT = LATS[1];

   #fnames = @sprintf("AMZv_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",LAT); titlenm = "mode 1"
   # fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",LAT); titlenm = "mode 1"  # hydro
     fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.30.nc",LAT); titlenm = "mode 1+2"  # hydro   
   # fnames = @sprintf("AMZ4_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",LAT); titlenm = "mode 1"     # nonhydro
   # fnames = @sprintf("AMZ4_%04.1f_hvis_12d_U1_0.40_U2_0.30.nc",LAT); titlenm = "mode 1+2"    # nonhydro

    fname_short2 = fnames[1:33]
    filename = string(dirsim,fnames)

    titlenm = string(LAT,"°N; ",titlenm)    
    titlenm2 = string(LAT,"°N")            
else
    # file ID
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")

    println("lat is ",LAT,"------------------------------------") 

    titlenm = string(LAT,"°N; mode 1")    
    titlenm2 = string(LAT,"°N")        
end


# load simulations ===========================================

ds = NCDataset(filename,"r");

tsec = ds["time"][:];
tday = tsec/24/3600;
dt = tday[2]-tday[1]

xf   = ds["x_faa"][:]; 
xc   = ds["x_caa"][:]; 
zc   = ds["z_aac"][:]; 

dx   = ds["Δx_caa"][:];
dz   = ds["Δz_aac"][:];

H  = sum(dz);   # depth

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# u, v, w velocities
# NOTE: in future select a certain x range away from boundaries

# 
#uf = ds["u"]; #x_faa × z_aac × time
#vf = ds["v"]; #x_caa × z_aac × time
#wf = ds["w"]; #x_caa × z_aaf × time

# is a loop a faster way of permuting?
uf = zeros(Nt,Nx+1,Nz);
vf = zeros(Nt,Nx,Nz);
wf = zeros(Nt,Nx,Nz+1);

for i in 1:Nt
    println(i)
    uf[i,:,:] = ds["u"][:, :, i];
    vf[i,:,:] = ds["v"][:, :, i];
    wf[i,:,:] = ds["w"][:, :, i];
end

# close the nc file
close(ds)


# compute at cell centers
# v is already at x,W centers
uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 
wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 

#= fig = Figure(); ax = Axis(fig[1, 1])
lines!(ax,tday,uf[:,1,1],color=:red)
fig =#


#= some more hovmullers
clims = (-0.5,0.5)
fig1 = Figure()
ax1 = Axis(fig1[1, 1], xlabel = "x [km]", ylabel = "time [days]", title=string("u surf [m/s] ",fname_short2)) 
hm=heatmap!(ax1, xc/1e3, tday , uc[:,:,end]', colormap = Reverse(:Spectral), colorrange = clims)
Colorbar(fig1[1, 2], hm)
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"hovmuller_",fname_short2,".png"), fig1)
end
=#

# filter all velocities ======================================

# todo: change N, apply a tukey 10%?

dth=dt*24; N = 8;   # all in hours

# low-pass
Tcut=9;  # all in hours
fstring = "low"

#= test
uu = uf[1,1,:]; uuf = lowhighpass_butter(uu,Tcut,dth,N,fstring)
ww = wf[ix,15,:].*tukey(Nt,0); wwf = lowhighpass_butter(ww,Tcut,dth,N,fstring) 

fig = Figure(); ax = Axis(fig[1, 1])
lines!(ax,tday,ww,color=:red)
lines!(ax,tday,wwf,color=:green,linestyle=:dash); fig
=#

@time begin
    ufl = lowhighpass_butter(uf,Tcut,dth,N,fstring);
    vfl = lowhighpass_butter(vf,Tcut,dth,N,fstring);
    wfl = lowhighpass_butter(wf,Tcut,dth,N,fstring);

    ucl = lowhighpass_butter(uc,Tcut,dth,N,fstring);
    wcl = lowhighpass_butter(wc,Tcut,dth,N,fstring);

    uucl = lowhighpass_butter(uc.*uc,Tcut,dth,N,fstring);
    uvcl = lowhighpass_butter(uc.*vf,Tcut,dth,N,fstring);
    uwcl = lowhighpass_butter(uc.*wc,Tcut,dth,N,fstring);

    vvcl = lowhighpass_butter(vf.*vf,Tcut,dth,N,fstring);    
    vwcl = lowhighpass_butter(vf.*wc,Tcut,dth,N,fstring);    

    wwcl = lowhighpass_butter(wc.*wc,Tcut,dth,N,fstring);        

    println("finished in ")
end


fig = Figure()
ax = Axis(fig[1, 1]) 
ix = 50
lines!(ax,tday,wf[:,ix,15],color = :red)
lines!(ax,tday,wfl[:,ix,15],color = :green, linestyle = :dash)
fig

#= compute the terms ======================================
Π = ((ui*uj)_ - ui_*uj_)*dui/dxj, i=j=1,2,3
units: m2/s2*1/s = W/kg
units*rho = m2/s2*1/s *kg/m3 = W/m3
u*u - u*u * dudx  cd 
u*w - u*w * dudz  

v*u - v*u * dvdx  
v*w - v*w * dvdz 

w*u - w*u * dwdx (these are likely small, for nonhydrostatic sims)
w*w - w*w * dwdz  cd

# omitted for 2D
u*v - u*v * dudy 
v*v - v*v * dvdy 
w*v - w*v * dwdy 
=#

# dui/dxj first ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# these gradients could be computed with a higher order?

# finite diff 
# dudx & dwdz
dxr = reshape(dx, 1, :, 1); 
dudx = diff(ufl,dims=2)./dxr;

dzr = reshape(dz, 1, 1, :); 
dwdz = diff(wfl,dims=3)./dzr;

# central finite diff
# dudz & dvdz -----------------------------
dudz = zeros(Nt,Nx,Nz);
dvdz = zeros(Nt,Nx,Nz);
dz2  = zc[3:end] - zc[1:end-2];
dzz2 = reshape(dz2,1,1,:);
dudz[:,:,2:end-1] = (ucl[:,:,3:end] - ucl[:,:,1:end-2])./dzz2;
dvdz[:,:,2:end-1] = (vfl[:,:,3:end] - vfl[:,:,1:end-2])./dzz2;

# take care of the boundaries
dzb  = zc[2] - zc[1]; #bottom
dudz[:,:,1] = (ucl[:,:,2] - ucl[:,:,1])/dzb;
dvdz[:,:,1] = (vfl[:,:,2] - vfl[:,:,1])/dzb;

dzs  = zc[end] - zc[end-1] #surface
dudz[:,:,end] = (ucl[:,:,end] - ucl[:,:,end-1])/dzs;
dvdz[:,:,end] = (vfl[:,:,end] - vfl[:,:,end-1])/dzs;

# dvdx & dwdx ------------------------------
dvdx = zeros(Nt,Nx,Nz);
dwdx = zeros(Nt,Nx,Nz);
dx2  = xc[3:end] - xc[1:end-2];
dxx2 = reshape(dx2,1,:,1);
dvdx[:,2:end-1,:] = (vfl[:,3:end,:] - vfl[:,1:end-2,:])./dxx2;
dwdx[:,2:end-1,:] = (wcl[:,3:end,:] - wcl[:,1:end-2,:])./dxx2;

# take care of the boundaries
dxl = xc[2] - xc[1]; #left
dvdx[:,1,:] = (vfl[:,2,:] - vfl[:,1,:])/dxl;
dwdx[:,1,:] = (wcl[:,2,:] - wcl[:,1,:])/dxl;

dxr = xc[end] - xc[end-1]; #right
dvdx[:,end,:] = (vfl[:,end,:] - vfl[:,end-1,:])/dxr;
dwdx[:,end,:] = (wcl[:,end,:] - wcl[:,end-1,:])/dxr;

#= Π: combine the terms ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
u*u - u*u * dudx  1 
v*u - v*u * dvdx

u*w - u*w * dudz  2
v*w - v*w * dvdz 

w*u - w*u * dwdx  3
w*w - w*w * dwdz
=#

# switch to positive -=>+
Πx = (ucl.*ucl .- uucl).*dudx +   
     (vfl.*ucl .- uvcl).*dvdx; 
Πz = (ucl.*wcl .- uwcl).*dudz +   
     (vfl.*wcl .- vwcl).*dvdz; 
Πnh = (ucl.*wcl .- uwcl).*dwdx +   
      (wcl.*wcl .- wwcl).*dwdz; 

#= snapshot
ylim = -500;
clims = (-maximum(Πx[:]),maximum(Πx[:]))

fig1 = Figure()
axa = Axis(fig1[1, 1])
axb = Axis(fig1[2, 1])
axc = Axis(fig1[3, 1])
hma = heatmap!(axa, xc/1e3, zc, Πx[300,:,:], colormap = Reverse(:Spectral), colorrange = clims);
hmb = heatmap!(axb, xc/1e3, zc, Πz[300,:,:], colormap = Reverse(:Spectral), colorrange = clims); 
hmc = heatmap!(axc, xc/1e3, zc, Πnh[300,:,:], colormap = Reverse(:Spectral), colorrange = clims); 

Colorbar(fig1[1,2], hma); ylims!(axa, ylim, 0)
Colorbar(fig1[2,2], hmb); ylims!(axb, ylim, 0)
Colorbar(fig1[3,2], hmc); ylims!(axc, ylim, 0)
fig1  
=#


# time average
# use an fixed number of tidal cycles after 4 days to capture modes 1 and 2
# and before Tend-TM2 to avoid ringing effects

# number of tidal cycles to exclude from the end
#EXCL = 2 
EXCL = 4
t1,t2 = 4, tday[end]-EXCL*T2/24
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)
#tday[Iday]

# f(x,z)
Πxa = dropdims(mean(Πx[Iday,:,:],dims=1),dims=1);
Πza = dropdims(mean(Πz[Iday,:,:],dims=1),dims=1);
Πnha = dropdims(mean(Πnh[Iday,:,:],dims=1),dims=1);

ylim = -500;

fig1 = Figure(size = (600, 800))
axa = Axis(fig1[1, 1],title=string("Pix [W/kg] ",fname_short2,"; ",titlenm2));  ylims!(axa, ylim, 0)
axb = Axis(fig1[2, 1],title=string("Piz [W/kg] "));  ylims!(axb, ylim, 0)
axc = Axis(fig1[3, 1],title=string("Pinh [W/kg] "));  ylims!(axc, ylim, 0)
axd = Axis(fig1[4, 1],title=string("Pisum [W/kg] "));  ylims!(axd, ylim, 0)
hm = heatmap!(axa, xc/1e3, zc, Πxa, colormap = (:Spectral)); Colorbar(fig1[1,2], hm); 
hm = heatmap!(axb, xc/1e3, zc, Πza, colormap = (:Spectral)); Colorbar(fig1[2,2], hm); 
hm = heatmap!(axc, xc/1e3, zc, Πnha, colormap = (:Spectral)); Colorbar(fig1[3,2], hm);
hm = heatmap!(axd, xc/1e3, zc, Πxa.+Πza.+Πnha, colormap = (:Spectral)); Colorbar(fig1[4,2], hm); 
fig1    

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"PIxz_",fname_short2,".png"), fig1)
end

#title=latexstring("sim. ",titlenm,"; \$\\Pi\$ [W/kg] ")

# only plot the sum of all coarse graining terms :balance :RdBu_5
clims = (-1e-6,1e-6)
fig1 = Figure(size = (1000, 250))
axa = Axis(fig1[1, 1], title=string(titlenm,"; cross-scale transfer [W/kg] "), xlabel="x [km]", ylabel="z [m]"); 
#hm = heatmap!(axa, xc/1e3, zc, Πxa.+Πza.+Πnha, colormap = Reverse(:Spectral), colorrange = clims); 
hm = heatmap!(axa, xc/1e3, zc, Πxa.+Πza.+Πnha, colormap = Reverse(:RdBu_5), colorrange = clims); 
ylims!(axa, ylim, 0)
Colorbar(fig1[1,2], hm); 
fig1    

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"PIsumxz_",fname_short2,".png"), fig1)
end

# f(z)
Πxza = dropdims(mean(Πxa,dims=1),dims=1);
Πzza = dropdims(mean(Πza,dims=1),dims=1);
Πnhza = dropdims(mean(Πnha,dims=1),dims=1);

fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "Π [W/kg]", ylabel = "z [m]", title=fname_short2)
lines!(ax,Πxza,zc,color=:red, label="Πx")
lines!(ax,Πzza,zc,color=:green, label="Πz")
lines!(ax,Πnhza,zc,color=:black, label="Πnh")
lines!(ax,Πnhza+Πzza+Πxza,zc,color=:orange, label="sum")
axislegend(ax,position = :rb; framevisible = false )
fig

# f(x)
dzz = reshape(dz,1,:)
Πxxa  = dropdims(sum(Πxa.*dzz,dims=2),dims=2);
Πzxa  = dropdims(sum(Πza.*dzz,dims=2),dims=2);
Πnhxa = dropdims(sum(Πnha.*dzz,dims=2),dims=2);

#
ax2 = Axis(fig[1, 2], xlabel = "x [km]", ylabel = "Π [W/kg*m]", title=string(fname_short2,"; ",titlenm2))
lines!(ax2,xc/1e3,Πxxa,color=:red, label="Πx")
lines!(ax2,xc/1e3,Πzxa,color=:green, label="Πz")
lines!(ax2,xc/1e3,Πnhxa,color=:black, label="Πnh")
lines!(ax2,xc/1e3,Πnhxa+Πzxa+Πxxa,color=:orange, label="sum")
axislegend(ax2,position = :rt; framevisible = false )
fig
#

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"PIz_PIx_",fname_short2,".png"), fig)
end

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

# save only the pi(x) UNSMOOTHED
xlim = 1400
#xlim = 700
fig = Figure(size = (300, 450));
 ax2 = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]", title=string(fname_short2,"; ",titlenm2))
lines!(ax2,xc/1e3,Πxxa,color=:red, label="Πx")
lines!(ax2,xc/1e3,Πzxa,color=:green, label="Πz")
lines!(ax2,xc/1e3,Πnhxa,color=:black, label="Πnh")
lines!(ax2,xc/1e3,Πnhxa+Πzxa+Πxxa,color=:orange, label="sum", linewidth = 3)
axislegend(ax2,position = :rt; framevisible = false )
xlims!(ax2, 0, xlim)
ylims!(ax2, -1e-4, 1e-4)
fig
#

if figflag==1; save(string(dirfig,"PIx_",fname_short2,".png"), fig)
end

stop()

# save f(x) profiles
fnameout = string("Etran_",fname_short2,".jld2")

jldsave(string(dirout,fnameout); LAT, xc, Πnhxa, Πzxa, Πxxa, zc, Πnhza, Πzza, Πxza);
println(string(fnameout)," data saved ........ ")

#####end # runnums

stop()

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