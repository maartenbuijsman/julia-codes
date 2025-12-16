#= IW_fft_x_omega.jl
Maarten Buijsman, USM DMS, 2025-12-2
compute KE spectra
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2

figflag = 1
WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
    dirEIG = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    dirsim = "/data3/mbui/ModelOutput/IW/";
    dirfig = "/data3/mbui/ModelOutput/figs/";
    dirout = "/data3/mbui/ModelOutput/diagout/";
    dirEIG = "/data3/mbui/ModelOutput/IW/forcingfiles/";
end

include(string(pathname,"include_functions.jl"))

# load simulations ===========================================

#fnames = "IW_fields_U0n0.1_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"
#fnames = "IW_fields_U0n0.2_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"

#fnames = "AMZ1_lat0_8d_U1_0.25_U2_0.00.nc"  # mode 1
#fnames = "AMZ1_lat0_8d_U1_0.00_U2_0.20.nc"  # mode 2
#fnames = "AMZ1_lat0_8d_U1_0.25_U2_0.20.nc"  # mode 1+2

#fnames = "AMZ2_lat0_8d_U1_0.50_U2_0.00.nc"  # mode 1
#fnames = "AMZ2_lat0_8d_U1_0.00_U2_0.40.nc"  # mode 2
#fnames = "AMZ2_lat0_8d_U1_0.50_U2_0.40.nc"  # mode 1+2

#fnames = "AMZ2_lat0_12d_U1_0.50_U2_0.00.nc"  # mode 1
#fnames = "AMZ2_lat0_12d_U1_0.00_U2_0.40.nc"  # mode 2
#fnames = "AMZ2_lat0_12d_U1_0.50_U2_0.40.nc"  # mode 1+2

# very smooth; but causes large decay
#fnames = "AMZ3_weno_12d_U1_0.50_U2_0.40.nc"  # mode 1+2
#fnames = "AMZ3_visc_12d_U1_0.50_U2_0.40.nc"  # mode 1+2
#fnames = "AMZ3_hvis_12d_U1_0.50_U2_0.40.nc"  # mode 1+2

#fnames = "AMZ3_hvis_12d_U1_0.40_U2_0.30.nc"; titlenm = "mode 1 + 2"  # mode 1+2
#fnames = "AMZ3_hvis_12d_U1_0.40_U2_0.00.nc"; titlenm = "mode 1"  # mode 1
#fnames = "AMZ3_hvis_12d_U1_0.00_U2_0.30.nc"; titlenm = "mode 2"  # mode 2

#fname_short2 = fnames[1:29]

# function of latitude
lat = 0

fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"
#fnames = @sprintf("AMZ4_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"
fname_short2 = fnames[1:33]

filename = string(dirsim,fnames)

const T2 = 12+25.2/60

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

# a loop is faster than permuting?
uf = zeros(Nt,Nx+1,Nz);
vf = zeros(Nt,Nx,Nz);

for i in 1:Nt
    println(i)
    uf[i,:,:] = ds["u"][:, :, i];
    vf[i,:,:] = ds["v"][:, :, i];
end
vc = vf; # all values at centers along x

# close the nc file
close(ds)

# compute at cell centers
# v is already at x,W centers
uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 

# some more hovmullers
fig1 = Figure(size=(660,800))
ax1a = fig1[1, 1] 
ax1b = fig1[2, 1] 
heatmap(ax1a, xc/1e3, tday, transpose(vc[:,:,end]))
heatmap(ax1b, xc/1e3, tday, transpose(uc[:,:,end]))
fig1


# compute some ffts of surface velocity ======================================================

EXCL = 0;  # can be zero for fft
t1,t2 = 4, tday[end]-EXCL*T2/24
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

tukeycf=0.2; numwin=2; linfit=true; prewhit=false;

i=1;
period, freq, pp = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit); #get the dimensions
poweru = zeros(length(period),Nx);
powerv = zeros(length(period),Nx);
for i in 1:Nx
    period, freq, poweru[:,i] = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);
    period, freq, powerv[:,i] = fft_spectra(tday[Iday], vc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);    
end

KEom = poweru .+ powerv;    # mode 1+2

# heatmap of spectral power
ylim = [0 24];
clims = (-0.05,0.05)

#tistr = " mode 1 + 2"
tistr = " mode 1"

fig1 = Figure()
axa = Axis(fig1[1, 1],yticks = (0:2:24),title=string("log10 KE [m2/s2] ",tistr),xlabel="x [km]",ylabel="frequency [cpd]");  
ylims!(axa, ylim[1], ylim[2])
hm = heatmap!(axa, xc/1e3, freq, log10.(transpose(KEom)),colormap = Reverse(:Spectral)); 
Colorbar(fig1[1,2], hm); 
hm.colorrange = (-6, 0)
fig1   

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"fft_usur_",fname_short2,".png"), fig1)
end


# spectra
xlim = [0 24];
ylim = [1e-8 1e0];

fig1 = Figure()
axa = Axis(fig1[1, 1],xticks = (0:2:24),
title=string("log10 KE [m2/s2] ",tistr),
ylabel="Power [m2/s2]",xlabel="frequency [cpd]",yscale = log10);  

#for i in 250:1000:2250
for i in 12:50:112
    println(xc[i])
    lines!(axa,freq,(KEom[:,i]),label=@sprintf("%04.1f km",xc[i]/1e3))
end
axislegend(axa, position = :lb)
xlims!(axa, xlim[1], xlim[2])
ylims!(axa, ylim[1], ylim[2])
fig1
