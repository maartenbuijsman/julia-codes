#= IW_fft.jl
Maarten Buijsman, USM DMS, 2025-10-3
Compute some basic diagnostics, such as ffts
and undecomposed energetics
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2

figflag = 1
WIN = 1;

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

# function of latitude
lats = [0, 2.5, 5, 10, 20, 30, 40];

KEom = zeros(1000,length(lats));
#ii=1
for ii in 1:length(lats)

    lat = lats[ii]
    fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); 
    fname_short2 = fnames[1:33]

    filename = string(dirsim,fnames)

    T2 = 12+25.2/60

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
    wf = zeros(Nt,Nx,Nz+1);

    for i in 1:Nt
        println(i)
        uf[i,:,:] = ds["u"][:, :, i];
        vf[i,:,:] = ds["v"][:, :, i];
        wf[i,:,:] = ds["w"][:, :, i];
    end
    vc = vf; # all values at centers along x

    # close the nc file
    close(ds)

    # compute at cell centers
    # v is already at x,W centers
    uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 
    wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 

    # compute some ffts of surface velocity ======================================================

    Sp_Region_right = 20000
    Ix = findall(item -> item<=xc[end]+dx[1]/2-Sp_Region_right, xc)

    EXCL = 0;  # can be zero for fft
    t1,t2 = 4, tday[end]-EXCL*T2/24
    numcycles = floor((t2-t1)/(T2/24))
    t2 = t1+numcycles*(T2/24)
    Iday = findall(item -> item >= t1 && item<= t2, tday)

    tukeycf=0.2; numwin=2; linfit=true; prewhit=false;

    i=1;
    period, freq, pp = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit); #get the dimensions
    lep = length(period);
    poweru = zeros(lep,Nx);
    powerv = zeros(lep,Nx);
    for i in 1:Nx
        period, freq, poweru[:,i] = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);
        period, freq, powerv[:,i] = fft_spectra(tday[Iday], vc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);    
    end

    # sum along x
    KEom[1:lep,ii] = sum(poweru[:,Ix] .+ powerv[:,Ix],dims=2);   
end



# heatmap of spectral power
ylim = [0 11];
clims = (-0.05,0.05)

#tistr = " mode 1 + 2"
tistr = " mode 1"

fig1 = Figure()
axa = Axis(fig1[1, 1],yticks = (0:2:10),title=string("log10 KE [m2/s2] ",tistr),xlabel="x [km]",ylabel="frequency [cpd]");  
ylims!(axa, ylim[1], ylim[2])
hm = heatmap!(axa, xc/1e3, freq, log10.(transpose(KEom)),colormap = Reverse(:Spectral)); 
Colorbar(fig1[1,2], hm); 
hm.colorrange = (-6, 0)
fig1   

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"fft_usur_",fname_short2,".png"), fig1)
end
