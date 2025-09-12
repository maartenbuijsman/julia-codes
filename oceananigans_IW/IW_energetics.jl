#= IW_energetics.jl
Maarten Buijsman, USM DMS, 2025-9-11
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

#pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\"
pathname = "/home/mbui/Documents/julia-codes/functions/"
include(string(pathname,"include_functions.jl"))

dirsim = "/data3/mbui/ModelOutput/IW/";
dirfig = "/data3/mbui/ModelOutput/figs/";

# load simulations ===========================================

#fnames = "IW_fields_U0n0.1_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"
#fnames = "IW_fields_U0n0.2_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"

fnames = "AMZ1_lat0_8d_U1_0.25_U2_0.00.nc"  # mode 1
#fnames = "AMZ1_lat0_8d_U1_0.00_U2_0.20.nc"  # mode 2
#fnames = "AMZ1_lat0_8d_U1_0.25_U2_0.20.nc"  # mode 1+2

fname_short = fnames[1:28]

#filename = string("C:\\Users\\w944461\\Documents\\work\\data\\julia\\",fnames)
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
wf = zeros(Nt,Nx,Nz+1);
bc = zeros(Nt,Nx,Nz);

for i in 1:Nt
    println(i)
    uf[i,:,:] = ds["u"][:, :, i];
    vf[i,:,:] = ds["v"][:, :, i];
    wf[i,:,:] = ds["w"][:, :, i];
    bc[i,:,:] = ds["b"][:, :, i];
end
vc = vf; # all values at centers along x

# close the nc file
close(ds)

# compute at cell centers
# v is already at x,W centers
uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 
wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 


# compute some ffts of surface velocity ======================================================
tukeycf=0.1; numwin=1; linfit=true; prewhit=false;

i=1;
period, freq, pp = fft_spectra(tday, uc[:,i,end]; tukeycf, numwin, linfit, prewhit); #get the dimensions
poweru = zeros(length(period),Nx);
powerv = zeros(length(period),Nx);
for i in 1:Nx
    period, freq, poweru[:,i] = fft_spectra(tday, uc[:,i,end]; tukeycf, numwin, linfit, prewhit);
    period, freq, powerv[:,i] = fft_spectra(tday, vc[:,i,end]; tukeycf, numwin, linfit, prewhit);    
end

KEom = poweru .+ powerv;    # mode 1+2
#KEom1 = poweru .+ powerv;  # mode 1
#KEom2 = poweru .+ powerv;  # mode 2
#KEomdiff = KEom - (KEom1 + KEom2);

#=
lines(uc[:,100,end])
period, freq, pp = fft_spectra(tday, uc[:,100,end]; tukeycf, numwin, linfit, prewhit);
lines(pp)
lines(poweru[:,100])
=#


# heatmap of spectral power
ylim = [0 8];
clims = (-0.05,0.05)

fig1 = Figure()
axa = Axis(fig1[1, 1],title=string("KE [m^2/s^2] ",fname_short));  
#ylims!(axa, ylim[1], ylim[2])
#hm = heatmap!(axa, xc/1e3, freq, log10.(transpose(KEom)), colormap = Reverse(:Spectral)); 
hm = heatmap!(axa, xc/1e3, freq, log10.(transpose(KEom1)), colormap = Reverse(:Spectral)); 
#hm = heatmap!(axa, xc/1e3, freq, (transpose(KEomdiff)), colormap = Reverse(:Spectral), colorrange = clims); 
Colorbar(fig1[1,2], hm); 
fig1   


# line plots 
Isel = 49; xc[Isel]/1e3 # hotspot
Isel = 68; xc[Isel]/1e3 # in between hotspots

fig = Figure()
ax = Axis(fig[1, 1], title = "Power Spectrum",xlabel = "Frequency [cpd]", ylabel = "KE",yscale = log10)
lines!(ax, freq, KEom[:,Isel], color = :black, linewidth = 2)
lines!(ax, freq, KEom1[:,Isel]+KEom2[:,Isel], color = :red, linewidth = 2)
fig



# project velocities on modes and then compute energetics per mode ------------------------------



#####################################################################

#fnames = "IW_fields_U0n0.1_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"
fnames = "IW_fields_U0n0.2_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"

#filename = string("C:\\Users\\w944461\\Documents\\work\\data\\julia\\",fnames)
filename = string("/data3/mbui/ModelOutput/IW/",fnames)

ds = NCDataset(filename,"r");

tsec = ds["time"];
tday = tsec/24/3600;
dt = tday[2]-tday[1]

xf   = ds["x_faa"]; 
xc   = ds["x_caa"]; 
zc   = ds["z_aac"]; 
dz   = ds["Δz_aac"];

H  = sum(dz);   # depth
Nb = 0.005;     # buoyancy freq

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# buoyancy [m/s2]
# background = 
# b = N2 * z = -g/rho0*drho/dz * z
# b = -g/rho0*rho_pert
# rho_pert = -b*rho0/g 
# rho = -(N^2 * z + b)*rho0/g 
b = ds["b"];

# create density as a function of time
const rho0=1020; const grav=9.81; 
Nb2z = Nb^2 .* reshape(zc, 1, :, 1);   # shape: (1, length(zc), 1)
rho  = -(Nb2z .+ b) * rho0 / grav;       # broadcast without repeat
#rho = @. -(Nb2z + b) * rho0 / grav;    # broadcast without repeat

it = 350
fig = Figure(); Axis(fig[1,1],title="b & ρ"); 
heatmap!(xc/1e3,zc,b[:,:,it]); 
contour!(xc/1e3,zc,rho[:,:,it], color = :black); fig

Figure(); lines(rho[10,:,100],zc)
Figure(); lines(Nb2z[1,:,1],zc)
Figure(); lines(-Nb2z[1,:,1]*rho0/grav,zc)

#check memory

# MAR660 hydrostatic pressure ============================
# rho_pert = -b*rho0/g 
# dp       = -g*rho*dz
# In Oceananigans: dp/dz = b = -g/rho0*rho_pert [m2/s2]
# because of kinematic pressure p/rho

#= this is not really faster .....
using Base.Threads

#Nx, Nz, Nt = size(b)
pfi = similar(b)
cnt = zeros(Nx * Nt,2)
Threads.@threads for t in 1:(Nx * Nt)
    if rem(t,100)==0; println("t=",t); end
    # Flatten (i,k) space to distribute across threads
    i = ((t - 1) % Nx) + 1
    k = ((t - 1) ÷ Nx) + 1

    cnt[t,1] = i
    cnt[t,2] = k    
#    println(t,"; ",i,"; ",k)
    acc = zero(eltype(b))
    @inbounds @simd for j in Nz:-1:1
        acc += b[i, j, k] * dz[j]
        pfi[i, j, k] = acc
    end
end

a = zeros(100)
@threads for i = 1:100
           a[i] = Threads.threadid()
       end

=#

# hydrostatic pressure
dzz  = reshape(dz, 1, :, 1);                                # shape: (1, length(zc), 1)
pfi = cumsum(b[:,end:-1:1,:].*dzz[:,end:-1:1,:], dims=2);  # reverse, z surface down, at faces
pfi = pfi * -1 * rho0 / grav;                             # convert to pert pressure

# average to centers, and reverse back (z bottom up)
pc = zeros(size(pfi));
pc[:,1:end-1,:] = pfi[:,end:-1:2,:]/2 + pfi[:,end-1:-1:1,:]/2; # compute center values
pc[:,end,:]     = pfi[:,1,:]/2;                                # add surface value
#pc[1,:,10]

# remove depth-mean
pa  = sum(pc.*dzz,dims=2)/H; # depth-mean pressure
#pa[1,:,100]
pcp = pc .- pa;             # the perturbation pressure!

#check integral of perturbation pressure should be zero 
#sum(pcp[100,:,300].*dz)   
Figure(); lines(pcp[10,:,100],zc)

fig = Figure(); Axis(fig[1,1],title="pk [m2/s2]"); 
vflmap!(xc/1e3,zc,pcp[:,:,300]); fig
contour!(xc/1e3,zc,b[:,:,300], color = :black); fig

# compute some energy terms ===================================

# centered velocities
# u(x_faa, z_aac, time)
uf = ds["u"];
uc = uf[1:end-1,:,:]/2 + uf[2:end,:,:]/2; #map to centers

# some more hovmullers
fig1 = Figure()
ax1a = fig1[1, 1] 
ax1b = fig1[2, 1] 
heatmap(ax1a, xc/1e3, tday, b[:,Nz ÷ 2,:])
heatmap(ax1b, xc/1e3, tday, uc[:,end,:])
fig1

KE = dropdims(mean(uc.^2, dims=3), dims=3);

fig2 = Figure()
ax2 = Axis(fig2[1,1]);
hm = heatmap!(ax2, xc/1e3, zc , KE, colormap = Reverse(:Spectral))
Colorbar(fig2[1,2], hm)
fig2

# depth-integrated pressure fluxes
Fx = sum(uc.*pcp.*dzz, dims=2);
Fx = dropdims(Fx, dims=2);

fig2 = Figure()
ax2 = Axis(fig2[1,1]);
hm = heatmap!(ax2, xc/1e3, tday, Fx/1e3, colormap = Reverse(:Spectral))
Colorbar(fig2[1,2], hm)
fig2

# fft on surface velocities along the transect ==============

# surf vel
ucs = uc[:,end,:]

fig3 = Figure()
ax3 = fig3[1, 1] 
heatmap(ax3, xc/1e3, tday, ucs)
fig3

tukeycf=0.0; numwin=1; linfit=true; prewhit=false;
Nfreq = Nt÷numwin÷2;
pwr = Matrix{Float64}(undef, Nx, Nfreq); 
period=[]; freq=[];
for i=1:Nx
    period, freq, pwr[i,:] = fft_spectra(tday, ucs[i,:]; tukeycf, numwin, linfit, prewhit);    
end

fig4 = Figure()
ax4 = Axis(fig4[1, 1])  # <-- create Axis, not GridPosition
heatmap!(xc ./ 1e3, freq, log10.(pwr))
ylims!(ax4, 0, 5)
fig4


# bandpass filtering ============================
Tl,Th,dth,N = 9,15,dt*24,4

ucsf = bandpass_butter(ucs',Tl,Th,dth,N)'

ix = 50

fig = Figure()
ax = Axis(fig[1, 1]) 
lines!(ax,tday,ucs[ix,:],color = :red)
lines!(ax,tday,ucsf[ix,:],color = :green, linestyle = :dash)
lines!(ax,tday,ucs[ix,:]-ucsf[ix,:],color = :magenta, linestyle = :dash)
fig

fig3 = Figure()
ax3 = fig3[1, 1] 
heatmap(ax3, xc/1e3, tday, ucsf)
fig3



# close the nc file
close(ds)
