#= IW_total_energetics.jl
Maarten Buijsman, USM DMS, 2025-12-22
Compute undecomposed energetics: KE, APE, and pressure fluxes
for total and high-passed fields
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2

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

# print figures
figflag = 1
oldnm   = 1  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing
const T2 = 12+25.2/60

# file name ===========================================

if oldnm==1
    # function of latitude
    lat = 40

   #fnames = @sprintf("AMZv_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"
    fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"

    fname_short2 = fnames[1:33]
    filename = string(dirsim,fnames)
else
    # file ID
    mainnm = 1
    runnm  = 17

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")
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

# some more hovmullers
fig1 = Figure(size=(600,600))
ax = Axis(fig1[1, 1],title = string(fname_short2 ," KE [m2/s2]"), xlabel = "x [km]", ylabel = "time [days]")
hm = heatmap!(ax, xc/1e3, tday, transpose(uc[:,:,end].^2 .+ vc[:,:,end].^2), colormap = Reverse(:Spectral)); Colorbar(fig1[1,2], hm); 
#hm = heatmap!(ax, xc/1e3, tday, transpose(uc[:,:,end].^2), colormap = Reverse(:Spectral)); Colorbar(fig1[1,2], hm); 
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"KE_hovmuller_", fname_short2 ,".png"), fig1)
end

stop()

# load N2 pressure -----------------------------------------------------------
# load profile created by AMZ_stratification_profile.jl
dirin     = "/data3/mbui/ModelOutput/IW/forcingfiles/";
fnamegrid = "N2_amz1.jld2";
path_fname = string(dirin,fnamegrid);

# variables loaded
# "N2w", "zfw", "lonsel", "latsel"

# Open the JLD2 file
gridfile = jldopen(path_fname, "r")
println(keys(gridfile))  # List the keys (variables) in the file
close(gridfile)

@load path_fname N2w zfw

# map to cell centers
N2c = N2w[1:end-1]/2 + N2w[2:end]/2;

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,N2c, zc)
ylims!(ax1, -500, 0)
#xlims!(ax1, -2000, 10)
fig


# compute pressure -----------------------------------------------------------

# rho_pert = -b*rho0/g 
# dp       = -g*rho*dz
# dp/dz    = b*rho0

# In Oceananigans: dpk/dz = b = -g/rho0*rho_pert [m/s2]
#                  [m2/s2*1/m] = [m/s2]               
# units b: [m/s2]
# because of kinematic pressure pk = p/rho0 = kg*m2/s2*1/m3 *m3/kg = m2/s2

const rho0=1020; 
const grav=9.81; 

# hydrostatic pressure
dzz = reshape(dz,1,1,:);                                    # shape: (1, length(zc), 1)
N2cc = reshape(N2c,1,1,:);
pfi = cumsum(bc[:,:,end:-1:1].*dzz[:,:,end:-1:1], dims=3);  # reverse, z surface down, at faces
pfi = pfi * -1 * rho0 / grav;                               # convert to pert pressure

# average to centers, and reverse back (z bottom up)
pc = zeros(size(pfi));
pc[:,:,1:end-1] = pfi[:,:,end:-1:2]/2 + pfi[:,:,end-1:-1:1]/2; # compute center values
pc[:,:,end]     = pfi[:,:,1]/2;                                # add surface value
#pc[1,:,10]

# remove depth-mean
pa  = sum(pc.*dzz,dims=3)/H; # depth-mean pressure
#pa[1,:,100]
pcp = pc .- pa;             # the perturbation pressure!

#check integral of perturbation pressure should be zero 
sum(pcp[200,50,:].*dz)   
Figure(); lines(tday, pcp[:,50,end])

# filter variables  ======================================================
Nf = 6;

# remove the low frequency motions - if any?
Tcut1 = 16/24;
uc2  = lowhighpass_butter(uc,Tcut1,dt,Nf,"high");
vc2  = lowhighpass_butter(vc,Tcut1,dt,Nf,"high");
pcp2 = lowhighpass_butter(pcp,Tcut1,dt,Nf,"high");
bc2  = lowhighpass_butter(bc,Tcut1,dt,Nf,"high");
# isolate the subtidal flows
ull = uc - uc2;
vll = vc - vc2;

# remove high freq from tidal freq
Tcut2 = 9/24;
uh = lowhighpass_butter(uc2,Tcut2,dt,Nf,"high");
vh = lowhighpass_butter(vc2,Tcut2,dt,Nf,"high");
ph = lowhighpass_butter(pcp2,Tcut2,dt,Nf,"high");
bh = lowhighpass_butter(bc2,Tcut2,dt,Nf,"high");
ul = uc2 - uh;
vl = vc2 - vh;
pl = pcp2 - ph;
bl = bc2 - bh;

# do not remove any low-pass motions
uh2 = lowhighpass_butter(uc,Tcut2,dt,Nf,"high");
vh2 = lowhighpass_butter(vc,Tcut2,dt,Nf,"high");
bh2 = lowhighpass_butter(bc,Tcut2,dt,Nf,"high");
ul2 = uc-uh2;
vl2 = vc-vh2;
bl2 = bc-bh2;

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax,tday,ul[:,100,end], color = :black)
lines!(ax,tday,ul2[:,100,end], color = :red)
fig

# some more hovmullers
clims = (-0.2,0.2)
fig1 = Figure(size=(660,800))
ax1a = Axis(fig1[1, 1],title="subtidal u")
ax1b = Axis(fig1[2, 1],title="subtidal v") 
hm1=heatmap!(ax1a, xc/1e3, tday, transpose(ull[:,:,end]), colormap = Reverse(:Spectral), colorrange = clims)
hm2=heatmap!(ax1b, xc/1e3, tday, transpose(vll[:,:,end]), colormap = Reverse(:Spectral), colorrange = clims)
Colorbar(fig1[1, 2], hm1)
Colorbar(fig1[2, 2], hm2)
fig1


fig1 = Figure(size=(660,800))
ax1a = Axis(fig1[1, 1],title="b")
ax1b = Axis(fig1[2, 1],title="bl") 
hm1=heatmap!(ax1a, xc/1e3, tday, transpose(bc[:,:,85]), colormap = Reverse(:Spectral))
hm2=heatmap!(ax1b, xc/1e3, tday, transpose(bl[:,:,85]), colormap = Reverse(:Spectral))
Colorbar(fig1[1, 2], hm1)
Colorbar(fig1[2, 2], hm2)
fig1


# filtered KE, APE, and fluxes =======================================================

# cycles to exclude
EXCL = 4;
t1,t2 = 4, tday[end]-EXCL*T2/24
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

# undecomposed time-mean KE energy 
fact = 1/2*rho0
KEt = fact*dropdims(mean(sum((uc[Iday,:,:].^2 .+ vc[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KE  = fact*dropdims(mean(sum((uc2[Iday,:,:].^2 .+ vc2[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEh = fact*dropdims(mean(sum((uh[Iday,:,:].^2 .+ vh[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEl = fact*dropdims(mean(sum((ul[Iday,:,:].^2 .+ vl[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEll = KEt - KE;

KEut = fact*dropdims(mean(sum(uc[Iday,:,:].^2  .*dzz,dims=3),dims=1), dims=(1,3))
KEu  = fact*dropdims(mean(sum(uc2[Iday,:,:].^2 .*dzz,dims=3),dims=1), dims=(1,3))
KEuh = fact*dropdims(mean(sum(uh[Iday,:,:].^2  .*dzz,dims=3),dims=1), dims=(1,3))
KEul = fact*dropdims(mean(sum(ul[Iday,:,:].^2  .*dzz,dims=3),dims=1), dims=(1,3))

# this is equal to KE?
KE2 = KEh + KEl;

# APE
# b = -g/rho0*rho_pert [m/s2]
# 1/2*rho0*b2/N2 [J/m3 = Nm/m3 = kg*m2/s2/m3]
#  [kg/m3 * m2/s4 * s2 =         kg*m2/s2/m3]

# omit N2c values <= 1e-10, keep the others
Ikp = findall(>(1e-10),N2c)

factA = 1/2*rho0*1.0./N2cc;
APEt = dropdims(mean(sum((bc[Iday,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3),dims=1), dims=(1,3))
APE  = dropdims(mean(sum((bc2[Iday,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3),dims=1), dims=(1,3))
APEh = dropdims(mean(sum((bh[Iday,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3),dims=1), dims=(1,3))
APEl = dropdims(mean(sum((bl[Iday,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3),dims=1), dims=(1,3))
APEll = APEt - APE;

# undecomposed time-mean flux 
Fxt = dropdims(mean(sum(uc[Iday,:,:].*pcp[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fx  = dropdims(mean(sum(uc2[Iday,:,:].*pcp2[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxh = dropdims(mean(sum(uh[Iday,:,:].*ph[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxl = dropdims(mean(sum(ul[Iday,:,:].*pl[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))

# this is equal to Fx?
Fx2 = Fxh + Fxl


# create some figures
ylimE = [0 75]
ylimA = [0 75]
ylimf = [0 15]

fig = Figure(size=(750,750))
ax = Axis(fig[1, 1],title = string(fname_short2 ," KE"), xlabel = "x [km]", ylabel = "KE [kJ/m2]")
lines!(ax, xc/1e3, KEt[:,1]/1e3, label = "tot", linestyle=:dash, color = :grey, linewidth = 3)
#lines!(ax, xc/1e3, KE[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax, xc/1e3, KE2[:,1]/1e3, label = "D2 + HH", linestyle=:dash, color = :yellow, linewidth = 3)
lines!(ax, xc/1e3, KEl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax, xc/1e3, KEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)

lines!(ax, xc/1e3, KEut[:,1]/1e3, label = "tot", linestyle=:dash, color = :blue, linewidth = 3)
lines!(ax, xc/1e3, KEu[:,1]/1e3, label = "D2 + HH", color = :blue, linewidth = 3)
lines!(ax, xc/1e3, KEul[:,1]/1e3, label = "D2", color = :orange, linewidth = 3)
lines!(ax, xc/1e3, KEuh[:,1]/1e3, label = "HH", color = :cyan, linewidth = 3)

xlims!(ax, 0, 500)
ylims!(ax, ylimE[1], ylimE[2])

ax2 = Axis(fig[2, 1],title = "APE", xlabel = "x [km]", ylabel = "APE [kJ/m2]")

lines!(ax2, xc/1e3, APEt[:,1]/1e3, label = "tot",linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax2, xc/1e3, APE[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax2, xc/1e3, APEl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax2, xc/1e3, APEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)

xlims!(ax2, 0, 500)
ylims!(ax2, ylimA[1], ylimA[2])


ax3 = Axis(fig[3, 1],title = "flux", xlabel = "x [km]", ylabel = "flux [W/m]")

lines!(ax3, xc/1e3, Fxt[:,1]/1e3, label = "tot",linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax3, xc/1e3, Fx[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax3, xc/1e3, Fxl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax3, xc/1e3, Fxh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)

xlims!(ax3, 0, 500)
ylims!(ax3, ylimf[1], ylimf[2])
axislegend(ax3, position = :rt)

fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"KE_flux_", fname_short2 ,".png"), fig)
end



# save the energy terms =========================================
fnameout = string("energetics_",fname_short2,".jld2")

jldsave(string(dirout,fnameout); xc, Fxt, Fx, Fxh, Fxl, Fx2, KEt, KE, KEh, KEl, KE2, KEut, KEu, KEuh, KEul);
println(string(fnameout)," data saved ........ ")


# nonhyd pressure
# nonlinear APE
# ratio KE/APE 
# check reflections
# fft
# make wider sponge layers to avoid reflections

stop
# ====================================================================
# ====================================================================
# ====================================================================






# project velocities/pressures on modes and then compute energetics per mode ------------------------------

# load eigen functions
# ["f", "om2", "zfw", "N2w", "nonhyd", "kn", "Ln", "Cn", "Cgn", "Cen", "Weig", "Ueig", "Ueig2"]
fnameEIG = @sprintf("EIG_amz_%04.1f.jld2",lat) 
path_fname2 = string(dirEIG,fnameEIG);

#=
datafile = jldopen(path_fname2, "r")
println(keys(datafile))  # List the keys (variables) in the file
close(datafile)
=#

# make sure to use the normalized Ueig2!
@load path_fname2 kn Ueig2 zfw N2w
lines(Ueig2[:,1],zc)
sum(Ueig2[:,2].^2 .*dz)/H   # depth-mean = 1


fig = Figure(size=(500,500))
ax = Axis(fig[1, 1],title = "N(z) Amazon", xlabel = "N [rad/s]", ylabel = "z [m]",yticks=(-1000:200:0))
lines!(ax, sqrt.(N2w), zfw, color = :red, linewidth = 3)
ylims!(ax, -1000,0)
fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"N2_AMZ.png"), fig)
end

# Ueig should be a zc
zU = zfw[1:end-1]/2 + zfw[2:end]/2;
Float32.(zU) == zc  # zU is a Float64

# project the first 5 modes on velocities
# un = 1/H*sum(uc*Ueig*dz)
MEIG = 5
un = zeros(Nt,Nx,MEIG);
vn = zeros(Nt,Nx,MEIG);
pn = zeros(Nt,Nx,MEIG);
ucr = copy(uc);
vcr = copy(vc);
pcpr = copy(pcp);
for i in 1:Nx        # x
    for l in 1:Nt            # time
        for m in 1:MEIG
            # #=
            un[l,i,m] = 1/H*sum(uc[l,i,:].*Ueig2[:,m].*dz);   
            vn[l,i,m] = 1/H*sum(vc[l,i,:].*Ueig2[:,m].*dz);               
            pn[l,i,m] = 1/H*sum(pcp[l,i,:].*Ueig2[:,m].*dz);
            ## =#

            #=removing fit does not make a difference
            un[l,i,m] = 1/H*sum(ucr[l,i,:].*Ueig2[:,m].*dz);   
            vn[l,i,m] = 1/H*sum(vcr[l,i,:].*Ueig2[:,m].*dz);               
            pn[l,i,m] = 1/H*sum(pcpr[l,i,:].*Ueig2[:,m].*dz);
            =#
            
            # remove fit for residual
            ucr[l,i,:]  = ucr[l,i,:] - un[l,i,m].*Ueig2[:,m]
            vcr[l,i,:]  = vcr[l,i,:] - vn[l,i,m].*Ueig2[:,m]
            pcpr[l,i,:] = pcpr[l,i,:] - pn[l,i,m].*Ueig2[:,m]
            ## =#
        end
    end
end

# show residual (small)
# depth-integrate
fig1 = Figure(size=(660,800))
ax1a = Axis(fig1[1, 1])
lines!(ax1a,tday,uc[:,115,end])
lines!(ax1a,tday,ucr[:,115,end])
fig1

# some more hovmullers
# no reflections
# mode 2s with mode 1 speed????
clims1 = (-0.2,0.2)
clims2 = (-0.1,0.1)
fig1 = Figure(size=(660,800))
ax1a = Axis(fig1[1, 1])
ax1b = Axis(fig1[2, 1]) 
hm1 = heatmap!(ax1a, xc/1e3, tday, transpose(un[:,:,1]), colormap = Reverse(:Spectral), colorrange = clims1)
hm2 = heatmap!(ax1b, xc/1e3, tday, transpose(un[:,:,2]), colormap = Reverse(:Spectral), colorrange = clims2)
Colorbar(fig1[1, 2], hm1)
Colorbar(fig1[2, 2], hm2)
fig1

# plot mode 1 mode 2 reconstructed field
# reconstruct all modes + residual
umode = zeros(Nt,Nx,Nz);
Im = 1
for i in 1:Nx
    for l in 1:Nt
        umode[l,i,:] = un[l,i,Im].*Ueig2[:,Im]
    end
end

# time series
clims1 = (-0.15,0.15)
fig1 = Figure(size=(800,600))
ax1a = Axis(fig1[1, 1])
hm1 = heatmap!(ax1a, tday, zc, (umode[:,115,:]),colormap = Reverse(:Spectral), colorrange = clims1)
Colorbar(fig1[1, 2], hm1)
fig1

# snapshot in time
clims1 = (-0.15,0.15)
fig1 = Figure(size=(800,600))
ax1a = Axis(fig1[1, 1])
hm1 = heatmap!(ax1a, xc/1e3, zc, (umode[100,:,:]),colormap = Reverse(:Spectral), colorrange = clims1)
Colorbar(fig1[1, 2], hm1)
fig1

# hovmuller
fig1 = Figure(size=(600,800))
ax1a = Axis(fig1[1, 1])
#hm1 = heatmap!(ax1a, tday, transpose(un[:,:,2]), colormap = Reverse(:Spectral), colorrange = clims1)
hm1 = heatmap!(ax1a, xc/1e3, tday, transpose(umode[:,:,end]), colormap = Reverse(:Spectral))
Colorbar(fig1[1, 2], hm1)
fig1


# need to compute the residual variance
# there is very little haha


# time series
fig = Figure()
ax = Axis(fig[1, 1],xlabel = "time [days]", ylabel = "u [m/s]")
lines!(ax, tday, un[:,100,1], color = :black, linewidth = 2)
lines!(ax, tday, un[:,100,2], color = :red, linewidth = 2)
lines!(ax, tday, un[:,100,3], color = :green, linewidth = 2)
lines!(ax, tday, un[:,100,4], color = :orange, linewidth = 2)
fig

ax2 = Axis(fig[2, 1],xlabel = "time [days]", ylabel = "p [N/m2]")
lines!(ax2, tday, pn[:,100,1], color = :black, linewidth = 2)
lines!(ax2, tday, pn[:,100,2], color = :red, linewidth = 2)
lines!(ax2, tday, pn[:,100,3], color = :green, linewidth = 2)
lines!(ax2, tday, pn[:,100,4], color = :orange, linewidth = 2)
fig

# filter variables  ======================================================

Nf = 8;

# undecomposed variables (as a function of depth)

# remove the low frequency motions - if any?
Tcut1 = 16/24;
uc2  = lowhighpass_butter(uc,Tcut1,dt,Nf,"high");
pcp2 = lowhighpass_butter(pcp,Tcut1,dt,Nf,"high");

Tcut2 = 9/24;
uh = lowhighpass_butter(uc2,Tcut2,dt,Nf,"high");
ph = lowhighpass_butter(pcp2,Tcut2,dt,Nf,"high");
ul = uc2 - uh;
pl = pcp2 - ph;

uh2 = lowhighpass_butter(uc,Tcut2,dt,Nf,"high");
ul2 = uc-uh2
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax,tday,ul[:,100,end], color = :black)
lines!(ax,tday,ul2[:,100,end], color = :red)
fig

# modes
un2 = lowhighpass_butter(un,Tcut1,dt,Nf,"high");
pn2 = lowhighpass_butter(pn,Tcut1,dt,Nf,"high");

unh = lowhighpass_butter(un2,Tcut2,dt,Nf,"high");
pnh = lowhighpass_butter(pn2,Tcut2,dt,Nf,"high");
unl = un2 - unh;
pnl = pn2 - pnh;

# time series
fig = Figure()
ax = Axis(fig[1, 1],xlabel = "time [days]", ylabel = "u [m/s]")
lines!(ax, tday, unh[:,10,2], color = :black, linewidth = 2)
lines!(ax, tday, unl[:,10,2], color = :red, linewidth = 2)
fig

# some more hovmullers
# no reflections
clims = (-0.2,0.2)
fig1 = Figure(size=(660,800))
ax1a = fig1[1, 1] 
ax1b = fig1[2, 1] 
#heatmap(ax1a, xc/1e3, tday, transpose(unl[:,:,2]))
#heatmap(ax1b, xc/1e3, tday, transpose(unh[:,:,2]))
#heatmap(ax1a, xc/1e3, tday, transpose(un[:,:,1]), colormap = Reverse(:Spectral), colorrange = clims)
#heatmap(ax1b, xc/1e3, tday, transpose(un[:,:,2]), colormap = Reverse(:Spectral), colorrange = clims)
heatmap(ax1a, xc/1e3, tday, transpose(ul[:,:,end]), colormap = Reverse(:Spectral), colorrange = clims)
heatmap(ax1b, xc/1e3, tday, transpose(uh[:,:,end]), colormap = Reverse(:Spectral), colorrange = clims)
fig1

# fluxes =======================================================

# need to adjust for ringing etc *******************************************************
# need to adjust for ringing etc *******************************************************
EXCL = 4;
t1,t2 = 4, tday[end]-EXCL*T2/24
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

# undecomposed time-mean flux 
Fx  = dropdims(mean(sum(uc2[Iday,:,:].*pcp2[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxh = dropdims(mean(sum(uh[Iday,:,:].*ph[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxl = dropdims(mean(sum(ul[Iday,:,:].*pl[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fx2 = Fxh + Fxl

# modal time-mean flux
Fxn  = dropdims(mean(H*un2[Iday,:,:].*pn2[Iday,:,:],dims=1),dims=1)
Fxnh = dropdims(mean(H*unh[Iday,:,:].*pnh[Iday,:,:],dims=1),dims=1)
Fxnl = dropdims(mean(H*unl[Iday,:,:].*pnl[Iday,:,:],dims=1),dims=1)

# compare sum of modes with undecomposed fluxes
Fxnt  = dropdims(sum(Fxn,dims=2),dims=2)
Fxnht = dropdims(sum(Fxnh,dims=2),dims=2)
Fxnlt = dropdims(sum(Fxnl,dims=2),dims=2)
Fxnt2 = Fxnht + Fxnlt   # should be the same as 

# save the fluxes for plotting in the same figure
fnameout = string("fluxes_",fname_short2,".jld2")

jldsave(string(dirout,fnameout); xc, Fx, Fxh, Fxl, Fx2, Fxn, Fxnh, Fxnl);
println(string(fnameout)," data saved ........ ")


# load and compare the fluxes  =======================================

fnamal = ["AMZ3_hvis_12d_U1_0.40_U2_0.00",  # mode 1
          "AMZ3_hvis_12d_U1_0.40_U2_0.30"]  # mode 1+2

fnamal = ["AMZ3_40.0_hvis_12d_U1_0.40_U2_0.0"]  # mode 1

# load and plot simulations

ylim = [0 7]

fig = Figure(size=(750,500))
ax = Axis(fig[1, 1],title = "mode 1  D2 tidal flux", xlabel = "x [km]", ylabel = "flux [W/m]")
ylims!(ax, ylim[1], ylim[2])

ax2 = Axis(fig[2, 1],title = "mode 1 supertidal flux", xlabel = "x [km]", ylabel = "flux [W/m]")
ylims!(ax2, ylim[1], ylim[2])

xc=0;  Fxl=0;  Fxh=0; Fxnl=0;  Fxnh=0;  
for i in 1:1
    path_fname = string(dirout,"fluxes_",fnamal[i],".jld2")

    @load path_fname xc Fxl Fxh Fxnl  Fxnh  
    if i==1 
        lines!(ax, xc/1e3, Fxnl[:,1]/1e3, label = "sim. mode 1", color = :red, linewidth = 3)
        lines!(ax2, xc/1e3, Fxnh[:,1]/1e3, label = "sim. mode 1", color = :red, linewidth = 3)
    elseif i==2
        lines!(ax, xc/1e3, Fxnl[:,1]/1e3, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :dash)
        lines!(ax2, xc/1e3, Fxnh[:,1]/1e3, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :dash)
    end
end
axislegend(ax, position = :rt)
xlims!(ax, 0, 500)
xlims!(ax2, 0, 500)
fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"flux_mode_hi_lo.png"), fig)
end


# plot total flux
fnamal = ["AMZ3_hvis_12d_U1_0.40_U2_0.00",  # mode 1
          "AMZ3_hvis_12d_U1_0.00_U2_0.30",  # mode 2
          "AMZ3_hvis_12d_U1_0.40_U2_0.30"]  # mode 1+2

          ylim = [0 7]

fig = Figure(size=(750,500))
ax = Axis(fig[1, 1],title = "undecomposed D2 tidal flux", xlabel = "x [km]", ylabel = "flux [W/m]")
ylims!(ax, ylim[1], ylim[2])

ax2 = Axis(fig[2, 1],title = "undecomposed supertidal flux", xlabel = "x [km]", ylabel = "flux [W/m]")
ylims!(ax2, ylim[1], ylim[2])

xc=0;  Fxl=0;  Fxh=0; Fxnl=0;  Fxnh=0;  
Fxls=0;  Fxhs=0; 
for i in 1:3
    path_fname = string(dirout,"fluxes_",fnamal[i],".jld2")

    @load path_fname xc Fxl Fxh  

    Fxls = Fxls .+ Fxl;
    Fxhs = Fxhs .+ Fxh;

    if i==2 
        lines!(ax, xc/1e3, Fxls[:,1]/1e3, label = "sim. mode 1 + sim. mode 1", color = :red, linewidth = 3)
        lines!(ax2, xc/1e3, Fxhs[:,1]/1e3, label = "sim. mode 1 + sim. mode 1", color = :red, linewidth = 3)
    elseif i==3
        lines!(ax, xc/1e3, Fxl[:,1]/1e3, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :dash)
        lines!(ax2, xc/1e3, Fxh[:,1]/1e3, label = "sim. mode 1+2", color = :green, linewidth = 3, linestyle = :dash)
    end
end
axislegend(ax, position = :rt)
xlims!(ax, 0, 500)
xlims!(ax2, 0, 500)
fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"flux_undecomp_hi_lo.png"), fig)
end

#=
ylim = [0 3000]
fig = Figure()
ax = Axis(fig[1, 1],xlabel = "x [km]", ylabel = "Fx [W/m]")
ylims!(ax, ylim[1], ylim[2])
lines!(ax, xc/1e3, Fxn[:,1], color = :black, linewidth = 2)
lines!(ax, xc/1e3, Fxn[:,2], color = :red, linewidth = 2)
lines!(ax, xc/1e3, Fxn[:,3], color = :green, linewidth = 2)
fig
=#


# for WTD seminar ---------------------------------
ylim = [0 7]

fig = Figure(size=(1000,250))
ax = Axis(fig[1, 1],title = titlenm, xlabel = "x [km]", ylabel = "flux [W/m]")
ylims!(ax, ylim[1], ylim[2])
#lines!(ax, xc/1e3, Fxnl[:,1]/1e3, label = "D2 mode 1", color = :black, linewidth = 2)
#lines!(ax, xc/1e3, Fxnh[:,1]/1e3, label = "HH mode 1", color = :red, linewidth = 2)
lines!(ax, xc/1e3, Fxnl[:,2]/1e3, label = "D2 mode 2", color = :black, linewidth = 2)
lines!(ax, xc/1e3, Fxnh[:,2]/1e3, label = "HH mode 2", color = :red, linewidth = 2)

lines!(ax, xc/1e3, Fx/1e3, label = "tot undecom.", color = :green, linewidth = 3, linestyle = :dash)
lines!(ax, xc/1e3, Fxl/1e3, label = "D2 undecom. ", color = :black, linewidth = 2, linestyle = :dash)
lines!(ax, xc/1e3, Fxh/1e3, label = "HH undecom.", color = :red, linewidth = 2, linestyle = :dash)
axislegend(ax, position = :rt)
xlims!(ax, 0, 500)
fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"flux_mode_tot_",fname_short2,".png"), fig)
end


ylim = [0 12000]

fig = Figure(size=(600,800))
ax = Axis(fig[1, 1],title = fname_short2, xlabel = "x [km]", ylabel = "mode 1 Fx [W/m]")
ylims!(ax, ylim[1], ylim[2])
lines!(ax, xc/1e3, Fxnl[:,1], color = :black, linewidth = 2)
lines!(ax, xc/1e3, Fxnh[:,1], color = :red, linewidth = 2)

ax2 = Axis(fig[2, 1],xlabel = "x [km]", ylabel = "mode 2 Fx [W/m]")
ylims!(ax2, ylim[1], ylim[2])
lines!(ax2, xc/1e3, Fxnl[:,2], color = :black, linewidth = 2)
lines!(ax2, xc/1e3, Fxnh[:,2], color = :red, linewidth = 2)

ax3 = Axis(fig[3, 1],xlabel = "x [km]", ylabel = "mode 2 Fx [W/m]")
ylims!(ax3, ylim[1], ylim[2])
lines!(ax3, xc/1e3, Fxnl[:,3], color = :black, linewidth = 2)
lines!(ax3, xc/1e3, Fxnh[:,3], color = :red, linewidth = 2)
fig



fig = Figure()
ax = Axis(fig[1, 1],title = string("total flux",fname_short2), xlabel = "x [km]", ylabel = "total Fx [W/m]")
ylims!(ax, ylim[1], ylim[2])
lines!(ax, xc/1e3, Fxnt, label = "tot", color = :yellow, linewidth = 3)
lines!(ax, xc/1e3, Fxnlt, label = "9-16h", color = :black, linewidth = 3)
lines!(ax, xc/1e3, Fxnht, label = "<9h", color = :red, linewidth = 3)
lines!(ax, xc/1e3, Fxnt2, label = "<16h", color = :blue, linewidth = 3) #
scatterlines!(ax, xc/1e3, Fx, label = "tot", marker = :cross, color = :green, linewidth = 1, linestyle = :dash)
scatterlines!(ax, xc/1e3, Fxl, label = "9-16h", marker = :cross, color = :grey, linewidth = 1, linestyle = :dash)
scatterlines!(ax, xc/1e3, Fxh, label = "<9h", marker = :cross, color = :orange, linewidth = 1, linestyle = :dash)
scatterlines!(ax, xc/1e3, Fx2, label = "<16h", marker = :cross, color = :cyan, linewidth = 1, linestyle = :dash)
axislegend(ax, position = :rt)
fig


# TO DO:
# -adjust for ringing and traveltime!! 
#    => comp. means over shorter time period!
# -increase velocities to match fluxes amazon
#    u surface should be ~0.5 m/s?
# -divergence for undecomposed filtered fields
#    => high-pass divergence should agree with coarsegraining patterns
# -compare sims and their patterns

# large conclusions for 4-km coarse resolution simulations:
# 1) for low velocities decay of mode 1 is the same for diff sims
#    as are integrated energy transfers along transect
# 2) however, spatial patterns are different
#    does this affect mixing? solitary wave formation? 


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


return


# line plots 
Isel = 49; xc[Isel]/1e3 # hotspot
#Isel = 68; xc[Isel]/1e3 # in between hotspots

fig = Figure()
ax = Axis(fig[1, 1], title = "Power Spectrum",xlabel = "Frequency [cpd]", ylabel = "KE",yscale = log10)
lines!(ax, freq, KEom[:,Isel], color = :black, linewidth = 2)
#lines!(ax, freq, KEom1[:,Isel]+KEom2[:,Isel], color = :red, linewidth = 2)
fig





# spectra
# power units y_unit^2*t_unit^2
tukeycf=0.2; numwin=2; linfit=true; prewhit=false;
Pun = zeros(length(period),Nx,MEIG);
Pvn = zeros(length(period),Nx,MEIG);
for m in 1:MEIG
    for i in 1:Nx
        period, freq, Pun[:,i,m] = fft_spectra(tday, un[:,i,m]; tukeycf, numwin, linfit, prewhit);
        period, freq, Pvn[:,i,m] = fft_spectra(tday, vn[:,i,m]; tukeycf, numwin, linfit, prewhit);    
    end
end

# KE per mode as f(x)
KEn = Pun + Pvn;   

# heatmap of spectral power
ylim = [0 8];
clims = (-0.05,0.05)

Im = 2

fig1 = Figure()
axa = Axis(fig1[1, 1],title=string("KE [m^2/s^2] mode ",Im));  
#ylims!(axa, ylim[1], ylim[2])
hm = heatmap!(axa, xc/1e3, freq, log10.(transpose(KEn[:,:,Im])), colormap = Reverse(:Spectral)); 
Colorbar(fig1[1,2], hm); 
fig1   

# PLOT PER FREQUENCY BAND
I2 = findall(x->x>24/T2-0.5 && x<24/T2+0.5, freq)
freq[I2]

I4 = findall(x->x>2*24/T2-0.5 && x<2*24/T2+0.5, freq)
freq[I4]

I6 = findall(x->x>3*24/T2-0.5 && x<3*24/T2+0.5, freq)
freq[I6]

# sum over freqs
df = freq[2]-freq[1]
KEnf = zeros(3,Nx,MEIG)
for k in 1:3
    if     k==1; II=I2
    elseif k==2; II=I4        
    elseif k==3; II=I6
    end                
    for i in 1:Nx
        for m in 1:MEIG
            KEnf[k,i,m] = sum(Pun[II,i,m] + Pvn[II,i,m])*df  # unit of m2/s2
        end
    end
end


fig = Figure(size = (600, 800))
for Im=1:3
    if Im==1; titstr = string(fname_short2,"; mode ",Im)
    else;     titstr = string("mode ",Im)
    end
    ax = Axis(fig[Im, 1], title = titstr, xlabel = "x [km]", ylabel = "P [m2/s2]", yscale = log10)
    ylims!(ax, (1e-8, 1e-2))
    lines!(ax, xc/1e3, KEnf[1,:,Im], color = :black, linewidth = 2, label = "M2")
    lines!(ax, xc/1e3, KEnf[2,:,Im], color = :red, linewidth = 2, label = "M4")
    lines!(ax, xc/1e3, KEnf[3,:,Im], color = :green, linewidth = 2, label = "M6")
    axislegend(ax, position = :rb)
end
fig




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
