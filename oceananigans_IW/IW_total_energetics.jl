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
using Interpolations
using Trapz


WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
    dirforce = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirfig = string(pth0,"figs/");
    dirout = string(pth0,"diagout/");
    dirforce = string(pth0,"IW/forcingfiles/");
end

include(string(pathname,"include_functions.jl"))

# print figures
figflag = 1
oldnm   = 0  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing
const T2 = 12+25.2/60
const rho0=1020; 
const grav=9.81; 

#=
#      38 39 40 41 42 43 44 45 46 47 48    49
mainnm = 1
LATS = [0 2.5 5 10 15 20 25 30 40 50 28.80 35];
runnms = [38 39 40 41 42 43 44 45 46 47 48 49];
#runnms = [1]
=#

#= D2
mainnm = 3
LATS   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 27.5, 30.0, 35.0, 40.0, 50.0];
runnms = [3,   4,   5,   6,    7,    8,    9,    10,   11,   12,   13,   14];
=#

#= D1
mainnm = 5
LATS   = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
runnms = [9,   10,  11,  12,  13,   14,   15,   16,   17,   18]  # is the same
=#

# one run
mainnm = 3
LATS   = [0]
runnms = [16] 
#

# for runnm in runnms  # runnms loop ---------------
    runnm = runnms[1]
        println("runnm is ",runnm) 

# file name ===========================================

if oldnm==1
    # function of latitude
    lat = 0

   #fnames = @sprintf("AMZv_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"
   # fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"  # hydro
   # fnames = @sprintf("AMZ4_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"     # nonhydro
   fnames = "AMZ4_00.0_hvis_12d_U1_0.40_U2_0.30_v3.nc"; titlenm = "mode 1+2"  # nonhydro and compare with

    fname_short2 = fnames[1:33]
    filename = string(dirsim,fnames)

    LAT = LATS[1];
else
    # file ID
    #runnm  = 47

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")

    LAT = LATS[runnm-minimum(runnms)+1];
    println("lat is ",LAT,"------------------------------------") 

end

# load simulations ===========================================

ds = NCDataset(filename,"r");
#println(ds)
println(keys(ds))

tsec = ds["time"][:];
tday = tsec/24/3600;
dt = tday[2]-tday[1]

xf   = ds["x_faa"][:]; 
xc   = ds["x_caa"][:]; 
zc   = ds["z_aac"][:]; 

dx   = ds["Δx_caa"][:];
dz   = ds["Δz_aac"][:];
Ldom = sum(dx);

H  = sum(dz);   # depth

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# u, v, w velocities
# NOTE: in future select a certain x range away from boundaries
@time begin
    println("reading nc file ",filename)
    uf  = permutedims(ds["u"][:,:,:],    (3,1,2))
    vc  = permutedims(ds["v"][:,:,:],    (3,1,2))
    wf  = permutedims(ds["w"][:,:,:],    (3,1,2))
    bc  = permutedims(ds["b"][:,:,:],    (3,1,2))
    pHY = permutedims(ds["pHY"][:,:,:],  (3,1,2))
    pNH = permutedims(ds["pNHS"][:,:,:], (3,1,2))
end

#= a loop is slower than the above w. permuting?
@time begin
uf = zeros(Nt,Nx+1,Nz);
vc = zeros(Nt,Nx,Nz);
wf = zeros(Nt,Nx,Nz+1);
bc = zeros(Nt,Nx,Nz);
pHY = zeros(Nt,Nx,Nz);
pNH = zeros(Nt,Nx,Nz);

for i in 1:Nt
    if rem(i,100)==0; println(i); end
    uf[i,:,:] = ds["u"][:, :, i];
    vc[i,:,:] = ds["v"][:, :, i];
    wf[i,:,:] = ds["w"][:, :, i];
    bc[i,:,:] = ds["b"][:, :, i];
    pHY[i,:,:] = ds["pHY"][:, :, i];    
    pNH[i,:,:] = ds["pNHS"][:, :, i];        
end
end
=#


# total pressure in N/m2
ptot = (pHY.+pNH)*rho0/grav;
#ptot = (pHY)*rho0/grav;
#ptot = (pNH)*rho0/grav;    # NH pressure only

# clear variables from memory
pHY= nothing; pNH = nothing; 
GC.gc()

# close the nc file
close(ds)

# compute at cell centers
# v is already at x,W centers
uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 
wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 

# clear variables from memory
uf= nothing; wf = nothing; 
GC.gc()

# plot surface velocity to check effect of resolution
idx, d = nearest_index(xc, 369e3)
fig1 = Figure(size=(600,600))
ax = Axis(fig1[1, 1],limits = (200, 500, nothing, nothing),title = string(fname_short2,"; lat=",LAT,"; u [m/s]"), xlabel = "x [km]", ylabel = "u [m/s]")
lines!(ax,xc/1e3,uc[end,:,1])
scatter!(ax, [xc[idx]/1e3], [uc[end, idx, 1]], color = :red, markersize = 12)
fig1


# some more hovmullers
fig1 = Figure(size=(600,600))
clims =(0,0.8)
ax = Axis(fig1[1, 1],title = string(fname_short2,"; lat=",LAT,"; KE [m2/s2]"), xlabel = "x [km]", ylabel = "time [days]")
hm = heatmap!(ax, xc/1e3, tday, transpose(uc[:,:,end].^2 .+ vc[:,:,end].^2), colormap = Reverse(:Spectral), colorrange = clims); Colorbar(fig1[1,2], hm); 
#hm = heatmap!(ax, xc/1e3, tday, transpose(uc[:,:,end].^2), colormap = Reverse(:Spectral)); Colorbar(fig1[1,2], hm); 
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"KE_hovmuller_", fname_short2 ,".png"), fig1)
end

#stop()

# load N2 profile -----------------------------------------------------------
# load profile created by AMZ_stratification_profile.jl
fnamegrid = "N2_amz1.jld2";
path_fname = string(dirforce,fnamegrid);

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

#= APE linear and nonlinear
# b = -g/rho0*rho_pert [m/s2]
# 1/2*rho0*b2/N2 [J/m3 = Nm/m3 = kg*m2/s2/m3]
#  [kg/m3 * m2/s4 * s2 =         kg*m2/s2/m3]

dzz = reshape(dz,1,1,:); 

# omit N2c values <= 1e-10, keep the others
Ikp = findall(>(1e-10),N2c);
N2cc = reshape(N2c,1,1,:);
factA = 1/2*rho0*1.0./N2cc;

APE3z = zeros(Nt,Nx,Nz);
APE3z[:,:,Ikp] = (bc[:,:,Ikp].^2).*factA[:,:,Ikp];
APE3  = sum((bc[:,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3);

# nonlinear addition
dN2dz  = diff(N2w) ./ diff(zfw)   # length Nz, lives on zc grid
dN2dzz = reshape(dN2dz,1,1,:);
factB  = .- 1/6*rho0 .* dN2dzz ./ N2cc.^3;

APE3nlz = copy(APE3z)
APE3nlz[:,:,Ikp] = APE3z[:,:,Ikp] .+ factB[:,:,Ikp] .* bc[:,:,Ikp].^3;
=#


# ref and pert densities ----------------------------------------------------
# compute reference density profile
# b = sum N2 * dz = sum -g/rho0*drho/dz * dz
# b = -g/rho0*rho_pert
# rho_pert = -b*rho0/g 

# bottom up!
breff = cumtrapz(zfw, N2w);   # length Nz+1, on zfw grid

# in the mod sims buoyancy is interpolated to cell centers
intzc   = interpolate((zfw,), breff, Gridded(Linear()));
rhorefc = -intzc.(zc) * rho0/grav;     # rho0 is not added!
rr      = reshape(rhorefc, 1, 1, :);
rhop    = -bc * rho0/grav .+ rr;  

#=
idx, d = nearest_index(xc, 1480e3)

# find vertical displacement xi
#it = 700 #xi down
#it = 718 #xi up
it = 740 #xi up
rhops = rhop[it,idx,:]  #xi is up

# map rhops values out of rhorefc range to local rhorefc
# this is due to machine errors
# near surface
Isel = rhorefc[end] .- rhops  .> 0 
rhops[Isel] = rhorefc[Isel]

# and near bottom
Isel = rhorefc[1] .- rhops  .< 0 
rhops[Isel] = rhorefc[Isel]

# exclude small differences to avoid weird extrapolations
Isel = abs.(rhorefc .- rhops)  .> 1/1e5 
Ilp = collect(1:Nz)
Ilp = Ilp[Isel]

# interpolate zstar along rhoref @ rhops
# minus sign is to accomodate positive increase 
itp   = interpolate((-rhorefc,), zc, Gridded(Linear()))
intrc = extrapolate(itp, Line())

zs = copy(zc)
zs[Isel] = intrc.(-rhops[Isel]) 

# vertical displacement (+ is upward)
xi = zc - zs

# loop over depth and non-zero xi
APE = zeros(size(zc))
for i in Ilp
    zstar = zs[i]  # location of rho on rhoref; z-xi
    zrho  = zc[i]  # peturbation density rho  
    if zrho < zstar      # xi<0; down
        Is = findall(zrho .< zc .< zstar)
        zz = [zrho; zc[Is]; zstar]        
        rr = rhops[i] .- [rhorefc[i] ;rhorefc[Is]; rhops[i]] 
        fac = -1
    elseif zrho > zstar  # xi>0; up
        Is = findall(zstar .< zc .< zrho)
        zz = [zstar; zc[Is]; zrho]        
        rr = rhops[i] .- [rhops[i] ;rhorefc[Is]; rhorefc[i]] 
        fac = 1        
    end
    APE[i] = fac * grav * trapz(zz,rr)
end
=#

# function to compute APE as in Kang And Fringer (2010) --------------------
# This is Claudformed code based on the expensive loop APE and its preamble

# if rhop-rhorefc > thresh, then APE  = 0
thresh = 1e-5;
function APEKFeq2(rhop, rhorefc, zc, grav, thresh)
    # suggested value thresh = 1e-5;
    Nt, Nx, Nz = size(rhop)
    APE        = zeros(Nt, Nx, Nz)

    itp_zs = extrapolate(interpolate((-rhorefc,), zc, Gridded(Linear())), Flat())
    F_ref  = cumtrapz(zc, rhorefc)
    rho_lo = rhorefc[end]   # lightest (surface)
    rho_hi = rhorefc[1]     # densest  (bottom)

    # exact cumulative integral of piecewise-linear rhorefc from zc[1] to z
    function exact_F(z)
        j = searchsortedfirst(zc, z)
        if j == 1
            return F_ref[1] + rhorefc[1] * (z - zc[1])
        elseif j > length(zc)
            return F_ref[end] + rhorefc[end] * (z - zc[end])
        else
            dz = zc[j] - zc[j-1]
            dt = z - zc[j-1]
            return F_ref[j-1] + rhorefc[j-1]*dt + (rhorefc[j]-rhorefc[j-1])/(2*dz) * dt^2
        end
    end

    Threads.@threads for it in 1:Nt
        for ix in 1:Nx
            for i in 1:Nz
                rho_i = rhop[it, ix, i]
                (rho_i < rho_lo || rho_i > rho_hi) && continue
                abs(rhorefc[i] - rho_i) < thresh    && continue

                zrho  = zc[i]
                zstar = itp_zs(-rho_i)
                z1    = min(zrho, zstar)
                z2    = max(zrho, zstar)
                fac   = zrho > zstar ? 1.0 : -1.0

                APE[it, ix, i] = fac * grav * (rho_i*(z2-z1) - (exact_F(z2) - exact_F(z1)))
            end
        end
    end
    return APE
end

# APE_4D = APEKFeq2(rhop, rhorefc, zc, grav, thresh);

#= compare the performance of the various APEs
fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
lines!(ax1,APE3z[it,idx,:], zc,   color = :red,   label = "APE3z")
lines!(ax1,APE, zc,               color = :black, label = "APE")
lines!(ax1,APE3nlz[it,idx,:], zc, color = :green, label = "APE3nlz")
lines!(ax1,APE_4D[it,idx,:], zc, color = :orange, label = "APE_4D", linestyle = :dash)
axislegend(ax1, position = :rb)

ax2 = Axis(fig[1,2])
lines!(ax2,APE .- APE3z[it,idx,:], zc,   color = :red,   label = "APE3z")
lines!(ax2,APE .- APE3nlz[it,idx,:], zc, color = :green, label = "APE3nlz")
lines!(ax2,APE .- APE_4D[it,idx,:], zc, color = :orange, label = "APE_4D")
axislegend(ax2, position = :rb)
#limits!(ax1, nothing, nothing,-300, 0)
fig
=#


# compute pressure -----------------------------------------------------------

# rho_pert = -b*rho0/g 
# dp       = -g*rho*dz
# dp/dz    = b*rho0

# In Oceananigans: dpk/dz = b = -g/rho0*rho_pert [m/s2]
#                  [m2/s2*1/m] = [m/s2]               
# units b: [m/s2]
# because of kinematic pressure pk = p/rho0 = kg*m2/s2*1/m3 *m3/kg = m2/s2

# hydrostatic p from b compares well with pHY*rho0/grav
# ptot = (pHY+pNH)*rho0/grav
# remove time-mean and then depth-mean

# remove time-mean
ptota = mean(ptot,dims=1);
ptotp = ptot .- ptota;

# remove depth-mean
dzz = reshape(dz,1,1,:);                                    # shape: (1, length(zc), 1)
ptotpa  = sum(ptotp.*dzz,dims=3)/H; # depth-mean pressure
pcp = ptotp .- ptotpa;              # the perturbation pressure!

#check integral of perturbation pressure should be zero 
sum(pcp[200,50,:].*dz)   
Figure(); lines(tday, pcp[:,50,end])

N2cc = reshape(N2c,1,1,:);

# compute density

#=
# indirect method, which compares well with pHY
# hydrostatic pressure
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


# compare diagnosed hydrostatic pressure, with b based pressure
# pHY needs to  be multiplied with rho0 / grav

fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
#limits!(ax1, nothing, nothing, -200, 0)
lines!(ax1,pHY[end,idx,:]*rho0/grav, zc, color = :black)
lines!(ax1,pNH[end,idx,:]*rho0/grav, zc, color = :blue)
lines!(ax1,(pHY[end,idx,:].+pNH[end,idx,:])*rho0/grav, zc, color = :magenta)
lines!(ax1,pc[end,idx,:], zc, color = :red)
lines!(ax1,(pHY[end,idx,:]*rho0/grav .- pc[end,idx,:])./pc[end,idx,:]*100, zc, color = :green)
fig

# plot NH pressure
fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
#limits!(ax1, nothing, nothing, -200, 0)
lines!(ax1,xc/1e3,pNH[end,:,1]*rho0/grav, color = :blue)
lines!(ax1,xc/1e3,pHY[end,:,1]*rho0/grav, color = :red)
lines!(ax1,xc/1e3,(pHY[end,:,1].+pNH[end,:,1])*rho0/grav, color = :magenta)

#lines!(ax1,xc/1e3,pNH[end,:,end]*rho0/grav, color = :red)

ax2 = Axis(fig[2,1])
lines!(ax2,xc/1e3,pNH[end,:,end]*rho0/grav, color = :blue)
lines!(ax2,xc/1e3,pHY[end,:,end]*rho0/grav, color = :red)
lines!(ax2,xc/1e3,(pHY[end,:,end].+pNH[end,:,end])*rho0/grav, color = :magenta)
fig

# check integral of pHY and pNH
# depth-integral has horizontal structure ...
ptot = pHY.+pNH;

# remove time-mean
ptota = mean(ptot,dims=1);
ptotp = ptot .- ptota;

# check time-mean profile
fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
#limits!(ax1, nothing, nothing, -200, 0)
lines!(ax1,dropdims(ptota[:,idx,:],dims=1)*rho0/grav, zc, color = :black)
fig
#pHNH = pHY;

# check x-integral
ptotix = dropdims(sum(ptotp,dims=2),dims=2);

# remove depth-integral (not really needed)
ptotpp = ptotp .- sum(ptotp,dims=2)

fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
lines!(ax1,ptotix[end,:]*rho0/grav, zc, color = :black)
fig

# check depth-integral
sss = dropdims(sum(ptotpp.*dzz, dims=3), dims=3)
fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
lines!(ax1,xc/1e3,sss[end,:], color = :magenta)
fig
=#

#stop()

# filter variables  ======================================================
# t is tidal 
# l is tidal+sub
# h is supertidal
# s is subtidal

# Nf = 6;
Nf = 8;

# remove the low frequency motions - if any?
if mainnm == 3     # D2
    Tcut1 = 16/24  #D2+HH
    Tcut2 = ( (T2+2*T2/3)/2 )/24 #day; HH M2-D3 = 10.35 h
    #Tcut2 = (T2+T2/2)/2/24      #day; HH M2-M4 = 9.315 h
elseif mainnm == 5 # D1
    Tcut1 = 30/24
    Tcut2 = 20/24
end

# only filter high-pass motions
# then remove hp from total
# we are ignoring the generation of mean flows
# for the 4 km cases
uca = dropdims(mean(uc,dims=1), dims=(1))

fig1 = Figure(size=(1000,750))
axa = Axis(fig1[1, 1],title="mean flow [m/s]",xlabel="fx [km]",ylabel="z [m]");  
hm = heatmap!(axa, xc/1e3, zc, uca,colormap = Reverse(:Spectral)); 
Colorbar(fig1[1,2], hm); 
hm.colorrange = (-0.02, 0.02)
fig1 

# high-pass the total
# dt in days
passflg = "high";
uh = lowhighpass_butter(uc,Tcut2,dt,Nf,passflg);
vh = lowhighpass_butter(vc,Tcut2,dt,Nf,passflg);
wh = lowhighpass_butter(wc,Tcut2,dt,Nf,passflg);
ph = lowhighpass_butter(pcp,Tcut2,dt,Nf,passflg);
bh = lowhighpass_butter(bc,Tcut2,dt,Nf,passflg);
rh = -bh*rho0/grav

# isolate "l" tidal-band frequencies + mean flows (small)
ul = uc .- uh 
vl = vc .- vh
wl = wc .- wh
pl = pcp .- ph
bl = bc .- bh
#rl = -bl*rho0/grav

# low-pass the total to get subtidal "s"
passflg = "low";
us = lowhighpass_butter(uc,Tcut1,dt,Nf,passflg);
vs = lowhighpass_butter(vc,Tcut1,dt,Nf,passflg);
ws = lowhighpass_butter(wc,Tcut1,dt,Nf,passflg);
ps = lowhighpass_butter(pcp,Tcut1,dt,Nf,passflg);
bs = lowhighpass_butter(bc,Tcut1,dt,Nf,passflg);
#rs = -bs*rho0/grav

# isolate "t" tidal-band frequencies
ut = ul .- us
vt = vl .- vs
wt = wl .- ws
pt = pl .- ps
bt = bl .- bs
rt = -bt*rho0/grav

#= plot bpassed vels
idx, d = nearest_index(xc, 1000e3)
fig = Figure(size=(800,400))
ax = Axis(fig[1, 1])
lines!(ax, tday, uc[:,idx,end], label = "uc", linestyle=:solid, color = :black, linewidth = 2)
lines!(ax, tday, ut[:,idx,end], label = "ut", linestyle=:solid, color = :red, linewidth = 4)
lines!(ax, tday, ul[:,idx,end], label = "ul", linestyle=:solid, color = :cyan, linewidth = 1)
lines!(ax, tday, uh[:,idx,end], label = "uh", linestyle=:solid, color = :green, linewidth = 1)
xlims!(ax,15,20)
fig
=#

ul=nothing; vl=nothing; wl=nothing; wl=nothing; pl=nothing; bl=nothing;
GC.gc()

# KE and APE ----------------------------------------------------------------

# cycles to exclude
EXCL = 2;
t1,t2 = 14, tday[end]-EXCL*T2/24
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

# time-mean, depth-intgr. KE energy 
fact = 1/2*rho0
KE  = fact*dropdims(mean(sum((uc[Iday,:,:].^2 .+ vc[Iday,:,:].^2 .+ wc[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEt = fact*dropdims(mean(sum((ut[Iday,:,:].^2 .+ vt[Iday,:,:].^2 .+ wc[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEh = fact*dropdims(mean(sum((uh[Iday,:,:].^2 .+ vh[Iday,:,:].^2 .+ wh[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEs = fact*dropdims(mean(sum((us[Iday,:,:].^2 .+ vs[Iday,:,:].^2 .+ ws[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))

# time-mean, depth-intgr. APE energy (Kang & Fringer, 2010 eq2)
APEz = APEKFeq2(rhop[Iday,:,:], rhorefc, zc, grav, thresh);
APE  = dropdims(mean(sum( APEz .* dzz,dims=3),dims=1), dims=(1,3));
APEz = APEKFeq2(rt[Iday,:,:], rhorefc, zc, grav, thresh);
APEt = dropdims(mean(sum( APEz .* dzz,dims=3),dims=1), dims=(1,3));
APEz = APEKFeq2(rh[Iday,:,:], rhorefc, zc, grav, thresh);
APEh = dropdims(mean(sum( APEz .* dzz,dims=3),dims=1), dims=(1,3));
APEz = APEKFeq2(rs[Iday,:,:], rhorefc, zc, grav, thresh);
APEs = dropdims(mean(sum( APEz .* dzz,dims=3),dims=1), dims=(1,3));

APEz=nothing;
GC.gc()

# undecomposed time-mean flux 
Fx  = dropdims(mean(sum(uc[Iday,:,:].*pcp[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxt = dropdims(mean(sum(ut[Iday,:,:].*pt[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxh = dropdims(mean(sum(uh[Iday,:,:].*ph[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxs = dropdims(mean(sum(us[Iday,:,:].*ps[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))

# create some figures ----------------------------------------------

#ylimE = [0 75]; ylimA = [0 75];
ylimE = [0 15]; ylimA = [0 15]; ylimf = [-2 7];

fig = Figure(size=(750,750))
ax = Axis(fig[1, 1],title = string(fname_short2,"; lat=",LAT,"; KE [kJ/m2]"), xlabel = "x [km]", ylabel = "KE [kJ/m2]")
lines!(ax, xc/1e3, KE[:,1]/1e3, label = "tot", linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax, xc/1e3, KEt[:,1]/1e3, label = "tidal", color = :red, linewidth = 3)
lines!(ax, xc/1e3, KEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)
lines!(ax, xc/1e3, KEs[:,1]/1e3, label = "sub", color = :yellow, linewidth = 2)
xlims!(ax, 0, Ldom/1e3)
ylims!(ax, ylimE[1], ylimE[2])

ax2 = Axis(fig[2, 1],title = "APE", xlabel = "x [km]", ylabel = "APE [kJ/m2]")
lines!(ax, xc/1e3, APE[:,1]/1e3, label = "tot", linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax, xc/1e3, APEt[:,1]/1e3, label = "tidal", color = :red, linewidth = 3)
lines!(ax, xc/1e3, APEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)
lines!(ax, xc/1e3, APEs[:,1]/1e3, label = "sub", color = :yellow, linewidth = 2)
xlims!(ax2, 0, Ldom/1e3)
ylims!(ax2, ylimA[1], ylimA[2])

ax3 = Axis(fig[3, 1],title = "flux", xlabel = "x [km]", ylabel = "flux [W/m]")
lines!(ax, xc/1e3, Fx[:,1]/1e3, label = "tot", linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax, xc/1e3, Fxt[:,1]/1e3, label = "tidal", color = :red, linewidth = 3)
lines!(ax, xc/1e3, Fxh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)
lines!(ax, xc/1e3, Fxs[:,1]/1e3, label = "sub", color = :yellow, linewidth = 2)
xlims!(ax3, 0, Ldom/1e3)
ylims!(ax3, ylimf[1], ylimf[2])
axislegend(ax3, position = :rt)
fig

# Save the figure as a PNG file
#if figflag==1; save(string(dirfig,"KE_flux_", fname_short2 ,"_NHONLY_v4.png"), fig)
#if figflag==1; save(string(dirfig,"KE_flux_", fname_short2 ,"_HYONLY_v3.png"), fig)
if figflag==1; save(string(dirfig,"KE_flux_", fname_short2 ,".png"), fig)
end

# stop()

println(fnames,"; max total flux is ",@sprintf("%5.2f",maximum(Fxt/1e3))," kW/m")
println(fnames,"; max D2+HH flux is ",@sprintf("%5.2f",maximum(Fx/1e3))," kW/m")



# compute some ffts of surface velocity ======================================================

EXCL = 0;  # can be zero for fft
t2 = tday[end]-EXCL*T2/24; #t1 is defined above
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

tukeycf=0.2; numwin=1; linfit=true; prewhit=false;

i=1;
period, freq, pp = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit); #get the dimensions
poweru = zeros(length(period),Nx);
powerv = zeros(length(period),Nx);
for i in 1:Nx
    period, freq, poweru[:,i] = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);
    period, freq, powerv[:,i] = fft_spectra(tday[Iday], vc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);    
end

println("max freq: ",freq[end]," cpd")

KEom = poweru .+ powerv;    # mode 1+2

# heatmap of spectral power
flim = [0 20]; fstp=2;
clims = (-0.05,0.05)

fig1 = Figure(size=(750,1000))
axa = Axis(fig1[1, 1],xticks = (flim[1]:fstp:flim[2]),
title=string(fname_short2,"; lat=",LAT,"; log10(KE) [m2/s2/cpd] "),xlabel="frequency [cpd]",ylabel="x [km]");  
xlims!(axa, flim[1], flim[2])
hm = heatmap!(axa, freq, xc/1e3, log10.(KEom),colormap = Reverse(:Spectral)); 
Colorbar(fig1[1,2], hm); 
hm.colorrange = (-6, 0)
fig1   

# average coefficients fall inside 75-500 km range
xlims = [0,1000]*1e3; # excl. generation on left and relaxation on right
#xlims = [75,500]*1e3; # excl. generation on left and relaxation on right
#xlims = [0,480]*1e3; # AMZ1 set up with specified boundaries excl. 20 km relaxation on right
Ix = findall(item -> item >= xlims[1] && item<= xlims[2], xc);
KEoma = vec(mean(KEom[:,Ix],dims=2)); 

# store the max M2 amplitude at the forcing boundary
# use for normalization
KEommax, maxidx = findmax(KEom)
#KEommax = KEom[Im2,Ix[1]]
println("KEomax=",log10(KEommax))


Plims = [-12 1];
axb = Axis(fig1[2, 1],xticks = (flim[1]:fstp:flim[2]),
    title="normalized power",xlabel="frequency [cpd]",ylabel="log10(KE/KEmax)");  
xlims!(axb, flim[1], flim[2])
ylims!(axb, Plims[1], Plims[2])
lines!(axb, freq, log10.(KEoma./KEommax), linestyle=:solid, color = :black, linewidth = 3)

# add coriolis rad/s => cpd
fcpd = coriolis(LAT)/(2*pi)*24*3600
#lines!(vec([fcpd fcpd]),vec([minimum(log10.(KEoma)) maximum(log10.(KEoma))]), linestyle=:dash, color = :red, linewidth = 2)
lines!(axb, vec([fcpd fcpd]),vec(Plims), linestyle=:dash, color = :red, linewidth = 2)
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"fft_usur_",fname_short2,".png"), fig1)
end


# save the energy terms =========================================
fnameout = string("energetics_",fname_short2,".jld2")

jldsave(string(dirout,fnameout); freq, KEoma, KEommax, xc, Fxt, Fx, Fxh, Fxl, Fx2, KEt, APEt, KE, APE, 
       KEh, APEh, KEl, APEl, KE2, KEut, KEu, KEuh, KEul);
println(string(fnameout)," data saved ........ ")

# end # runnms loop ---------------


stop()


# ====================================================================
# ====================================================================
# ====================================================================

# load some data and plot spectra
flim = [0 24]; fstp=2;
Plims = [-8 0];

# hydrostatic
fnamein = string(dirout,"energetics_AMZ3_00.0_hvis_12d_U1_0.40_U2_0.0.jld2")

#    gridfile = jldopen(fnamein, "r")
#    println(keys(gridfile))  # List the keys (variables) in the file
#    close(gridfile)

@load fnamein freq  KEoma  KEommax

fig1 = Figure()
ax1 = Axis(fig1[1, 1],xticks = (flim[1]:fstp:flim[2]),
title=L"power normalized by M$_2$ forcing energy KE0",xlabel=L"$\omega$ [cpd]",ylabel=L"$\log_{10}$(KE/KE0)");  
xlims!(ax1, flim[1], flim[2])
ylims!(ax1, Plims[1], Plims[2])
lines!(ax1, freq, log10.(KEoma./KEommax), linestyle=:solid, color = :black, linewidth = 3,label="4 km")

# nonhydrostatic
fnamein = string(dirout,"energetics_AMZ4_00.0_hvis_12d_U1_0.40_U2_0.0.jld2")
@load fnamein freq  KEoma  KEommax
lines!(ax1, freq, log10.(KEoma./KEommax), linestyle=:solid, color = :red, linewidth = 3,label="200 m")
axislegend(ax1, position = :lb)
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"fft_KE_hyd_nonhyd.png"), fig1)
end


# project velocities/pressures on modes and then compute energetics per mode ------------------------------

# load eigen functions
# ["f", "om2", "zfw", "N2w", "nonhyd", "kn", "Ln", "Cn", "Cgn", "Cen", "Weig", "Ueig", "Ueig2"]
fnameEIG = @sprintf("EIG_amz_%04.1f.jld2",lat) 
path_fname2 = string(dirforce,fnameEIG);

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
ax = Axis(fig[1, 1],title = string(" lat=",LAT, " mode 1  D2 tidal flux"), xlabel = "x [km]", ylabel = "flux [W/m]")
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
xlims!(ax, 0, Ldom/1e3)
xlims!(ax2, 0, Ldom/1e3)
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
xlims!(ax, 0, Ldom/1e3)
xlims!(ax2, 0, Ldom/1e3)
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
xlims!(ax, 0, Ldom/1e3)
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
Nb2z = Nb^2 .* reshape(zc, 1, :, 1);    # shape: (1, length(zc), 1)
rho  = -(Nb2z .+ b) * rho0 / grav;      # broadcast without repeat
#rhor  = -(Nb2z) * rho0 / grav;          # reference density
#rho = @. -(Nb2z + b) * rho0 / grav;    # broadcast without repeat

it = 350
fig = Figure(); Axis(fig[1,1],title="b & ρ"); 
heatmap!(xc/1e3,zc,b[:,:,it]); 
contour!(xc/1e3,zc,rho[:,:,it], color = :black); fig

Figure(); 
lines(rho[10,:,100],zc)
#lines(rhor[10,:,100],zc)

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




stop()

# old filter and KE/APE code -------------------------------
# old filter and KE/APE code -------------------------------
# old filter and KE/APE code -------------------------------

# remove the low frequency motions - if any?
if mainnm == 3     # D2
    Tcut1 = 16/24  #low
    Tcut2 = 9/24   #high
elseif mainnm == 5 # D1
    Tcut1 = 30/24
    Tcut2 = 20/24
end

uc2  = lowhighpass_butter(uc,Tcut1,dt,Nf,"high"); # all tidal+supertidal
vc2  = lowhighpass_butter(vc,Tcut1,dt,Nf,"high");
wc2  = lowhighpass_butter(wc,Tcut1,dt,Nf,"high");
pcp2 = lowhighpass_butter(pcp,Tcut1,dt,Nf,"high");
bc2  = lowhighpass_butter(bc,Tcut1,dt,Nf,"high");
# isolate the subtidal flows
ull = uc - uc2;
vll = vc - vc2;

# remove high freq from tidal freq
# high
uh = lowhighpass_butter(uc2,Tcut2,dt,Nf,"high");
vh = lowhighpass_butter(vc2,Tcut2,dt,Nf,"high");
wh = lowhighpass_butter(wc2,Tcut2,dt,Nf,"high");
ph = lowhighpass_butter(pcp2,Tcut2,dt,Nf,"high");
bh = lowhighpass_butter(bc2,Tcut2,dt,Nf,"high");

# l refers to tidal
ul = uc2 - uh;
vl = vc2 - vh;
wl = wc2 - wh;
pl = pcp2 - ph;
bl = bc2 - bh;


# filtered KE, APE, and fluxes =======================================================

# cycles to exclude
EXCL = 4;
t1,t2 = 4, tday[end]-EXCL*T2/24
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

# undecomposed time-mean KE energy 
# KEt is total, unfiltered
# KE  is D2+HH  filtered at once
# KE2 = KEh + KEl is the sum
# KEh is HH
# KEl is D2
fact = 1/2*rho0
KEt = fact*dropdims(mean(sum((uc[Iday,:,:].^2  .+ vc[Iday,:,:].^2  .+ wc[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KE  = fact*dropdims(mean(sum((uc2[Iday,:,:].^2 .+ vc2[Iday,:,:].^2 .+ wc2[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEh = fact*dropdims(mean(sum((uh[Iday,:,:].^2  .+ vh[Iday,:,:].^2  .+ wh[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEl = fact*dropdims(mean(sum((ul[Iday,:,:].^2  .+ vl[Iday,:,:].^2  .+ wl[Iday,:,:].^2).*dzz,dims=3),dims=1), dims=(1,3))
KEll = KEt - KE;

#KEut = fact*dropdims(mean(sum(uc[Iday,:,:].^2  .*dzz,dims=3),dims=1), dims=(1,3))
#KEu  = fact*dropdims(mean(sum(uc2[Iday,:,:].^2 .*dzz,dims=3),dims=1), dims=(1,3))
#KEuh = fact*dropdims(mean(sum(uh[Iday,:,:].^2  .*dzz,dims=3),dims=1), dims=(1,3))
#KEul = fact*dropdims(mean(sum(ul[Iday,:,:].^2  .*dzz,dims=3),dims=1), dims=(1,3))

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

# nonlinear addition eq3 Kang and Fringer 2010
dN2dz  = diff(N2w) ./ diff(zfw)   # length Nz, lives on zc grid
dN2dzz = reshape(dN2dz,1,1,:);
factB  = .- 1/6*rho0 .* dN2dzz ./ N2cc.^3;

# theoretical
APEtnl  = APEt .+ dropdims(mean(sum(factB[:,:,Ikp] .* bc[:,:,Ikp].^3 .* dzz[:,:,Ikp],dims=3),dims=1), dims=(1,3))

# precise
APENL = APEKFeq2(rhop, rhorefc, zc, grav, thresh);
APEtnl2 = dropdims(mean(sum( APENL[:,:,Ikp] .* dzz[:,:,Ikp],dims=3),dims=1), dims=(1,3));

# undecomposed time-mean flux 
Fxt = dropdims(mean(sum(uc[Iday,:,:].*pcp[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fx  = dropdims(mean(sum(uc2[Iday,:,:].*pcp2[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxh = dropdims(mean(sum(uh[Iday,:,:].*ph[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))
Fxl = dropdims(mean(sum(ul[Iday,:,:].*pl[Iday,:,:].*dzz,dims=3),dims=1), dims=(1,3))

# this is equal to Fx?
Fx2 = Fxh + Fxl


 create some figures ----------------------------------------------

#ylimE = [0 75]; ylimA = [0 75];
ylimE = [0 15]; ylimA = [0 15];

fig = Figure(size=(750,750))
ax = Axis(fig[1, 1],title = string(fname_short2,"; lat=",LAT,"; KE [kJ/m2]"), xlabel = "x [km]", ylabel = "KE [kJ/m2]")
lines!(ax, xc/1e3, KEt[:,1]/1e3, label = "tot", linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax, xc/1e3, KE[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax, xc/1e3, KEl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax, xc/1e3, KEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)
#lines!(ax, xc/1e3, KEut[:,1]/1e3, label = "tot", linestyle=:dash, color = :blue, linewidth = 3)
#lines!(ax, xc/1e3, KEu[:,1]/1e3, label = "D2 + HH", color = :blue, linewidth = 3)
#lines!(ax, xc/1e3, KEul[:,1]/1e3, label = "D2", color = :orange, linewidth = 3)
#lines!(ax, xc/1e3, KEuh[:,1]/1e3, label = "HH", color = :cyan, linewidth = 3)
xlims!(ax, 0, Ldom/1e3)
ylims!(ax, ylimE[1], ylimE[2])

ax2 = Axis(fig[2, 1],title = "APE", xlabel = "x [km]", ylabel = "APE [kJ/m2]")
# add nonlinear stuff
lines!(ax2, xc/1e3, APEtnl[:,1]/1e3, label = "tnl",linestyle=:dash, color = :cyan, linewidth = 3)
lines!(ax2, xc/1e3, APEtnl2[:,1]/1e3, label = "tnl2",linestyle=:dash, color = :blue, linewidth = 3)

lines!(ax2, xc/1e3, APEt[:,1]/1e3, label = "tot",linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax2, xc/1e3, APE[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax2, xc/1e3, APEl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax2, xc/1e3, APEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)
xlims!(ax2, 0, Ldom/1e3)
ylims!(ax2, ylimA[1], ylimA[2])
axislegend(ax2, position = :rt)


ax3 = Axis(fig[3, 1],title = "flux", xlabel = "x [km]", ylabel = "flux [W/m]")
lines!(ax3, xc/1e3, Fxt[:,1]/1e3, label = "tot",linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax3, xc/1e3, Fx[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax3, xc/1e3, Fxl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax3, xc/1e3, Fxh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)
xlims!(ax3, 0, Ldom/1e3)
#ylimf = [0 15]
ylimf = [-2 15]
ylims!(ax3, ylimf[1], ylimf[2])
axislegend(ax3, position = :rt)

fig