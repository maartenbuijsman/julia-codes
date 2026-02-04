#= IW_pressure_soliton.jl
Maarten Buijsman, USM DMS, 2026-1-24
zoom in on soliton; plot ssh
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2
using ColorSchemes
using LaTeXStrings

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
figflag = 0
oldnm   = 1  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing
const T2 = 12+25.2/60

#      38 39 40 41 42 43 44 45 46 47 48    49
LATS = [0 2.5 5 10 15 20 25 30 40 50 28.80 35]

#runnms = [38 39 40 41 42 43 44 45 46 47];
runnms = [49]


# file name ===========================================

if oldnm==1
    # function of latitude
    lat = 0

   #fnames = @sprintf("AMZv_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"
   #fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"  # hydro
   fnames = @sprintf("AMZ4_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); titlenm = "mode 1"     # nonhydro

   # fnames = "AMZ4_00.0_hvis_12d_U1_0.40_U2_0.30_v2.nc"; titlenm = "mode 1+2"  # nonhydro and compare with
   # fnames = "AMZ4km_00.0_hvis_12d_U1_0.40_U2_0.30.nc"; titlenm = "mode 1+2"  # hydro

    fname_short2 = fnames[1:33]
    filename = string(dirsim,fnames)

    LAT = LATS[1];
else
    # file ID
    mainnm = 1
    runnm  = 47

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")

    LAT = LATS[runnm-37];
    #println("lat is ",LAT) 

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
Ldom = sum(dx);

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
#pHY = zeros(Nt,Nx,Nz);
#pNHS = zeros(Nt,Nx,Nz);

for i in 1:Nt
    println(i)
    uf[i,:,:] = ds["u"][:, :, i];
    vf[i,:,:] = ds["v"][:, :, i];
    wf[i,:,:] = ds["w"][:, :, i];
    bc[i,:,:] = ds["b"][:, :, i];
    #pHY[i,:,:] = ds["pHY"][:, :, i];
    #pNHS[i,:,:] = ds["pNHS"][:, :, i];    
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
clims =(0,0.8)
ax = Axis(fig1[1, 1],title = string(fname_short2,"; lat=",LAT,"; KE [m2/s2]"), xlabel = "x [km]", ylabel = "time [days]")
hm = heatmap!(ax, xc/1e3, tday, transpose(uc[:,:,end].^2 .+ vc[:,:,end].^2), colormap = Reverse(:Spectral), colorrange = clims); Colorbar(fig1[1,2], hm); 
#hm = heatmap!(ax, xc/1e3, tday, transpose(uc[:,:,end].^2), colormap = Reverse(:Spectral)); Colorbar(fig1[1,2], hm); 
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"KE_hovmuller_", fname_short2 ,".png"), fig1)
end

#stop()

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

# In Oceananigans: dpk/dz = b = -g/rho0*rho_pert [m/s2*m3/kg*kg/m3 = m/s2]
#                  [m2/s2*1/m] = [m/s2]               
# units b: [m/s2]
# because of kinematic pressure pk = p/rho0 = kg*m2/s2*1/m3 *m3/kg = m2/s2

rho0=1020; 
grav=9.81; 

# hydrostatic pressure
dzz = reshape(dz,1,1,:);                                    # shape: (1, length(zc), 1)
#N2cc = reshape(N2c,1,1,:);
N2cc = permutedims(repeat(N2c,1,Nt,Nx),[2, 3, 1]);
pfi = cumsum(bc[:,:,end:-1:1].*dzz[:,:,end:-1:1], dims=3);  # reverse, z surface down, at faces
pfi = pfi * -1 * rho0 / grav;                               # convert to pert pressure

# compute density => for plotting purposes
# N2 = -g/rho0*drho/dz
# drho = N2*dz*rho0/g
rhoBf = ones(Nt,Nx,Nz+1).*rho0
rhoBf[:,:,2:Nz+1] = rho0 .- cumsum(N2cc.*dzz, dims=3)*rho0/grav;
rhoBc = rhoBf[:,:,1:end-1]/2 .+ rhoBf[:,:,2:end]/2

# add perturbation to the background
rhoc = rhoBc .- bc*rho0/grav;

Figure(); lines(1:Nz,rhoBc[1,1,:])




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

# plot a heatmap of velocity and include density contours
cmap = Reverse(:RdBu_9);
#cmap = (:temperaturemap);
#clims  = (-1,1)
clims  = (-0.3,0.3)
#xlim = (435,470)
xlim = (0,500); zlim = (-4000,0); fstr = "ent_dom_u_rho_";
xlim = (325,385); zlim = (-1000,0); fstr = "zoom_u_rho_";
#xlim = (367,387); zlim = (-1000,0); fstr = "zoom2_u_rho_";


It = 459;

titstr = string(L"zonal u [m/s]; $0^\circ$N; mode 1; hydrostatic, $\Delta x = 4$ km");
#titstr = string(L"zonal u [m/s]; 0^\circ N; mode 1; nonhydrostatic, $\Delta x = 200$ m");

fig = Figure(size = (1000,250))
ax = Axis(fig[1, 1],title=titstr, xlabel="x [km]", ylabel="z [m]")#, xticks = 367:2:387)
hm=heatmap!(ax, xc/1e3, zc, uc[It,:,:], colormap = cmap, colorrange = clims) # Customize colormap as needed

my_levels = [1015, 1015.5, 1018.5, 1019, 1019.5, 1019.95]
contour!(ax, xc/1e3, zc, rhoc[It,:,:]; 
    levels = my_levels,  # Prescribes exact values for lines
    color = :black,      # Sets all lines to black
    labels = true,       # Optional: shows level values
    linewidth = 1
)

Colorbar(fig[1, 2], hm)
xlims!(ax, xlim[1], xlim[2])
ylims!(ax, zlim[1], zlim[2])
fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig, fstr, fname_short2 ,".png"), fig)
end

stop()

# convert to SSH: p = g*rho*eta => eta = p/(rho*g)
eta = pcp[:,:,end]/(grav*rho0);

# save as a nc file
using DataStructures: OrderedDict

# Create a new NetCDF file
dds = NCDataset(string(dirout,"AMZnonhyd_eta_mode12.nc"), "c")

# Define dimensions
defDim(dds, "xc", length(xc))
defDim(dds, "tday", length(tday))

# Define variable
v = defVar(dds, "eta", Float64, ("tday","xc"))
v.attrib["units"] = "m"
v.attrib["long_name"] = "Sea Surface Height"

w = defVar(dds, "xc", Float64, ("xc",))
w.attrib["units"] = "km"
w.attrib["long_name"] = "x value at cell centers"

x = defVar(dds, "tday", Float64, ("tday",))
x.attrib["units"] = "days"
x.attrib["long_name"] = "time"

# Write data
v[:, :] = eta
w[:] = xc
x[:] = tday

# Close the file
close(dds)


#NEED to INCLUDE hyd/nonhyd pressure!!

fig1 = Figure(); 
ax1a = Axis(fig1[1, 1],title="SSH")
#lines!(ax1a, xc/1e3, pcp[end,:,end], color = :black)
#lines!(ax1a, xc/1e3, pHY[end,:,end], color = :blue)
#lines!(ax1a, xc/1e3, pHY[end,:,1], color = :red)
lines!(ax1a, xc/1e3, pNHS[end,:,end], color = :green)
lines!(ax1a, xc/1e3, pNHS[end,:,1], color = :grey)
fig1

Ix = 1500;
fig1 = Figure()
ax1a = Axis(fig1[1, 1],title="SSH")
lines!(ax1a, pHY[end,Ix,:],zc, color = :red)
lines!(ax1a, pNHS[end,Ix,:],zc, color = :grey)
fig1

# some more hovmullers
clims = (-0.025,0.025)
fig1 = Figure(size=(660,800))
ax1a = Axis(fig1[1, 1],title="SSH")
hm1=heatmap!(ax1a, xc/1e3, tday, transpose(eta), colormap = Reverse(:Spectral), colorrange = clims)
Colorbar(fig1[1, 2], hm1)
fig1

stop()



# filter variables  ======================================================
# uc2: highpassed D2 + HH
# uh:  HH
# 
Nf = 6;

#= isolate the D2 motions
Tl=(T2+T2/4)/24,Th=(T2-T2/4)/24
uc2 = bandpass_butter(uc,Tl,Th,dt,Nf)

Tcut1 = 16/24;
uc2  = lowhighpass_butter(uc,Tcut1,dt,Nf,"high");
vc2  = lowhighpass_butter(vc,Tcut1,dt,Nf,"high");
pcp2 = lowhighpass_butter(pcp,Tcut1,dt,Nf,"high");
bc2  = lowhighpass_butter(bc,Tcut1,dt,Nf,"high");
=#

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
# this is D2
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
# KEt is total, unfiltered
# KE  is D2+HH
# KEh is HH
# KEl is D2
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
ax = Axis(fig[1, 1],title = string(fname_short2,"; lat=",LAT,"; KE [kJ/m2]"), xlabel = "x [km]", ylabel = "KE [kJ/m2]")
lines!(ax, xc/1e3, KEt[:,1]/1e3, label = "tot", linestyle=:dash, color = :grey, linewidth = 3)
#lines!(ax, xc/1e3, KE[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax, xc/1e3, KE2[:,1]/1e3, label = "D2 + HH", linestyle=:dash, color = :yellow, linewidth = 3)
lines!(ax, xc/1e3, KEl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax, xc/1e3, KEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)

lines!(ax, xc/1e3, KEut[:,1]/1e3, label = "tot", linestyle=:dash, color = :blue, linewidth = 3)
lines!(ax, xc/1e3, KEu[:,1]/1e3, label = "D2 + HH", color = :blue, linewidth = 3)
lines!(ax, xc/1e3, KEul[:,1]/1e3, label = "D2", color = :orange, linewidth = 3)
lines!(ax, xc/1e3, KEuh[:,1]/1e3, label = "HH", color = :cyan, linewidth = 3)

xlims!(ax, 0, Ldom/1e3)
ylims!(ax, ylimE[1], ylimE[2])

ax2 = Axis(fig[2, 1],title = "APE", xlabel = "x [km]", ylabel = "APE [kJ/m2]")

lines!(ax2, xc/1e3, APEt[:,1]/1e3, label = "tot",linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax2, xc/1e3, APE[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax2, xc/1e3, APEl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax2, xc/1e3, APEh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)

xlims!(ax2, 0, Ldom/1e3)
ylims!(ax2, ylimA[1], ylimA[2])


ax3 = Axis(fig[3, 1],title = "flux", xlabel = "x [km]", ylabel = "flux [W/m]")

lines!(ax3, xc/1e3, Fxt[:,1]/1e3, label = "tot",linestyle=:dash, color = :grey, linewidth = 3)
lines!(ax3, xc/1e3, Fx[:,1]/1e3, label = "D2 + HH", color = :black, linewidth = 3)
lines!(ax3, xc/1e3, Fxl[:,1]/1e3, label = "D2", color = :red, linewidth = 3)
lines!(ax3, xc/1e3, Fxh[:,1]/1e3, label = "HH", color = :green, linewidth = 3)

xlims!(ax3, 0, Ldom/1e3)
ylims!(ax3, ylimf[1], ylimf[2])
axislegend(ax3, position = :rt)

fig

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"KE_flux_", fname_short2 ,".png"), fig)
end


println(fnames,"; max total flux is ",@sprintf("%5.2f",maximum(Fxt/1e3))," kW/m")
println(fnames,"; max D2+HH flux is ",@sprintf("%5.2f",maximum(Fx/1e3))," kW/m")


# save the energy terms =========================================
fnameout = string("energetics_",fname_short2,".jld2")

jldsave(string(dirout,fnameout); xc, Fxt, Fx, Fxh, Fxl, Fx2, KEt, APEt, KE, APE, KEh, APEh, KEl, APEl, KE2, KEut, KEu, KEuh, KEul);
println(string(fnameout)," data saved ........ ")


# compute some ffts of surface velocity ======================================================

EXCL = 0;  # can be zero for fft
t1,t2 = 4, tday[end]-EXCL*T2/24
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
xlims = [75,500]*1e3;
Plims = [-12 1];
Ix = findall(item -> item >= xlims[1] && item<= xlims[2], xc);
KEoma = vec(mean(KEom[:,Ix],dims=2)); 

axb = Axis(fig1[2, 1],xticks = (flim[1]:fstp:flim[2]),xlabel="frequency [cpd]",ylabel="log10(KE) [m2/s2/cpd]");  
xlims!(axb, flim[1], flim[2])
ylims!(axb, Plims[1], Plims[2])
lines!(axb, freq, log10.(KEoma), linestyle=:solid, color = :black, linewidth = 3)

# add coriolis rad/s => cpd
fcpd = coriolis(LAT)/(2*pi)*24*3600
#lines!(vec([fcpd fcpd]),vec([minimum(log10.(KEoma)) maximum(log10.(KEoma))]), linestyle=:dash, color = :red, linewidth = 2)
lines!(axb, vec([fcpd fcpd]),vec(Plims), linestyle=:dash, color = :red, linewidth = 2)
fig1

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"fft_usur_",fname_short2,".png"), fig1)
end



# nonhyd pressure
# nonlinear APE
# ratio KE/APE 
# check reflections
# fft
# make wider sponge layers to avoid reflections

stop()
