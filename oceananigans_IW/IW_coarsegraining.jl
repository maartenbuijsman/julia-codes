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

# load variables ===========================================

#pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\"
pathname = "/home/mbui/Documents/julia-codes/functions/"
include(string(pathname,"include_functions.jl"))


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
const Nb = 0.005;     # buoyancy freq
const T2 = 12+25.2/60

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# u, v, w velocities
# NOTE: in future select a certain x range away from boundaries

# 
uf = ds["u"]; #x_faa × z_aac × time
vf = ds["v"]; #x_caa × z_aac × time
wf = ds["w"]; #x_caa × z_aaf × time

# work with permuted matrices: time, x, z


@time begin
    pp = permutedims(uf, (3, 1, 2)); uf = pp;
    pp = permutedims(vf, (3, 1, 2)); vf = pp;
    pp = permutedims(wf, (3, 1, 2)); wf = pp;
    println("finished permutation in ")
end

# compute at cell centers
# v is already at x,W centers
uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 
wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 

fig = Figure(); ax = Axis(fig[1, 1])
lines!(ax,tday,uf[:,1,1],color=:red)
fig

# filter all velocities ======================================
# low-pass
Tcut=9; dth=dt*24; N = 8;   # all in hours
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
u*u - u*u * dudx 
u*w - u*w * dudz 

v*u - v*u * dvdx
v*w - v*w * dvdz 

w*u - w*u * dwdx (these are likely small, for nonhydrostatic sims)
w*w - w*w * dwdz 

# omitted for 2D
u*v - u*v * dudy 
v*v - v*v * dvdy 
w*v - w*v * dwdy 
=#

u1 = 

# close the nc file
close(ds)
