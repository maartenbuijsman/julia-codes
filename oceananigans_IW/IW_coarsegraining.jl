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

dx   = ds["Δx_caa"];
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

Πx = -(ucl.*ucl .- uucl).*dudx +   
     -(vfl.*ucl .- uvcl).*dvdx; 
Πz = -(ucl.*wcl .- uwcl).*dudz +   
     -(vfl.*wcl .- vwcl).*dvdz; 
Πnh = -(ucl.*wcl .- uwcl).*dwdx +   
      -(wcl.*wcl .- wwcl).*dwdz; 


fig1 = Figure()
axa = Axis(fig1[1, 1])
axb = Axis(fig1[2, 1])
axc = Axis(fig1[3, 1])
hm = heatmap!(axa, xc/1e3, zc, Πx[300,:,:], colormap = :Spectral); Colorbar(fig1[1,2], hm)
hm = heatmap!(axb, xc/1e3, zc, Πz[300,:,:], colormap = :Spectral); Colorbar(fig1[2,2], hm)
hm = heatmap!(axc, xc/1e3, zc, Πnh[300,:,:], colormap = :Spectral); Colorbar(fig1[3,2], hm)
fig1      

# time average
# f(x,z)
Πxa = dropdims(mean(Πx,dims=1),dims=1);
Πza = dropdims(mean(Πz,dims=1),dims=1);
Πnha = dropdims(mean(Πnh,dims=1),dims=1);

fig1 = Figure(size = (600, 800))
axa = Axis(fig1[1, 1])
axb = Axis(fig1[2, 1])
axc = Axis(fig1[3, 1])
axd = Axis(fig1[4, 1])
hm = heatmap!(axa, xc/1e3, zc, Πxa,  colormap = :Spectral); Colorbar(fig1[1,2], hm)
hm = heatmap!(axb, xc/1e3, zc, Πza,  colormap = :Spectral); Colorbar(fig1[2,2], hm)
hm = heatmap!(axc, xc/1e3, zc, Πnha, colormap = :Spectral); Colorbar(fig1[3,2], hm)
hm = heatmap!(axd, xc/1e3, zc, Πxa.+Πza.+Πnha, colormap = :Spectral); Colorbar(fig1[4,2], hm)
fig1      

# f(z)
Πxza = dropdims(mean(Πxa,dims=1),dims=1);
Πzza = dropdims(mean(Πza,dims=1),dims=1);
Πnhza = dropdims(mean(Πnha,dims=1),dims=1);

fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "Π [W/kg]", ylabel = "z [m]")
lines!(ax,Πxza,zc,color=:red, label="Πx")
lines!(ax,Πzza,zc,color=:green, label="Πz")
lines!(ax,Πnhza,zc,color=:black, label="Πnh")
lines!(ax,Πnhza+Πzza+Πxza,zc,color=:orange, label="sum")
axislegend(position = :rb)
fig

# f(x)
dzz = reshape(dz,1,:)
Πxxa  = dropdims(sum(Πxa.*dzz,dims=2),dims=2);
Πzxa  = dropdims(sum(Πza.*dzz,dims=2),dims=2);
Πnhxa = dropdims(sum(Πnha.*dzz,dims=2),dims=2);

fig = Figure(); 
ax = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]")
lines!(ax,xc/1e3,Πxxa,color=:red, label="Πx")
lines!(ax,xc/1e3,Πzxa,color=:green, label="Πz")
lines!(ax,xc/1e3,Πnhxa,color=:black, label="Πnh")
lines!(ax,xc/1e3,Πnhxa+Πzxa+Πxxa,color=:orange, label="sum")
axislegend(position = :rb)
fig







#@time begin 
#    println("finished in ")
#end



# close the nc file
close(ds)
