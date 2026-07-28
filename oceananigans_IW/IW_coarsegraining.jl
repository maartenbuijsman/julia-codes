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
const T2 = 12+25.2/60

# run names --------------------------------
# D2, mode 1 + 2 interactions
LATS   = [0.0, 0.0, 0.0];
#runnms = [1,   2,   3]; mainnm = 6;
runnms = [4,   5,   6]; mainnm = 6;    # nonhyd 4 km, weno, v=1e-2
#runnms = [1,   2,   3]; mainnm = 7;
#runnms = [4,   5,   6]; mainnm = 7;
runnms = [13, 14, 15]; mainnm = 7;

xlim = 700

# function of latitude
mainnm = 3
LATS   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 50.0]
runnms = [3,   4,   5,   6,    7,    8,    9,    10,   11,   12,   13,   14]

xlim = 2000


# coarsegraining function -----------------------------
function run_coarsegraining(runnm,LAT)
#IS = 3; runnm = runnms[IS]; LAT = LATS[IS]

# file ID ----------------------
fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

fname_short2 = fnames
filename = string(dirsim,fnames,".nc")

titlenm = string(LAT,"°N; mode 1")    
titlenm2 = string(LAT,"°N")        

println(fname_short2,"; lat=",LAT," -------------------") 


# load simulations ===========================================
ds = NCDataset(filename,"r");
println(keys(ds))

# only select data after the spinup time 
# => i.e., time it takes before waves reach the eastern boundary
#tspin = 4; #days  # for 700 km domain
tspin = 10; #days  # for 2000 km domain

# select Indices after spinup
tday0 = ds["time"][:]/24/3600;
Isel = findall(>=(tspin),tday0);

tsec = ds["time"][Isel];
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
@time begin
    println("reading nc file ",filename)
    uf  = permutedims(ds["u"][:,:,Isel],(3,1,2));
    vf  = permutedims(ds["v"][:,:,Isel],(3,1,2));
    wf  = permutedims(ds["w"][:,:,Isel],(3,1,2));
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
#Tcut=9;  # all in hours
#Tcut=  (T2+2*T2/3)/2 
Tcut=  (T2+2*T2/4)/2 
fstring = "low"

#= test
uu = uf[1,1,:]; uuf = lowhighpass_butter(uu,Tcut,dth,N,fstring)
ww = wf[ix,15,:].*tukey(Nt,0); wwf = lowhighpass_butter(ww,Tcut,dth,N,fstring) 

fig = Figure(); ax = Axis(fig[1, 1])
lines!(ax,tday,ww,color=:red)
lines!(ax,tday,wwf,color=:green,linestyle=:dash); fig
=#

# starting low-passing
# instead of filtering the full 3D array at once, filter slices in parallel
@time begin
    println("starting low-pass filtering")    
    ufl = similar(uf); vfl = similar(uc); wfl = similar(wf); 
    ucl = similar(uc); wcl = similar(uc);
    uucl = similar(uc); uvcl = similar(uc); uwcl = similar(uc);
    vvcl = similar(uc); vwcl = similar(uc); wwcl = similar(uc);
    Threads.@threads for ix in 1:Nx+1
        for iz in 1:Nz
            ufl[:, ix, iz] = lowhighpass_butter(uf[:, ix, iz], Tcut, dth, N, fstring)
        end
        if ix <= Nx
            for iz in 1:Nz
                vfl[:, ix, iz] = lowhighpass_butter(vf[:, ix, iz],Tcut,dth,N,fstring);
                wfl[:, ix, iz] = lowhighpass_butter(wf[:, ix, iz],Tcut,dth,N,fstring);

                ucl[:, ix, iz] = lowhighpass_butter(uc[:, ix, iz],Tcut,dth,N,fstring);
                wcl[:, ix, iz] = lowhighpass_butter(wc[:, ix, iz],Tcut,dth,N,fstring);

                uucl[:, ix, iz] = lowhighpass_butter(uc[:, ix, iz].*uc[:, ix, iz],Tcut,dth,N,fstring);
                uvcl[:, ix, iz] = lowhighpass_butter(uc[:, ix, iz].*vf[:, ix, iz],Tcut, dth, N, fstring); # co-loc in x
                uwcl[:, ix, iz] = lowhighpass_butter(uc[:, ix, iz].*wc[:, ix, iz],Tcut,dth,N,fstring);

                vvcl[:, ix, iz] = lowhighpass_butter(vf[:, ix, iz].*vf[:, ix, iz],Tcut,dth,N,fstring);    
                vwcl[:, ix, iz] = lowhighpass_butter(vf[:, ix, iz].*wc[:, ix, iz],Tcut,dth,N,fstring);    

                wwcl[:, ix, iz] = lowhighpass_butter(wc[:, ix, iz].*wc[:, ix, iz],Tcut,dth,N,fstring);    
            end
            wfl[:, ix, Nz+1] = lowhighpass_butter(wf[:, ix, Nz+1],Tcut,dth,N,fstring);
        end
    end
end

#=N = 4;  
wfl = lowhighpass_butter(wf,Tcut,dth,N,fstring);

fig = Figure()
ax = Axis(fig[1, 1]) 
ix = 1000
lines!(ax,tday,wf[:,ix,15],color = :red)
lines!(ax,tday,wfl[:,ix,15],color = :green, linestyle = :dash)
fig
=#

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

# clear many variables      
dudx=nothing; dwdz=nothing; dudz=nothing; dvdz=nothing; dvdx=nothing; dwdx=nothing;
uucl=nothing; uvcl=nothing; uwcl=nothing; vvcl=nothing; vwcl=nothing; wwcl=nothing;
ucl=nothing; wcl=nothing; ufl=nothing; vfl=nothing; wfl=nothing;
GC.gc()      
      
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

# cycles to exclude
EXCL = 2; t1 = tday[1]+EXCL*T2/24; t2 = tday[end]-EXCL*T2/24;

# exclude 0.5 and 1 days to get 10 tidal cycles for 11 day sim
# EXCL = 1; t1 = tday[1]+0.5; t2 = tday[end]-EXCL;
numcycles = floor((t2-t1)/(T2/24))
t2 = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)

# f(x,z)
Πxa = dropdims(mean(Πx[Iday,:,:],dims=1),dims=1);
Πza = dropdims(mean(Πz[Iday,:,:],dims=1),dims=1);
Πnha = dropdims(mean(Πnh[Iday,:,:],dims=1),dims=1);

ylim = -700;

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
Πtot = Πxa.+Πza.+Πnha;

# smooth only for small dx
if dx[1] < 500
    # 3-pnt ave is enough to remove noise!
    Πtot[2:end-1,:]  = 1/3*Πtot[1:end-2,:] + 1/3*Πtot[2:end-1,:] + 1/3*Πtot[3:end,:]; #best

#    Πtot[2:end-1,:]  = 1/4*Πtot[1:end-2,:] + 1/2*Πtot[2:end-1,:] + 1/4*Πtot[3:end,:];     
#    Πtot[3:end-2,:]  = 1/5*Πtot[1:end-4,:] + 1/5*Πtot[2:end-3,:] + 1/5*Πtot[3:end-2,:] +
#    1/5*Πtot[4:end-1,:] + 1/5*Πtot[5:end,:]; 
end 

clims = (-1e-6,1e-6)
fig1 = Figure(size = (1000, 250))
axa = Axis(fig1[1, 1], title=string(fname_short2,"; ",titlenm,"; cross-scale transfer [W/kg] "), xlabel="x [km]", ylabel="z [m]"); 
#hm = heatmap!(axa, xc/1e3, zc, Πxa.+Πza.+Πnha, colormap = Reverse(:Spectral), colorrange = clims); 
hm = heatmap!(axa, xc/1e3, zc, Πtot, colormap = Reverse(:RdBu_5), colorrange = clims); 
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
#xlim = 1400
fig = Figure(size = (600, 300));
 ax2 = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]", title=string(fname_short2,"; ",titlenm2))
lines!(ax2,xc/1e3,Πxxa,color=:red, label="Πx")
lines!(ax2,xc/1e3,Πzxa,color=:green, label="Πz")
lines!(ax2,xc/1e3,Πnhxa,color=:black, label="Πnh")
lines!(ax2,xc/1e3,Πnhxa+Πzxa+Πxxa,color=:orange, label="sum", linewidth = 3)
axislegend(ax2,position = :rt; framevisible = false )
xlims!(ax2, 0, xlim)
ylims!(ax2, -1e-4, 2e-4)
fig
#

if figflag==1; save(string(dirfig,"PIx_",fname_short2,".png"), fig)
end

# save f(x,z) profiles

# 2D profile; needs averaging in space
Πxztot = Πxa.+Πza.+Πnha;

fnameout = string("Etran_",fname_short2,".jld2")

jldsave(string(dirout,fnameout); LAT, xc, Πnhxa, Πzxa, Πxxa, zc, Πnhza, Πzza, Πxza, Πxztot);
println(string(fnameout)," data saved ........ ")

#stop()
end         #function run_coarsegraining(runnm,LATS)

# runnms loop ---------------
elapsed = @elapsed begin
    for (runnm, LAT) in zip(runnms, LATS)
        run_coarsegraining(runnm, LAT)
    end
end
println("finished in $(round(elapsed, digits=1)) s")



