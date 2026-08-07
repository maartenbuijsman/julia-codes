#= IW_coarsegraining_tile.jl
Maarten Buijsman, USM DMS, 2026-7-30
Load model runs and perform coarsegraining diagnostics
Tiled in x to avoid memory overflow for large domains (e.g. 200 m, 2000 km)
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
#LATS   = [0.0, 0.0, 0.0];
#runnms = [1,   2,   3]; mainnm = 6;
#runnms = [4,   5,   6]; mainnm = 6;    # nonhyd 4 km, weno, v=1e-2
#runnms = [1,   2,   3]; mainnm = 7;
#runnms = [4,   5,   6]; mainnm = 7;
#runnms = [13, 14, 15]; mainnm = 7;

xlim = 700

# function of latitude
#mainnm = 3
#LATS   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 50.0]
#runnms = [3,   4,   5,   6,    7,    8,    9,    10,   11,   12,   13,   14]

mainnm = 9
#LATS   = [0.0, 2.5, 5.0]
#runnms = [1,   2,   3]
LATS   = collect(50.0:5.0:60.0)
runnms = collect(12:1:14)

xlim = 2000; # km

ntile = 10   # number of x-tiles; increase if memory is still tight


# coarsegraining function (tiled in x) -----------------------------
function run_coarsegraining(runnm, LAT)
# IS = 3; runnm = runnms[IS]; LAT = LATS[IS]

# file ID ----------------------
fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
fname_short2 = fnames
filename = string(dirsim,fnames,".nc")
titlenm  = string(LAT,"°N; mode 1")
titlenm2 = string(LAT,"°N")
println(fname_short2,"; lat=",LAT," -------------------")

# === Load grid and time metadata — keep ds open for tile reads ===
ds = NCDataset(filename,"r");
println(keys(ds))

tspin = 10; # days (2000 km domain)
tday0 = ds["time"][:]/24/3600;
Isel  = findall(>=(tspin),tday0);
tsec  = ds["time"][Isel];
tday  = tsec/24/3600;
dt    = tday[2]-tday[1]

xc = ds["x_caa"][:];
zc = ds["z_aac"][:];
dx = ds["Δx_caa"][:];
dz = ds["Δz_aac"][:];

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# Time averaging window (same for all tiles, compute once)
EXCL = 2; t1 = tday[1]+EXCL*T2/24; t2 = tday[end]-EXCL*T2/24;
numcycles = floor((t2-t1)/(T2/24))
t2   = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item <= t2, tday)

# Filter parameters
dth = dt*24; N = 8;
Tcut = (T2+2*T2/4)/2
fstring = "low"

# z-gradient helpers (same for all tiles)
dz2  = zc[3:end] - zc[1:end-2];
dzz2 = reshape(dz2,1,1,:);
dzr  = reshape(dz,1,1,:);
dzb  = zc[2] - zc[1];      # bottom one-sided
dzs  = zc[end] - zc[end-1]; # surface one-sided

# === Pre-allocate full time-mean output arrays ===
Πxa_full  = zeros(Nx,Nz);
Πza_full  = zeros(Nx,Nz);
Πnha_full = zeros(Nx,Nz);

# === Tile parameters ===
# nhalo=1: one x-center point on each side, needed for central diff of dvdx, dwdx.
# The Butterworth filter is purely temporal so needs NO halo.
nhalo   = 1;
nx_base = Nx ÷ ntile;

# === Tile loop ===
for i_tile in 1:ntile
    println("  tile ",i_tile," / ",ntile)

    # interior x-center indices (global, 1-based)
    ix_a = (i_tile-1)*nx_base + 1;
    ix_b = (i_tile == ntile) ? Nx : i_tile*nx_base;

    # x-center indices with halo
    ix_ah = max(1, ix_a-nhalo);
    ix_bh = min(Nx, ix_b+nhalo);
    nx_h  = ix_bh - ix_ah + 1;

    # x-face indices for u: one extra face to the right of the rightmost center
    ixu_ah = ix_ah;
    ixu_bh = ix_bh + 1;    # always ≤ Nx+1 since ix_bh ≤ Nx

    # Load tile: all selected time steps, full z range
    @time begin
        uf_t = permutedims(ds["u"][ixu_ah:ixu_bh, :, Isel], (3,1,2)); # Nt × (nx_h+1) × Nz
        vf_t = permutedims(ds["v"][ix_ah:ix_bh,   :, Isel], (3,1,2)); # Nt × nx_h     × Nz
        wf_t = permutedims(ds["w"][ix_ah:ix_bh,   :, Isel], (3,1,2)); # Nt × nx_h     × (Nz+1)
    end

    # Cell-center velocities for tile
    uc_t = uf_t[:,1:end-1,:]/2 + uf_t[:,2:end,:]/2;  # Nt × nx_h × Nz
    wc_t = wf_t[:,:,1:end-1]/2 + wf_t[:,:,2:end]/2;  # Nt × nx_h × Nz

    # === Filter (time-domain: each (ix,iz) column independent, no halo needed) ===
    ufl_t  = similar(uf_t);
    vfl_t  = similar(vf_t);
    wfl_t  = similar(wf_t);
    ucl_t  = similar(uc_t);
    wcl_t  = similar(wc_t);
    uucl_t = similar(uc_t);
    uvcl_t = similar(uc_t);
    uwcl_t = similar(uc_t);
    vvcl_t = similar(uc_t);
    vwcl_t = similar(uc_t);
    wwcl_t = similar(uc_t);

    nx_uf = size(uf_t,2);   # = nx_h+1

    Threads.@threads for ix in 1:nx_uf
        for iz in 1:Nz
            ufl_t[:,ix,iz] = lowhighpass_butter(uf_t[:,ix,iz],Tcut,dth,N,fstring)
        end
    end

    Threads.@threads for ix in 1:nx_h
        for iz in 1:Nz
            vfl_t[:,ix,iz]  = lowhighpass_butter(vf_t[:,ix,iz],Tcut,dth,N,fstring);
            ucl_t[:,ix,iz]  = lowhighpass_butter(uc_t[:,ix,iz],Tcut,dth,N,fstring);
            wcl_t[:,ix,iz]  = lowhighpass_butter(wc_t[:,ix,iz],Tcut,dth,N,fstring);
            uucl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*uc_t[:,ix,iz],Tcut,dth,N,fstring);
            uvcl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*vf_t[:,ix,iz],Tcut,dth,N,fstring);
            uwcl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*wc_t[:,ix,iz],Tcut,dth,N,fstring);
            vvcl_t[:,ix,iz] = lowhighpass_butter(vf_t[:,ix,iz].*vf_t[:,ix,iz],Tcut,dth,N,fstring);
            vwcl_t[:,ix,iz] = lowhighpass_butter(vf_t[:,ix,iz].*wc_t[:,ix,iz],Tcut,dth,N,fstring);
            wwcl_t[:,ix,iz] = lowhighpass_butter(wc_t[:,ix,iz].*wc_t[:,ix,iz],Tcut,dth,N,fstring);
        end
        wfl_t[:,ix,Nz+1] = lowhighpass_butter(wf_t[:,ix,Nz+1],Tcut,dth,N,fstring);
    end

    uf_t=nothing; vf_t=nothing; wf_t=nothing;
    uc_t=nothing; wc_t=nothing;
    GC.gc()

    # === Gradients ===

    # dudx: forward diff over x-faces → x-centers (nx_h values)
    dxr_t  = reshape(dx[ix_ah:ix_bh],1,:,1);
    dudx_t = diff(ufl_t,dims=2)./dxr_t;           # Nt × nx_h × Nz

    # dwdz: forward diff over z-faces → z-centers
    dwdz_t = diff(wfl_t,dims=3)./dzr;             # Nt × nx_h × Nz

    # dudz, dvdz: central diff in z; one-sided at z-boundaries
    dudz_t = zeros(Nt,nx_h,Nz);
    dvdz_t = zeros(Nt,nx_h,Nz);
    dudz_t[:,:,2:end-1] = (ucl_t[:,:,3:end] - ucl_t[:,:,1:end-2])./dzz2;
    dvdz_t[:,:,2:end-1] = (vfl_t[:,:,3:end] - vfl_t[:,:,1:end-2])./dzz2;
    dudz_t[:,:,1]   = (ucl_t[:,:,2] - ucl_t[:,:,1])/dzb;
    dvdz_t[:,:,1]   = (vfl_t[:,:,2] - vfl_t[:,:,1])/dzb;
    dudz_t[:,:,end] = (ucl_t[:,:,end] - ucl_t[:,:,end-1])/dzs;
    dvdz_t[:,:,end] = (vfl_t[:,:,end] - vfl_t[:,:,end-1])/dzs;

    # dvdx, dwdx: central diff in x
    # Interior of tile (j=2:nx_h-1): halo provides the extra stencil points so
    # central diff is exact at all interior x-points ix_a:ix_b.
    # Tile boundaries (j=1 and j=nx_h): one-sided diff, only reached when
    # the tile boundary coincides with the global domain boundary.
    dvdx_t = zeros(Nt,nx_h,Nz);
    dwdx_t = zeros(Nt,nx_h,Nz);
    if nx_h >= 3
        dx2_t  = xc[ix_ah+2:ix_bh] - xc[ix_ah:ix_bh-2];   # length nx_h-2
        dxx2_t = reshape(dx2_t,1,:,1);
        dvdx_t[:,2:end-1,:] = (vfl_t[:,3:end,:] - vfl_t[:,1:end-2,:])./dxx2_t;
        dwdx_t[:,2:end-1,:] = (wcl_t[:,3:end,:] - wcl_t[:,1:end-2,:])./dxx2_t;
    end
    # left tile boundary (forward diff; exact only when ix_ah == 1)
    dxl_t = xc[ix_ah+1] - xc[ix_ah];
    dvdx_t[:,1,:] = (vfl_t[:,2,:] - vfl_t[:,1,:])/dxl_t;
    dwdx_t[:,1,:] = (wcl_t[:,2,:] - wcl_t[:,1,:])/dxl_t;
    # right tile boundary (backward diff; exact only when ix_bh == Nx)
    dxr_t2 = xc[ix_bh] - xc[ix_bh-1];
    dvdx_t[:,end,:] = (vfl_t[:,end,:] - vfl_t[:,end-1,:])/dxr_t2;
    dwdx_t[:,end,:] = (wcl_t[:,end,:] - wcl_t[:,end-1,:])/dxr_t2;

    # === Π terms ===
    Πx_t  = (ucl_t.*ucl_t .- uucl_t).*dudx_t .+
            (vfl_t.*ucl_t .- uvcl_t).*dvdx_t;
    Πz_t  = (ucl_t.*wcl_t .- uwcl_t).*dudz_t .+
            (vfl_t.*wcl_t .- vwcl_t).*dvdz_t;
    Πnh_t = (ucl_t.*wcl_t .- uwcl_t).*dwdx_t .+
            (wcl_t.*wcl_t .- wwcl_t).*dwdz_t;

    dudx_t=nothing; dwdz_t=nothing; dudz_t=nothing; dvdz_t=nothing;
    dvdx_t=nothing; dwdx_t=nothing;
    uucl_t=nothing; uvcl_t=nothing; uwcl_t=nothing;
    vvcl_t=nothing; vwcl_t=nothing; wwcl_t=nothing;
    ucl_t=nothing; wcl_t=nothing; ufl_t=nothing; vfl_t=nothing; wfl_t=nothing;
    GC.gc()

    # === Time-average over Iday ===
    Πxa_t  = dropdims(mean(Πx_t[Iday,:,:],dims=1),dims=1);   # nx_h × Nz
    Πza_t  = dropdims(mean(Πz_t[Iday,:,:],dims=1),dims=1);
    Πnha_t = dropdims(mean(Πnh_t[Iday,:,:],dims=1),dims=1);
    Πx_t=nothing; Πz_t=nothing; Πnh_t=nothing; GC.gc()

    # === Store interior (strip halo) into full arrays ===
    jx_a = ix_a - ix_ah + 1;   # local tile index of interior start
    jx_b = ix_b - ix_ah + 1;   # local tile index of interior end

    Πxa_full[ix_a:ix_b,:]  = Πxa_t[jx_a:jx_b,:];
    Πza_full[ix_a:ix_b,:]  = Πza_t[jx_a:jx_b,:];
    Πnha_full[ix_a:ix_b,:] = Πnha_t[jx_a:jx_b,:];

end  # tile loop

close(ds)

# Rename to match the rest of the code unchanged
Πxa  = Πxa_full;
Πza  = Πza_full;
Πnha = Πnha_full;

# === Plotting and saving (unchanged from original) ===

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

if figflag==1; save(string(dirfig,"PIxz_",fname_short2,".png"), fig1)
end

# only plot the sum of all coarse graining terms
Πtot = Πxa.+Πza.+Πnha;

# smooth only for small dx
if dx[1] < 500
    Πtot[2:end-1,:]  = 1/3*Πtot[1:end-2,:] + 1/3*Πtot[2:end-1,:] + 1/3*Πtot[3:end,:];
end

clims = (-1e-6,1e-6)
fig1 = Figure(size = (1000, 250))
axa = Axis(fig1[1, 1], title=string(fname_short2,"; ",titlenm,"; cross-scale transfer [W/kg] "), xlabel="x [km]", ylabel="z [m]");
hm = heatmap!(axa, xc/1e3, zc, Πtot, colormap = Reverse(:RdBu_5), colorrange = clims);
ylims!(axa, ylim, 0)
Colorbar(fig1[1,2], hm);
fig1

if figflag==1; save(string(dirfig,"PIsumxz_",fname_short2,".png"), fig1)
end

# f(z)
Πxza  = dropdims(mean(Πxa,dims=1),dims=1);
Πzza  = dropdims(mean(Πza,dims=1),dims=1);
Πnhza = dropdims(mean(Πnha,dims=1),dims=1);

fig = Figure();
ax = Axis(fig[1, 1], xlabel = "Π [W/kg]", ylabel = "z [m]", title=fname_short2)
lines!(ax,Πxza,zc,color=:red, label="Πx")
lines!(ax,Πzza,zc,color=:green, label="Πz")
lines!(ax,Πnhza,zc,color=:black, label="Πnh")
lines!(ax,Πnhza+Πzza+Πxza,zc,color=:orange, label="sum")
axislegend(ax,position = :rb; framevisible = false)
fig

# f(x)
dzz   = reshape(dz,1,:);
Πxxa  = dropdims(sum(Πxa.*dzz,dims=2),dims=2);
Πzxa  = dropdims(sum(Πza.*dzz,dims=2),dims=2);
Πnhxa = dropdims(sum(Πnha.*dzz,dims=2),dims=2);

ax2 = Axis(fig[1, 2], xlabel = "x [km]", ylabel = "Π [W/kg*m]", title=string(fname_short2,"; ",titlenm2))
lines!(ax2,xc/1e3,Πxxa,color=:red, label="Πx")
lines!(ax2,xc/1e3,Πzxa,color=:green, label="Πz")
lines!(ax2,xc/1e3,Πnhxa,color=:black, label="Πnh")
lines!(ax2,xc/1e3,Πnhxa+Πzxa+Πxxa,color=:orange, label="sum")
axislegend(ax2,position = :rt; framevisible = false)
fig

if figflag==1; save(string(dirfig,"PIz_PIx_",fname_short2,".png"), fig)
end

# save only the pi(x) UNSMOOTHED
fig = Figure(size = (600, 300));
ax2 = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "Π [W/kg*m]", title=string(fname_short2,"; ",titlenm2))
lines!(ax2,xc/1e3,Πxxa,color=:red, label="Πx")
lines!(ax2,xc/1e3,Πzxa,color=:green, label="Πz")
lines!(ax2,xc/1e3,Πnhxa,color=:black, label="Πnh")
lines!(ax2,xc/1e3,Πnhxa+Πzxa+Πxxa,color=:orange, label="sum", linewidth = 3)
axislegend(ax2,position = :rt; framevisible = false)
xlims!(ax2, 0, xlim)
ylims!(ax2, -1e-4, 2e-4)
fig

if figflag==1; save(string(dirfig,"PIx_",fname_short2,".png"), fig)
end

# save f(x,z) profiles
Πxztot = Πxa.+Πza.+Πnha;

fnameout = string("Etran_",fname_short2,".jld2")
jldsave(string(dirout,fnameout); LAT, xc, Πnhxa, Πzxa, Πxxa, zc, Πnhza, Πzza, Πxza, Πxztot);
println(string(fnameout)," data saved ........ ")

end  # function run_coarsegraining


# runnms loop ---------------
elapsed = @elapsed begin
    for (runnm, LAT) in zip(runnms, LATS)
        looptime = @elapsed begin
            run_coarsegraining(runnm, LAT)
        end
        println("finished ", runnm," in $(round(looptime, digits=1)) s")        
    end
end
println("finished in $(round(elapsed, digits=1)) s")
