#= IW_total_energetics_tile.jl
Maarten Buijsman, USM DMS, 2026-7-31
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
savefl  = 0
figflag = 1
oldnm   = 0  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing
const T2 = 12+25.2/60
const rho0=1020; 
const grav=9.81; 

# D2, mode 1 + 2 interactions
#LATS   = [0.0, 0.0, 0.0];
#runnms = [1,   2,   3]; mainnm = 6;   # nonhyd 4 km, centered
#runnms = [4,   5,   6]; mainnm = 6;    # nonhyd 4 km, weno, v=1e-2
#runnms = [1,   2,   3]; mainnm = 7;   # nonhyd 200 m, centered v=1e-2
#runnms = [4,   5,   6]; mainnm = 7;   # nonhyd 200 m, weno v=1e-5
#runnms = [7,   8,   9]; mainnm = 7;   # nonhyd 200 m, centered v=1e-5
#runnms = [13,   14,   15]; mainnm = 7;   # nonhyd 200 m, centered v=1e-5
#runnms = [1,   2,   3]; mainnm = 8;   # hyd, 4 km, 200 m, 200 m, weno
#

#runnms = [3]; mainnm = 7;   # nonhyd 200 m, centered v=1e-2

#= D1
mainnm = 5
LATS   = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]
runnms = [9,   10,  11,  12,  13,   14,   15,   16,   17,   18]  # is the same
=#

#= D2
mainnm = 3
LATS  = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 50.0]
runnms = [3,   4,   5,   6,    7,    8,    9,    10,   11,   12,   13,   14]
=#

# D2 NH
mainnm = 9
#LATS  = [0.0, 2.5, 5.0, 10.0]
#runnms = [1, 2, 3, 4]
#LATS  =  [15, 20, 25, 30, 35, 40, 45] 
#runnms = [5, 6,  7,  8,  9,  10,  11]
LATS   = collect(50.0:5.0:60.0)
runnms = collect(12:1:14)
# do the analysis in a function
function run_analysis(runnm,LAT,savefl)
# IS = 1; runnm = runnms[IS]; LAT = LATS[IS]

# filename
fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
fname_short2 = fnames
filename = string(dirsim,fnames,".nc")

println(fname_short2,"; lat=",LAT," -------------------")

# open nc file and keep open for tiled reads
ds = NCDataset(filename,"r");
println(keys(ds))

# only select data after the spinup time
tspin = 10; #days  # for 2000 km domain

# select indices after spinup
tday0 = ds["time"][:]/24/3600;
Isel  = findall(>=(tspin),tday0);

tsec = ds["time"][Isel];
tday = tsec/24/3600;
dt   = tday[2]-tday[1]

xf   = ds["x_faa"][:];
xc   = ds["x_caa"][:];
zc   = ds["z_aac"][:];
dx   = ds["Δx_caa"][:];
dz   = ds["Δz_aac"][:];
Ldom = sum(dx);
H    = sum(dz);

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# load N2 profile -----------------------------------------------------------
fnamegrid = "N2_amz1.jld2";
path_fname = string(dirforce,fnamegrid);
@load path_fname N2w zfw
N2c = N2w[1:end-1]/2 + N2w[2:end]/2;

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,N2c, zc)
ylims!(ax1, -500, 0)
fig

# epsilon calculations -------------------------------------------------------
function getomres(ω,LAT,nonhyd,Nm)
    nk = 4;
    fcor   = coriolis(LAT);
    omi = collect(range(ω, nk*ω, nk))
    function itom(zfw, N2w, fcor, omi, nonhyd, nk, kr, Nm)
        ki  = zeros(nk,)
        for i=1:nk
            kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
                sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, omi[i], nonhyd);
                ki[i] = kn[Nm]
        end
        intzc   = interpolate((ki,), omi, Gridded(Linear()));
        omr = intzc.(kr);
        return omr
    end
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd);
    kr  = 2*kn[Nm]
    omr = itom(zfw, N2w, fcor, omi, nonhyd, nk, kr, Nm)
    om2 = collect(range(0.75*omr, 1.25*omr, nk))
    omr = itom(zfw, N2w, fcor, om2, nonhyd, nk, kr, Nm)
    return omr
end

function getkom(ω,LAT,nonhyd,Nm)
    fcor   = coriolis(LAT);
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd);
    kom  = kn[Nm]
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, 2ω, nonhyd);
    k2om = kn[Nm]
    return kom, k2om
end

ω      = 2π / (T2*3600)
nonhyd = 1; Nm = 1;

omr   = getomres(ω,LAT,nonhyd,Nm)
epsnh = ((2*ω)^2 - omr^2)/(2*ω)^2

nonhyd = 0;
omr   = getomres(ω,LAT,nonhyd,Nm)
epshy = ((2*ω)^2 - omr^2)/(2*ω)^2

nonhyd=1;
kom, k2om = getkom(ω,LAT,nonhyd,Nm)
epsomnh   = ((2*kom)^2 - k2om^2)/(2*kom)^2

nonhyd=0;
kom, k2om = getkom(ω,LAT,nonhyd,Nm)
epsomhy   = ((2*kom)^2 - k2om^2)/(2*kom)^2

# reference density ----------------------------------------------------------
breff   = cumtrapz(zfw, N2w);
intzc   = interpolate((zfw,), breff, Gridded(Linear()));
rhorefc = -intzc.(zc) * rho0/grav;

# time window for KE/APE averaging ------------------------------------------
EXCL = 2; t1 = tday[1]+EXCL*T2/24; t2 = tday[end]-EXCL*T2/24;
numcycles = floor((t2-t1)/(T2/24))
t2   = t1+numcycles*(T2/24)
Iday = findall(item -> item >= t1 && item<= t2, tday)
Nlen = length(Iday)
imid = Nlen÷2   # midpoint index within Iday-length arrays

# filter settings ------------------------------------------------------------
Nf    = 8;
Tcut1 = 18/24       #D2+HH
Tcut2 = (T2+T2/2)/2/24  #day; HH M2-M4
if mainnm == 5
    Tcut1 = 30/24
    Tcut2 = 20/24
end

# helpers defined once (z-only, broadcasted over tiles) ----------------------
dzz   = reshape(dz,1,1,:);
N2cc  = reshape(N2c,1,1,:);
Ikp   = findall(>(1e-10),N2c);
factA = 1/2*rho0*1.0./N2cc;
thresh     = 1e-5;
rrr_shape  = reshape(rhorefc,1,1,:);

# load surface velocities for hovmuller and FFT (cheap 2D slices) ------------
# ds["u"] has layout (Nx+1, Nz, Nt_full) → surface = last z-index
println("loading surface velocities...")
uf_surf_tmp = permutedims(ds["u"][:, end, Isel], (2,1));        # (Nt, Nx+1)
uc_surf = uf_surf_tmp[:,1:end-1]/2 .+ uf_surf_tmp[:,2:end]/2;  # (Nt, Nx)
uf_surf_tmp = nothing; GC.gc()
vc_surf = permutedims(ds["v"][:, end, Isel], (2,1));  # (Nt, Nx)

# pre-allocate full-domain output arrays (1D: Nx; 2D: Nx×Nz) ----------------
KE    = zeros(Nx); KEt   = zeros(Nx); KEh   = zeros(Nx); KEs   = zeros(Nx)
APE   = zeros(Nx); APEt  = zeros(Nx); APEh  = zeros(Nx); APEs  = zeros(Nx)
APElin = zeros(Nx)
Fx    = zeros(Nx); Fxt   = zeros(Nx); Fxh   = zeros(Nx); Fxs   = zeros(Nx)
FKx   = zeros(Nx); FKxt  = zeros(Nx); FKxh  = zeros(Nx); FKxs  = zeros(Nx)
FAx   = zeros(Nx); FAxt  = zeros(Nx); FAxh  = zeros(Nx); FAxs  = zeros(Nx)
uca_full = zeros(Nx, Nz)
Zzt_mid  = zeros(Nx, Nz)
A0nl     = 0.0
fact     = 1/2*rho0

# tile loop (nhalo=0: no x-gradients needed) ---------------------------------
ntile   = 20
nx_base = Nx ÷ ntile

println("starting tile loop, ntile=",ntile)
for i_tile in 1:ntile
    ix_a = (i_tile-1)*nx_base + 1
    ix_b = (i_tile == ntile) ? Nx : i_tile*nx_base
    nx_t = ix_b - ix_a + 1
    println("  tile ",i_tile,"/",ntile,"  ix=",ix_a,":",ix_b)

    # load tile data ---------------------------------------------------------
    # u on x-faces: need one extra face at ix_b+1 to center to ix_b
    uf_t  = permutedims(ds["u"][ix_a:ix_b+1, :, Isel], (3,1,2));   # (Nt, nx_t+1, Nz)
    vc_t  = permutedims(ds["v"][ix_a:ix_b,   :, Isel], (3,1,2));   # (Nt, nx_t,   Nz)
    wf_t  = permutedims(ds["w"][ix_a:ix_b,   :, Isel], (3,1,2));   # (Nt, nx_t,   Nz+1)
    bc_t  = permutedims(ds["b"][ix_a:ix_b,   :, Isel], (3,1,2));   # (Nt, nx_t,   Nz)
    pHY_t = permutedims(ds["pHY"][ix_a:ix_b, :, Isel], (3,1,2));
    pNH_t = permutedims(ds["pNHS"][ix_a:ix_b,:, Isel], (3,1,2));

    # cell-center velocities
    uc_t = uf_t[:,1:end-1,:]/2 .+ uf_t[:,2:end,:]/2;  # (Nt, nx_t, Nz)
    wc_t = wf_t[:,:,1:end-1]/2 .+ wf_t[:,:,2:end]/2;  # (Nt, nx_t, Nz)
    uf_t = nothing; wf_t = nothing; GC.gc()

    # pressure perturbation (local in x: time-mean and depth-mean removal) ---
    ptot_t   = (pHY_t .+ pNH_t) * rho0/grav;
    pHY_t = nothing; pNH_t = nothing; GC.gc()
    ptota_t  = mean(ptot_t, dims=1);
    ptotp_t  = ptot_t .- ptota_t;
    ptot_t   = nothing;
    ptotpa_t = sum(ptotp_t .* dzz, dims=3) / H;
    pcp_t    = ptotp_t .- ptotpa_t;
    ptotp_t  = nothing; GC.gc()

    # density
    rhop_t = -bc_t * rho0/grav .+ rrr_shape;

    # track time-mean flow
    uca_full[ix_a:ix_b,:] = dropdims(mean(uc_t, dims=1), dims=1);

    # filter: high-pass (supertidal) -----------------------------------------
    passflg = "high";
    uh_t = zeros(size(uc_t)); vh_t = zeros(size(vc_t))
    wh_t = zeros(size(wc_t)); ph_t = zeros(size(pcp_t)); bh_t = zeros(size(bc_t))
    Threads.@threads for ix = 1:nx_t
        for iz = 1:Nz
            uh_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz], Tcut2, dt, Nf, passflg)
            vh_t[:,ix,iz] = lowhighpass_butter(vc_t[:,ix,iz], Tcut2, dt, Nf, passflg)
            wh_t[:,ix,iz] = lowhighpass_butter(wc_t[:,ix,iz], Tcut2, dt, Nf, passflg)
            ph_t[:,ix,iz] = lowhighpass_butter(pcp_t[:,ix,iz], Tcut2, dt, Nf, passflg)
            bh_t[:,ix,iz] = lowhighpass_butter(bc_t[:,ix,iz], Tcut2, dt, Nf, passflg)
        end
    end
    rh_t = -bh_t * rho0/grav;

    # filter: low-pass (subtidal) --------------------------------------------
    passflg = "low";
    us_t = zeros(size(uc_t)); vs_t = zeros(size(vc_t))
    ws_t = zeros(size(wc_t)); ps_t = zeros(size(pcp_t)); bs_t = zeros(size(bc_t))
    Threads.@threads for ix = 1:nx_t
        for iz = 1:Nz
            us_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz], Tcut1, dt, Nf, passflg)
            vs_t[:,ix,iz] = lowhighpass_butter(vc_t[:,ix,iz], Tcut1, dt, Nf, passflg)
            ws_t[:,ix,iz] = lowhighpass_butter(wc_t[:,ix,iz], Tcut1, dt, Nf, passflg)
            ps_t[:,ix,iz] = lowhighpass_butter(pcp_t[:,ix,iz], Tcut1, dt, Nf, passflg)
            bs_t[:,ix,iz] = lowhighpass_butter(bc_t[:,ix,iz], Tcut1, dt, Nf, passflg)
        end
    end
    rs_t = -bs_t * rho0/grav;

    # tidal band = (total - highpass) - subtidal
    ut_t = (uc_t .- uh_t) .- us_t
    vt_t = (vc_t .- vh_t) .- vs_t
    wt_t = (wc_t .- wh_t) .- ws_t
    pt_t = (pcp_t .- ph_t) .- ps_t
    bt_t = (bc_t .- bh_t) .- bs_t
    rt_t = -bt_t * rho0/grav;

    # KE and advective KE flux -----------------------------------------------
    KEz_t = uc_t[Iday,:,:].^2 .+ vc_t[Iday,:,:].^2 .+ wc_t[Iday,:,:].^2
    KE[ix_a:ix_b]  = fact*dropdims(mean(sum(KEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FKx[ix_a:ix_b] = dropdims(mean(sum(uc_t[Iday,:,:].*KEz_t.*dzz,dims=3),dims=1),dims=(1,3))

    KEz_t = ut_t[Iday,:,:].^2 .+ vt_t[Iday,:,:].^2 .+ wt_t[Iday,:,:].^2
    KEt[ix_a:ix_b]  = fact*dropdims(mean(sum(KEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FKxt[ix_a:ix_b] = dropdims(mean(sum(ut_t[Iday,:,:].*KEz_t.*dzz,dims=3),dims=1),dims=(1,3))

    KEz_t = uh_t[Iday,:,:].^2 .+ vh_t[Iday,:,:].^2 .+ wh_t[Iday,:,:].^2
    KEh[ix_a:ix_b]  = fact*dropdims(mean(sum(KEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FKxh[ix_a:ix_b] = dropdims(mean(sum(uh_t[Iday,:,:].*KEz_t.*dzz,dims=3),dims=1),dims=(1,3))

    KEz_t = us_t[Iday,:,:].^2 .+ vs_t[Iday,:,:].^2 .+ ws_t[Iday,:,:].^2
    KEs[ix_a:ix_b]  = fact*dropdims(mean(sum(KEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FKxs[ix_a:ix_b] = dropdims(mean(sum(us_t[Iday,:,:].*KEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    KEz_t = nothing; GC.gc()

    # APE (Kang & Fringer 2010 eq2) and advective APE flux ------------------
    APEz_t, Zz_t = APEKFeq2(rhop_t[Iday,:,:], rhorefc, zc, grav, thresh)
    APE[ix_a:ix_b] = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FAx[ix_a:ix_b] = dropdims(mean(sum(uc_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    Zz_t = nothing

    APEz_t, Zzt_t = APEKFeq2(rt_t[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
    APEt[ix_a:ix_b]  = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FAxt[ix_a:ix_b]  = dropdims(mean(sum(ut_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    Zzt_mid[ix_a:ix_b,:] = Zzt_t[imid,:,:]   # snapshot at mid-time
    A0nl = max(A0nl, maximum(Zzt_t[max(1,imid-100):min(Nlen,imid+100),:,:]))
    Zzt_t = nothing

    APEz_t, Zz_t = APEKFeq2(rh_t[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
    APEh[ix_a:ix_b]  = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FAxh[ix_a:ix_b]  = dropdims(mean(sum(uh_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    Zz_t = nothing

    APEz_t, Zz_t = APEKFeq2(rs_t[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
    APEs[ix_a:ix_b]  = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    FAxs[ix_a:ix_b]  = dropdims(mean(sum(us_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
    APEz_t = nothing; Zz_t = nothing; GC.gc()

    # linear APE
    APElin[ix_a:ix_b] = dropdims(mean(sum((bc_t[Iday,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3),dims=1),dims=(1,3))

    # pressure flux ----------------------------------------------------------
    Fx[ix_a:ix_b]  = dropdims(mean(sum(uc_t[Iday,:,:].*pcp_t[Iday,:,:].*dzz,dims=3),dims=1),dims=(1,3))
    Fxt[ix_a:ix_b] = dropdims(mean(sum(ut_t[Iday,:,:].*pt_t[Iday,:,:].*dzz,dims=3),dims=1),dims=(1,3))
    Fxh[ix_a:ix_b] = dropdims(mean(sum(uh_t[Iday,:,:].*ph_t[Iday,:,:].*dzz,dims=3),dims=1),dims=(1,3))
    Fxs[ix_a:ix_b] = dropdims(mean(sum(us_t[Iday,:,:].*ps_t[Iday,:,:].*dzz,dims=3),dims=1),dims=(1,3))

    # clear tile memory
    uc_t=nothing; vc_t=nothing; wc_t=nothing; bc_t=nothing; pcp_t=nothing; rhop_t=nothing
    uh_t=nothing; vh_t=nothing; wh_t=nothing; ph_t=nothing; bh_t=nothing; rh_t=nothing
    us_t=nothing; vs_t=nothing; ws_t=nothing; ps_t=nothing; bs_t=nothing; rs_t=nothing
    ut_t=nothing; vt_t=nothing; wt_t=nothing; pt_t=nothing; bt_t=nothing; rt_t=nothing
    GC.gc()
end

close(ds)

# post-tile scalar computations ----------------------------------------------
I1  = argmin(abs.(zc .- -100))
I2  = argmin(abs.(zc .- -300))
dnl = (zc[I1] - zc[I2]) / log(N2c[I1]/N2c[I2])

alpnl        = A0nl/dnl
alpepshy     = alpnl/epshy
alpepsnh     = alpnl/epsnh
Tbeatnh_days = 2π/(epsnh*ω)/(24*3600)
Tbeathy_days = 2π/(epshy*ω)/(24*3600)

# hovmuller (using surface velocities loaded before tile loop) ---------------
fig1 = Figure(size=(600,600))
clims = (0,0.6)
ax = Axis(fig1[1,1],title=string(fname_short2,"; lat=",LAT,"; KE [m2/s2]"),xlabel="x [km]",ylabel="time [days]")
hm = heatmap!(ax, xc/1e3, tday, transpose(uc_surf.^2 .+ vc_surf.^2), colormap=Reverse(:Spectral), colorrange=clims)
Colorbar(fig1[1,2], hm)
fig1
if figflag==1; save(string(dirfig,"KE_hovmuller_",fname_short2,".png"), fig1); end

# mean flow plot (using uca_full accumulated across tiles) -------------------
fig2 = Figure(size=(1000,750))
axa = Axis(fig2[1,1],title="mean flow [m/s]",xlabel="x [km]",ylabel="z [m]")
hm = heatmap!(axa, xc/1e3, zc, uca_full, colormap=Reverse(:Spectral))
Colorbar(fig2[1,2], hm)
hm.colorrange = (-0.02, 0.02)
fig2

# Zzt snapshot heatmap (midpoint in time, accumulated across tiles) ----------
fig3 = Figure(size=(1000,750))
axa = Axis(fig3[1,1],title="zeta [m]",xlabel="x [km]",ylabel="z [m]")
hm = heatmap!(axa, xc/1e3, zc, Zzt_mid, colormap=Reverse(:Spectral))
Colorbar(fig3[1,2], hm)
hm.colorrange = (-100, 100)
fig3

# KE/APE/flux figures --------------------------------------------------------
ylimE = [0 30]; ylimA = [0 30]; ylimf = [-1 8];

fig4 = Figure(size=(750,750))
ax = Axis(fig4[1,1],title=string(fname_short2,"; lat=",LAT,"; KE [kJ/m2]"),xlabel="x [km]",ylabel="KE [kJ/m2]")
lines!(ax, xc/1e3, (KEt+KEh)/1e3, label="t+HH", color=:black, linewidth=3)
lines!(ax, xc/1e3, KE/1e3,   label="tot",   linestyle=:dash,  color=:grey,   linewidth=3)
lines!(ax, xc/1e3, KEt/1e3,  label="tidal", color=:red,    linewidth=3)
lines!(ax, xc/1e3, KEh/1e3,  label="HH",    color=:green,  linewidth=3)
lines!(ax, xc/1e3, KEs/1e3,  label="sub",   color=:yellow, linewidth=2)
xlims!(ax, 0, Ldom/1e3); ylims!(ax, ylimE[1], ylimE[2])

ax2 = Axis(fig4[2,1],title="APE",xlabel="x [km]",ylabel="APE [kJ/m2]")
lines!(ax2, xc/1e3, (APEt+APEh)/1e3, label="t+HH", color=:black, linewidth=3)
lines!(ax2, xc/1e3, APE/1e3,   label="tot",      linestyle=:dash, color=:grey,  linewidth=3)
lines!(ax2, xc/1e3, APEt/1e3,  label="tidal",    color=:red,   linewidth=3)
lines!(ax2, xc/1e3, APEh/1e3,  label="HH",       color=:green, linewidth=3)
lines!(ax2, xc/1e3, APElin/1e3,label="tot lin",  color=:grey,  linewidth=1)
xlims!(ax2, 0, Ldom/1e3); ylims!(ax2, ylimA[1], ylimA[2])

ax3 = Axis(fig4[3,1],title="flux",xlabel="x [km]",ylabel="flux [W/m]")
lines!(ax3, xc/1e3, (Fxt+Fxh)/1e3,          label="t+HH",   color=:black, linewidth=3)
lines!(ax3, xc/1e3, Fx/1e3,                  label="tot",    linestyle=:dash, color=:grey,  linewidth=3)
lines!(ax3, xc/1e3, Fxt/1e3,                 label="tidal",  color=:red,   linewidth=3)
lines!(ax3, xc/1e3, Fxh/1e3,                 label="HH",     color=:green, linewidth=3)
lines!(ax3, xc/1e3, (FKxt+FAxt+FKxh+FAxh)/1e3, linestyle=:dash, color=:black, linewidth=3)
lines!(ax3, xc/1e3, (FKx+FAx)/1e3,           linestyle=:dash, color=:grey,  linewidth=3)
lines!(ax3, xc/1e3, (Fx+FKx+FAx)/1e3,        label="up+uE", linestyle=:dash, color=:blue,  linewidth=3)
lines!(ax3, xc/1e3, (FKxt+FAxt)/1e3,          linestyle=:dash, color=:red,   linewidth=3)
lines!(ax3, xc/1e3, (FKxh+FAxh)/1e3,          linestyle=:dash, color=:green, linewidth=3)
xlims!(ax3, 0, Ldom/1e3); ylims!(ax3, ylimf[1], ylimf[2])
axislegend(ax3, position=:rt)
fig4
if figflag==1; save(string(dirfig,"KE_flux_",fname_short2,".png"), fig4); end

println(fnames,"; max total flux is ",@sprintf("%5.2f",maximum(Fxt/1e3))," kW/m")
println(fnames,"; max D2+HH flux is ",@sprintf("%5.2f",maximum(Fx/1e3))," kW/m")

# FFT spectra of surface velocity (uses uc_surf/vc_surf loaded before tile loop)
tukeycf=0.2; numwin=1; linfit=true; prewhit=false;

i=1;
period, freq, pp = fft_spectra(tday[Iday], uc_surf[Iday,i]; tukeycf, numwin, linfit, prewhit);
poweru = zeros(length(period),Nx);
powerv = zeros(length(period),Nx);
for i in 1:Nx
    period, freq, poweru[:,i] = fft_spectra(tday[Iday], uc_surf[Iday,i]; tukeycf, numwin, linfit, prewhit);
    period, freq, powerv[:,i] = fft_spectra(tday[Iday], vc_surf[Iday,i]; tukeycf, numwin, linfit, prewhit);
end

println("max freq: ",freq[end]," cpd")
KEom = poweru .+ powerv;

fmax = 48; fstp = 2;
flim = [0 fmax];

fig5 = Figure(size=(500,750))
axa = Axis(fig5[1,1],xticks=(flim[1]:fstp:flim[2]),
    title=string(fname_short2,"; lat=",LAT,"; log10(KE) [m2/s2/cpd] "),
    xlabel="frequency [cpd]",ylabel="x [km]")
xlims!(axa, flim[1], flim[2])
hm = heatmap!(axa, freq, xc/1e3, log10.(KEom), colormap=Reverse(:Spectral))
Colorbar(fig5[1,2], hm)
hm.colorrange = (-6, 0)
fig5

xlims_fft = [75,1800]*1e3;
Ix     = findall(item -> item >= xlims_fft[1] && item<= xlims_fft[2], xc);
KEoma  = vec(mean(KEom[:,Ix],dims=2));
KEommax, maxidx = findmax(KEom)
println("KEomax=",log10(KEommax))

flim2 = [0 fmax]; Plims = [-14 1];
axb = Axis(fig5[2,1],xticks=(flim[1]:fstp:flim[2]), yscale=log10,
    title="normalized power",xlabel="frequency [cpd]",ylabel="KE/KEmax")
xlims!(axb, flim2[1], flim2[2])
ylims!(axb, 10.0^Plims[1], 10.0^Plims[2])
lines!(axb, freq, KEoma./KEommax, linestyle=:solid, color=:black, linewidth=3)
fcpd = coriolis(LAT)/(2*pi)*24*3600
lines!(axb, vec([fcpd fcpd]), [10.0^Plims[1], 10.0^Plims[2]], linestyle=:dash, color=:red, linewidth=2)
fig5
if figflag==1; save(string(dirfig,"fft_usur_",fname_short2,".png"), fig5); end

# save energy terms ----------------------------------------------------------
if savefl == 1
    fnameout = string("energetics_",fname_short2,".jld2")
    jldsave(string(dirout,fnameout);
        dnl, A0nl, alpnl, alpepshy, alpepsnh, Tbeatnh_days, Tbeathy_days,
        epsnh, epshy, epsomnh, epsomhy,
        xc, freq, KEoma, KEommax, Fx, Fxt, Fxh, Fxs,
        FAx, FAxt, FAxh, FAxs, FKx, FKxt, FKxh, FKxs,
        KE, KEt, KEh, KEs, APE, APEt, APEh, APEs);
    println(string(fnameout)," data saved ........ ")
end

end # function run_analysis(runnm,LAT,savefl)


# runnms loop ---------------
elapsed = @elapsed begin
    for (runnm, LAT) in zip(runnms, LATS)
        looptime = @elapsed begin
            run_analysis(runnm,LAT,savefl)
        end
        println("finished ", runnm," in $(round(looptime, digits=1)) s")        
    end 
end
println("finished in $(round(elapsed, digits=1)) s")


stop()


# ====================================================================
# ====================================================================
# ====================================================================

##

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
#const rho0=1020; const grav=9.81; 
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


## create some figures ----------------------------------------------

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

jldsave(string(dirout,fnameout); freq, KEoma, KEommax, xc, Fxt, Fx, Fxh, Fxl, Fx2, KEt, APEt, KE, APE, 
       KEh, APEh, KEl, APEl, KE2, KEut, KEu, KEuh, KEul);
println(string(fnameout)," data saved ........ ")
