#= IW_total_energetics_tile.jl
Maarten Buijsman, USM DMS, 2026-8-8 (2)
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

# Flags --------------------------
savefl  = 1  # save data
figflag = 1  # print figures
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
mainnm  = 9
#runnms  = collect(1:14) # constant N2 WOCE AMZ
runnms  = collect(15:28) # varying  N2 MERCATOR
#runnms  = collect(29:42) # constant N2 MERCATOR 2.5N
LATS    = vcat(collect(0:2.5:5), collect(10:5:60))

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
if mainnm == 9 && runnm <= 14
    fnamegrid = "N2_amz1.jld2"
elseif mainnm == 9 && runnm <= 28
    fnamegrid = @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", LAT)
elseif mainnm == 9 && runnm <= 42
    fnamegrid = "N2_ZonalMeanAtl_lat02.5.jld2"
else
    fnamegrid = "N2_amz1.jld2"
end
path_fname = string(dirforce,fnamegrid);
@load path_fname N2w zfw
N2c = N2w[1:end-1]/2 + N2w[2:end]/2;

fig = Figure()
ax1 = Axis(fig[1,1], title=fnamegrid, xlabel="N² (s⁻²)", ylabel="z [m]")
lines!(ax1,N2c, zc)
ylims!(ax1, -500, 0)
fig
display(fig)

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
lines!(ax, xc/1e3, (KEt+KEh)/1e3, color=:black, linewidth=3)                #,  label="t+HH" 
lines!(ax, xc/1e3, KE/1e3,  linestyle=:dash,  color=:grey,   linewidth=3)   #,  label="tot"
lines!(ax, xc/1e3, KEt/1e3, color=:red,    linewidth=3)                     #,  label="tidal"
lines!(ax, xc/1e3, KEh/1e3, color=:green,  linewidth=3)                     #,  label="HH"
lines!(ax, xc/1e3, KEs/1e3,  label="sub",   color=:yellow, linewidth=2)
xlims!(ax, 0, Ldom/1e3); ylims!(ax, ylimE[1], ylimE[2])
axislegend(ax, position=:rt, labelsize=8, rowgap=0, framevisible=false)

ax2 = Axis(fig4[2,1],title="APE",xlabel="x [km]",ylabel="APE [kJ/m2]")
lines!(ax2, xc/1e3, (APEt+APEh)/1e3, color=:black, linewidth=3)              #,  label="t+HH"
lines!(ax2, xc/1e3, APE/1e3,  linestyle=:dash, color=:grey,  linewidth=3)    #,  label="tot"
lines!(ax2, xc/1e3, APEt/1e3, color=:red,   linewidth=3)                     #,  label="tidal"
lines!(ax2, xc/1e3, APEh/1e3, color=:green, linewidth=3)                     #,  label="HH"
lines!(ax2, xc/1e3, APElin/1e3,label="tot lin",  color=:grey,  linewidth=1)
xlims!(ax2, 0, Ldom/1e3); ylims!(ax2, ylimA[1], ylimA[2])
axislegend(ax2, position=:rt, labelsize=8, rowgap=0, framevisible=false)

ax3 = Axis(fig4[3,1],title="flux",xlabel="x [km]",ylabel="flux [W/m]")
lines!(ax3, xc/1e3, (Fxt+Fxh)/1e3,           label="Fp t+HH",   color=:black, linewidth=3)   # pressure flux: tidal + HH
lines!(ax3, xc/1e3, Fx/1e3,                  label="Fp tot",    linestyle=:dash, color=:grey,  linewidth=3)  # pressure flux: total
lines!(ax3, xc/1e3, Fxt/1e3,                 label="Fp tidal",  color=:red,   linewidth=3)    # pressure flux: tidal
lines!(ax3, xc/1e3, Fxh/1e3,                 label="Fp HH",     color=:green, linewidth=3)    # pressure flux: supertidal HH
lines!(ax3, xc/1e3, (FKxt+FAxt+FKxh+FAxh)/1e3, label="FKE+APE t+HH",  linestyle=:dash, color=:black, linewidth=3)  # KE+APE advective flux: tidal + HH
lines!(ax3, xc/1e3, (FKx+FAx)/1e3,           label="FKE+APE tot",    linestyle=:dash, color=:orange,  linewidth=3)  # KE+APE advective flux: total
lines!(ax3, xc/1e3, (Fx+FKx+FAx)/1e3,        label="Fp+KE+APE tot", linestyle=:dash, color=:blue,  linewidth=3)  # total pressure + KE + APE flux
lines!(ax3, xc/1e3, (FKxt+FAxt)/1e3,         label="FKE+APE tidal",  linestyle=:dash, color=:red,   linewidth=3)  # KE+APE advective flux: tidal
lines!(ax3, xc/1e3, (FKxh+FAxh)/1e3,         label="FKE+APE HH",     linestyle=:dash, color=:green, linewidth=3)  # KE+APE advective flux: HH
xlims!(ax3, 0, Ldom/1e3); ylims!(ax3, ylimf[1], ylimf[2])
axislegend(ax3, position=:rt, labelsize=8, rowgap=0, framevisible=false)
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
