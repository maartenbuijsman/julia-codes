#= IW_total_energetics_tile.jl
Maarten Buijsman, USM DMS, 2026-8-21
Compute undecomposed energetics: KE, APE, and pressure fluxes
for total and high-passed fields
Oceananigans pressure is kinematic p_kin = p_true/rho0
buoyancy = -g*rho'/rho0
Non-dimensional parameters/time scales moved to IW_nondim_params.jl
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
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

# Flags --------------------------
savefl  = 1  # save data
figflag = 1  # print figures
oldnm   = 0  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing

# tiles
ntile   = 20
#ntile   = 10

const T2 = 12+25.2/60
const rho0=1020; 
const grav=9.81; 

# run-ID selection: only mainnm + runnms need to be prescribed here; LAT and
# the N2 stratification profile are looked up from run_master.jl, so run-ID
# and latitude can never drift out of sync. runnms need not be a full block --
# any subset of run-IDs already present in RUN_TABLE works.

# D2 NH flux forcing
mainnm  = 11
#runnms  = collect(1:14)  # constant N2 WOCE AMZ
#runnms  = collect(27:39) # varying  N2 MERCATOR      25kW/m
#runnms  = collect(40:52) # constant N2 MERCATOR 2.5N 25kW/m
#runnms  = collect(53:65) # constant N2 MERCATOR 50N  25kW/m 
#runnms  = collect(66:78) # varying  N2 MERCATOR;      50kW/m

#test
#mainnm  = 10
#runnms  = 66

runs = get_runs(mainnm, runnms)   # errors immediately if a runnm isn't in RUN_TABLE
LATS = [r.lat for r in runs]

# do the analysis in a function
function run_analysis(runnm,LAT,savefl)
# IS = 1; runnm = runnms[IS]; LAT = LATS[IS]

# filename
fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
fname_short2 = fnames
filename = string(dirsim,fnames,".nc")

println(fname_short2,"; lat=",LAT," -------------------")

# look up this run's metadata (lat/Flux/DX/N2 source) from the master table
row = get_runs(mainnm, [runnm])[1]

# open nc file and keep open for tiled reads
ds = NCDataset(filename,"r");
println(keys(ds))

# only select data after the spinup time
tspin = 10; #days  # for 2000 km domain

# set E and flux lims for figures, keyed off this run's forcing flux
Elim, Flim = elim_flim(row)

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
fnamegrid = n2_filename(row)
path_fname = string(dirforce,fnamegrid);
@load path_fname N2w zfw
N2c = N2w[1:end-1]/2 + N2w[2:end]/2;

fig = Figure()
ax1 = Axis(fig[1,1], title=fnamegrid, xlabel="N² (s⁻²)", ylabel="z [m]")
lines!(ax1,N2c, zc)
ylims!(ax1, -500, 0)
fig

# resonance parameter epsilon, nondim parameters, and A0nl/timescales moved to
# IW_nondim_params.jl (run it separately; it saves nondim_AMZexptXX.YY.jld2)

# reference density ----------------------------------------------------------
# compute reference density profile
# b = sum N2 * dz = sum -g/rho0*drho/dz * dz
# b = -g/rho0*rho_pert
# rho_pert = -b*rho0/g 
breff   = cumtrapz(zfw, N2w);                              # bottom up!
intzc   = interpolate((zfw,), breff, Gridded(Linear()));
rhorefc = -intzc.(zc) * rho0/grav;                         # rho0 is not added!

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

# filter coefficients (Tcut1/Tcut2/dt/Nf are fixed for the whole run, design once)
b_high = digitalfilter(Highpass(1/Tcut2), Butterworth(Nf); fs=1/dt)
b_low  = digitalfilter(Lowpass(1/Tcut1),  Butterworth(Nf); fs=1/dt)

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
uf_surf_tmp = nothing
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
fact     = 1/2*rho0

# tile loop (nhalo=0: no x-gradients needed) ---------------------------------
nx_base = Nx ÷ ntile

# per-tile phase timers (diagnostics for locating the runtime bottleneck)
tim_io    = zeros(ntile)
tim_prep  = zeros(ntile)
tim_filth = zeros(ntile)
tim_filtl = zeros(ntile)
tim_ke    = zeros(ntile)
tim_ape   = zeros(ntile)
tim_press = zeros(ntile)

println("starting tile loop, ntile=",ntile)
for i_tile in 1:ntile
    ix_a = (i_tile-1)*nx_base + 1
    ix_b = (i_tile == ntile) ? Nx : i_tile*nx_base
    nx_t = ix_b - ix_a + 1
    println("  tile ",i_tile,"/",ntile,"  ix=",ix_a,":",ix_b)

    # load tile data ---------------------------------------------------------
    # u on x-faces: need one extra face at ix_b+1 to center to ix_b
    tim_io[i_tile] = @elapsed begin
        uf_t  = permutedims(ds["u"][ix_a:ix_b+1, :, Isel], (3,1,2));   # (Nt, nx_t+1, Nz)
        vc_t  = permutedims(ds["v"][ix_a:ix_b,   :, Isel], (3,1,2));   # (Nt, nx_t,   Nz)
        wf_t  = permutedims(ds["w"][ix_a:ix_b,   :, Isel], (3,1,2));   # (Nt, nx_t,   Nz+1)
        bc_t  = permutedims(ds["b"][ix_a:ix_b,   :, Isel], (3,1,2));   # (Nt, nx_t,   Nz)
        pHY_t = permutedims(ds["pHY"][ix_a:ix_b, :, Isel], (3,1,2));
        pNH_t = permutedims(ds["pNHS"][ix_a:ix_b,:, Isel], (3,1,2));
    end

    tim_prep[i_tile] = @elapsed begin
        Nt_here = size(bc_t,1)
        uc_t   = zeros(eltype(uf_t),  Nt_here, nx_t, Nz)
        wc_t   = zeros(eltype(wf_t),  Nt_here, nx_t, Nz)
        pcp_t  = zeros(eltype(pHY_t), Nt_here, nx_t, Nz)
        rhop_t = zeros(Float64,       Nt_here, nx_t, Nz)

        # x-chunked across threads: each column is independent (no x-gradients needed)
        nxchunks = Threads.nthreads()
        bounds = round.(Int, range(0, nx_t, length=nxchunks+1))
        Threads.@threads for c in 1:nxchunks
            rng = (bounds[c]+1):bounds[c+1]
            isempty(rng) && continue
            gx = ix_a .+ rng .- 1

            # cell-center velocities
            uc_t[:,rng,:] = uf_t[:,rng,:]/2 .+ uf_t[:,rng.+1,:]/2
            wc_t[:,rng,:] = wf_t[:,rng,1:end-1]/2 .+ wf_t[:,rng,2:end]/2

            # pressure perturbation (local in x: time-mean and depth-mean removal) ---
            ptot_c   = (pHY_t[:,rng,:] .+ pNH_t[:,rng,:]) * rho0;   # kinematic (m²/s²) → dynamic pressure [Pa]
            ptota_c  = mean(ptot_c, dims=1);
            ptotp_c  = ptot_c .- ptota_c;
            ptotpa_c = sum(ptotp_c .* dzz, dims=3) / H;
            pcp_t[:,rng,:] = ptotp_c .- ptotpa_c

            # density
            rhop_t[:,rng,:] = -bc_t[:,rng,:] * rho0/grav .+ rrr_shape;

            # track time-mean flow
            uca_full[gx,:] = dropdims(mean(uc_t[:,rng,:], dims=1), dims=1);
        end

        uf_t = nothing; wf_t = nothing; pHY_t = nothing; pNH_t = nothing
    end

    # filter: high-pass (supertidal) -----------------------------------------
    uh_t = zeros(size(uc_t)); vh_t = zeros(size(vc_t))
    wh_t = zeros(size(wc_t)); ph_t = zeros(size(pcp_t)); bh_t = zeros(size(bc_t))
    tim_filth[i_tile] = @elapsed begin
        ncols = nx_t*Nz
        nchunks = Threads.nthreads()
        bounds = round.(Int, range(0, ncols, length=nchunks+1))
        uc_flat  = reshape(uc_t,  Nt_here, ncols); vc_flat = reshape(vc_t, Nt_here, ncols)
        wc_flat  = reshape(wc_t,  Nt_here, ncols); pcp_flat = reshape(pcp_t, Nt_here, ncols)
        bc_flat  = reshape(bc_t,  Nt_here, ncols)
        uh_flat  = reshape(uh_t,  Nt_here, ncols); vh_flat = reshape(vh_t, Nt_here, ncols)
        wh_flat  = reshape(wh_t,  Nt_here, ncols); ph_flat = reshape(ph_t, Nt_here, ncols)
        bh_flat  = reshape(bh_t,  Nt_here, ncols)
        Threads.@threads for c in 1:nchunks
            rng = (bounds[c]+1):bounds[c+1]
            isempty(rng) && continue
            uh_flat[:,rng] = filtfilt(b_high, uc_flat[:,rng])
            vh_flat[:,rng] = filtfilt(b_high, vc_flat[:,rng])
            wh_flat[:,rng] = filtfilt(b_high, wc_flat[:,rng])
            ph_flat[:,rng] = filtfilt(b_high, pcp_flat[:,rng])
            bh_flat[:,rng] = filtfilt(b_high, bc_flat[:,rng])
        end
    end
    rh_t = -bh_t * rho0/grav;

    # filter: low-pass (subtidal) --------------------------------------------
    us_t = zeros(size(uc_t)); vs_t = zeros(size(vc_t))
    ws_t = zeros(size(wc_t)); ps_t = zeros(size(pcp_t)); bs_t = zeros(size(bc_t))
    tim_filtl[i_tile] = @elapsed begin
        ncols = nx_t*Nz
        nchunks = Threads.nthreads()
        bounds = round.(Int, range(0, ncols, length=nchunks+1))
        uc_flat  = reshape(uc_t,  Nt_here, ncols); vc_flat = reshape(vc_t, Nt_here, ncols)
        wc_flat  = reshape(wc_t,  Nt_here, ncols); pcp_flat = reshape(pcp_t, Nt_here, ncols)
        bc_flat  = reshape(bc_t,  Nt_here, ncols)
        us_flat  = reshape(us_t,  Nt_here, ncols); vs_flat = reshape(vs_t, Nt_here, ncols)
        ws_flat  = reshape(ws_t,  Nt_here, ncols); ps_flat = reshape(ps_t, Nt_here, ncols)
        bs_flat  = reshape(bs_t,  Nt_here, ncols)
        Threads.@threads for c in 1:nchunks
            rng = (bounds[c]+1):bounds[c+1]
            isempty(rng) && continue
            us_flat[:,rng] = filtfilt(b_low, uc_flat[:,rng])
            vs_flat[:,rng] = filtfilt(b_low, vc_flat[:,rng])
            ws_flat[:,rng] = filtfilt(b_low, wc_flat[:,rng])
            ps_flat[:,rng] = filtfilt(b_low, pcp_flat[:,rng])
            bs_flat[:,rng] = filtfilt(b_low, bc_flat[:,rng])
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
    tim_ke[i_tile] = @elapsed begin
        nxchunks = Threads.nthreads()
        bounds = round.(Int, range(0, nx_t, length=nxchunks+1))
        Threads.@threads for c in 1:nxchunks
            rng = (bounds[c]+1):bounds[c+1]
            isempty(rng) && continue
            gx = ix_a .+ rng .- 1

            KEz = uc_t[Iday,rng,:].^2 .+ vc_t[Iday,rng,:].^2 .+ wc_t[Iday,rng,:].^2
            KE[gx]  = fact*dropdims(mean(sum(KEz.*dzz,dims=3),dims=1),dims=(1,3))
            FKx[gx] = fact*dropdims(mean(sum(uc_t[Iday,rng,:].*KEz.*dzz,dims=3),dims=1),dims=(1,3))

            KEz = ut_t[Iday,rng,:].^2 .+ vt_t[Iday,rng,:].^2 .+ wt_t[Iday,rng,:].^2
            KEt[gx]  = fact*dropdims(mean(sum(KEz.*dzz,dims=3),dims=1),dims=(1,3))
            FKxt[gx] = fact*dropdims(mean(sum(ut_t[Iday,rng,:].*KEz.*dzz,dims=3),dims=1),dims=(1,3))

            KEz = uh_t[Iday,rng,:].^2 .+ vh_t[Iday,rng,:].^2 .+ wh_t[Iday,rng,:].^2
            KEh[gx]  = fact*dropdims(mean(sum(KEz.*dzz,dims=3),dims=1),dims=(1,3))
            FKxh[gx] = fact*dropdims(mean(sum(uh_t[Iday,rng,:].*KEz.*dzz,dims=3),dims=1),dims=(1,3))

            KEz = us_t[Iday,rng,:].^2 .+ vs_t[Iday,rng,:].^2 .+ ws_t[Iday,rng,:].^2
            KEs[gx]  = fact*dropdims(mean(sum(KEz.*dzz,dims=3),dims=1),dims=(1,3))
            FKxs[gx] = fact*dropdims(mean(sum(us_t[Iday,rng,:].*KEz.*dzz,dims=3),dims=1),dims=(1,3))
        end
    end

    # APE (Kang & Fringer 2010 eq2) and advective APE flux ------------------
    tim_ape[i_tile] = @elapsed begin
        # total
        APEz_t, Zz_t = APEKFeq2(rhop_t[Iday,:,:], rhorefc, zc, grav, thresh)
        APE[ix_a:ix_b] = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        FAx[ix_a:ix_b] = dropdims(mean(sum(uc_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        Zz_t = nothing

        # tidal, D2
        APEz_t, Zzt_t = APEKFeq2(rt_t[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
        APEt[ix_a:ix_b]  = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        FAxt[ix_a:ix_b]  = dropdims(mean(sum(ut_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        Zzt_mid[ix_a:ix_b,:] = Zzt_t[imid,:,:]   # snapshot at mid-time
        Zzt_t = nothing

        # HH
        APEz_t, Zz_t = APEKFeq2(rh_t[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
        APEh[ix_a:ix_b]  = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        FAxh[ix_a:ix_b]  = dropdims(mean(sum(uh_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        Zz_t = nothing

        # subtidal
        APEz_t, Zz_t = APEKFeq2(rs_t[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
        APEs[ix_a:ix_b]  = dropdims(mean(sum(APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        FAxs[ix_a:ix_b]  = dropdims(mean(sum(us_t[Iday,:,:].*APEz_t.*dzz,dims=3),dims=1),dims=(1,3))
        APEz_t = nothing; Zz_t = nothing

        # linear APE
        APElin[ix_a:ix_b] = dropdims(mean(sum((bc_t[Iday,:,Ikp].^2).*factA[:,:,Ikp].*dzz[:,:,Ikp],dims=3),dims=1),dims=(1,3))
    end

    # pressure flux ----------------------------------------------------------
    tim_press[i_tile] = @elapsed begin
        nxchunks = Threads.nthreads()
        bounds = round.(Int, range(0, nx_t, length=nxchunks+1))
        Threads.@threads for c in 1:nxchunks
            rng = (bounds[c]+1):bounds[c+1]
            isempty(rng) && continue
            gx = ix_a .+ rng .- 1

            Fx[gx]  = dropdims(mean(sum(uc_t[Iday,rng,:].*pcp_t[Iday,rng,:].*dzz,dims=3),dims=1),dims=(1,3))
            Fxt[gx] = dropdims(mean(sum(ut_t[Iday,rng,:].*pt_t[Iday,rng,:].*dzz,dims=3),dims=1),dims=(1,3))
            Fxh[gx] = dropdims(mean(sum(uh_t[Iday,rng,:].*ph_t[Iday,rng,:].*dzz,dims=3),dims=1),dims=(1,3))
            Fxs[gx] = dropdims(mean(sum(us_t[Iday,rng,:].*ps_t[Iday,rng,:].*dzz,dims=3),dims=1),dims=(1,3))
        end
    end

    println("    io=",round(tim_io[i_tile],digits=1)," prep=",round(tim_prep[i_tile],digits=1),
            " filt_high=",round(tim_filth[i_tile],digits=1)," filt_low=",round(tim_filtl[i_tile],digits=1),
            " KE=",round(tim_ke[i_tile],digits=1)," APE=",round(tim_ape[i_tile],digits=1),
            " press=",round(tim_press[i_tile],digits=1)," s")

    # clear tile memory
    uc_t=nothing; vc_t=nothing; wc_t=nothing; bc_t=nothing; pcp_t=nothing; rhop_t=nothing
    uh_t=nothing; vh_t=nothing; wh_t=nothing; ph_t=nothing; bh_t=nothing; rh_t=nothing
    us_t=nothing; vs_t=nothing; ws_t=nothing; ps_t=nothing; bs_t=nothing; rs_t=nothing
    ut_t=nothing; vt_t=nothing; wt_t=nothing; pt_t=nothing; bt_t=nothing; rt_t=nothing
    GC.gc()
end

close(ds)

println("phase totals across all tiles [s]: io=",round(sum(tim_io),digits=1),
        " prep=",round(sum(tim_prep),digits=1)," filt_high=",round(sum(tim_filth),digits=1),
        " filt_low=",round(sum(tim_filtl),digits=1)," KE=",round(sum(tim_ke),digits=1),
        " APE=",round(sum(tim_ape),digits=1)," press=",round(sum(tim_press),digits=1))

# post-tile scalar computations ----------------------------------------------
# (nondim parameters dnl/alpnl/Tbeat moved to IW_nondim_params.jl)

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
#display(fig3)
println("max D2 displacement is ", maximum(filter(!isnan, Zzt_mid)))
println("min D2 displacement is ", minimum(filter(!isnan, Zzt_mid)))

# KE/APE/flux figures --------------------------------------------------------
# yaxis lims are determined above

fig4 = Figure(size=(750,750))
ax = Axis(fig4[1,1],title=string(fname_short2,"; lat=",LAT,"; KE [kJ/m2]"),xlabel="x [km]",ylabel="KE [kJ/m2]")
lines!(ax, xc/1e3, KE/1e3,        label="tot",                    color=:orange, linewidth=7, alpha=1)
lines!(ax, xc/1e3, (KEt+KEh)/1e3, label="D2+HH",                  color=:black,  linewidth=3)                    
lines!(ax, xc/1e3, KEt/1e3,       label="D2",                     color=:red,    linewidth=3)  
lines!(ax, xc/1e3, KEh/1e3,       label="HH",                     color=:green,  linewidth=3)  
lines!(ax, xc/1e3, KEs/1e3,       label="sub",                    color=:purple, linewidth=3)
xlims!(ax, 0, Ldom/1e3); ylims!(ax, Elim[1], Elim[2])
axislegend(ax, position=:rt, labelsize=9, rowgap=0) #, framevisible=false)

ax2 = Axis(fig4[2,1],title="APE",xlabel="x [km]",ylabel="APE [kJ/m2]")
lines!(ax2, xc/1e3, APE/1e3,                                      color=:orange,linewidth=7, alpha=1)    #,  label="tot"
lines!(ax2, xc/1e3, (APEt+APEh)/1e3,                              color=:black, linewidth=3)    #,  label="t+HH"
lines!(ax2, xc/1e3, APEt/1e3,                                     color=:red,   linewidth=3)    #,  label="tidal"
lines!(ax2, xc/1e3, APEh/1e3,                                     color=:green, linewidth=3)    #,  label="HH"
lines!(ax2, xc/1e3, APElin/1e3,   label="tot lin",linestyle=:dash,color=:cyan,  linewidth=1)
lines!(ax, xc/1e3,  APEs/1e3,                                     color=:purple,linewidth=3)
xlims!(ax2, 0, Ldom/1e3); ylims!(ax2, Elim[1], Elim[2])
axislegend(ax2, position=:rt, labelsize=9, rowgap=0)

ax3 = Axis(fig4[3,1],title="pressure (solid) and advective flux",xlabel="x [km]",ylabel="flux [kW/m]")
lines!(ax3, xc/1e3, Fx/1e3,                                                           color=:orange,linewidth=7, alpha=1)  # pressure flux: total
lines!(ax3, xc/1e3, (Fxt+Fxh)/1e3,                                                    color=:black, linewidth=3)  # pressure flux: tidal + HH
lines!(ax3, xc/1e3, Fxt/1e3,                                                          color=:red,   linewidth=3)  # pressure flux: tidal
lines!(ax3, xc/1e3, Fxh/1e3,                                                          color=:green, linewidth=3)  # pressure flux: supertidal HH
lines!(ax3, xc/1e3, (FKxt+FAxt+FKxh+FAxh)/1e3,label="Fadv   D2+HH",linestyle=:dashdot,color=:black, linewidth=3)  # KE+APE advective flux: tidal + HH
lines!(ax3, xc/1e3, (FKx+FAx)/1e3,            label="Fadv   tot",  linestyle=:dashdot,color=:orange,linewidth=3)  # KE+APE advective flux: total
lines!(ax3, xc/1e3, (FKxt+FAxt)/1e3,          label="Fadv   D2",   linestyle=:dashdot,color=:red,   linewidth=3)  # KE+APE advective flux: tidal
lines!(ax3, xc/1e3, (FKxh+FAxh)/1e3,          label="Fadv   HH",   linestyle=:dashdot,color=:green, linewidth=3)  # KE+APE advective flux: HH
lines!(ax3, xc/1e3, (Fx+FKx+FAx)/1e3,         label="Fp+adv tot",  linestyle=:dash,   color=:blue,  linewidth=3)  # total pressure + KE + APE flux
xlims!(ax3, 0, Ldom/1e3); ylims!(ax3,  Flim[1], Flim[2])

axislegend(ax3, position=:rt, labelsize=9, rowgap=0)
fig4
display(fig4)
if figflag==1; save(string(dirfig,"KE_flux_",fname_short2,".png"), fig4); end

println(fnames,"; max total flux is ",@sprintf("%5.2f",maximum(Fxt/1e3))," kW/m")
println(fnames,"; max D2+HH flux is ",@sprintf("%5.2f",maximum(Fx/1e3))," kW/m")

# FFT spectra of surface velocity (uses uc_surf/vc_surf loaded before tile loop)
tukeycf=0.2; numwin=1; linfit=true; prewhit=false;

i=1;
period, freq, pp = fft_spectra(tday[Iday], uc_surf[Iday,i]; tukeycf, numwin, linfit, prewhit);
poweru = zeros(length(period),Nx);
powerv = zeros(length(period),Nx);
Threads.@threads for i in 1:Nx
    _, _, poweru[:,i] = fft_spectra(tday[Iday], uc_surf[Iday,i]; tukeycf, numwin, linfit, prewhit);
    _, _, powerv[:,i] = fft_spectra(tday[Iday], vc_surf[Iday,i]; tukeycf, numwin, linfit, prewhit);
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
