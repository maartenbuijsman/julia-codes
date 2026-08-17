#= IW_nondim_params.jl
Maarten Buijsman, USM DMS, 2026-8-16
Non-dimensional parameters and time scales for the D2 internal-tide PSI/resonance
problem (Sutherland & Dhaliwal 2022): resonance parameter epsilon (frequency- and
wavenumber-based, hydrostatic and nonhydrostatic), the nonlinearity length scale
dnl, the nonlinearity parameter alpha/epsilon, and the PSI beat period.
Split out of IW_total_energetics_tile.jl (was lines 126-183 and 385-397 there).

A0nl (max tidal-band vertical isopycnal displacement) is estimated from a SINGLE
x-column time series just east of the Gaussian forcing patch (see gausW_center/
gausW_width in IW_flux_LAT_2000km_bash_cuda.jl) -- the "parent wave" amplitude
close to the generation site, before it decays across the rest of the domain --
rather than a domain-wide max. That keeps A0nl a single run-classifying scalar,
consistent with the other nondim parameters here, instead of an x-dependent
field. Only ds["b"] at that one column is read, so this is fast -- no tiling
needed.
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using Statistics
using CairoMakie
using JLD2
using Interpolations
using Trapz

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";
    dirforce = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirout = string(pth0,"diagout/");
    dirforce = string(pth0,"IW/forcingfiles/");
    dirfig = string(pth0,"figs/");
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

# Flags --------------------------
savefl   = 1  # save data
figflag  = 1  # print the vs-latitude summary figure at the bottom of this file
calcA0mod = false  # compute A0nl from the model time series (opens the sim netCDF,
                   # Butterworth-filters a water column, runs APEKFeq2 -- by far the
                   # slowest part of this script). Set false to skip it entirely once
                   # A0nlana has been checked against A0nl for a run set -- A0nlana
                   # alone is normally enough, and skipping avoids all netCDF I/O.

const T2 = 12+25.2/60
const rho0=1020;
const grav=9.81;

# Gaussian source patch (mirrors IW_flux_LAT_2000km_bash_cuda.jl) -- keep in
# sync if that file's forcing patch center/width ever changes
const gausW_center     = 80_000   # m
const gausW_width      = 16_000   # m
const A0_offset_sigma  = 2        # A0nl extraction point: this many sigma east of the source center
const xA0               = gausW_center + A0_offset_sigma*gausW_width   # 112 km

# run-ID selection: only mainnm + runnms need to be prescribed here; LAT and
# the N2 stratification profile are looked up from run_master.jl, so run-ID
# and latitude can never drift out of sync. runnms need not be a full block --
# any subset of run-IDs already present in RUN_TABLE works.
mainnm  = 11
#runnms  = collect(1:13)   # varying  N2 MERCATOR
runnms  = collect(27:39) # varying  N2 MERCATOR
#runnms  = collect(40:52) # constant N2 MERCATOR 2.5N
#runnms  = collect(53:65) # constant N2 MERCATOR 50N

#test
#mainnm  = 10
#runnms = 1

runs = get_runs(mainnm, runnms)   # errors immediately if a runnm isn't in RUN_TABLE
LATS = [r.lat for r in runs]

# do the analysis in a function
function run_analysis(runnm, LAT, savefl)
# IS = 1; runnm = runnms[IS]; LAT = LATS[IS]

fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
fname_short2 = fnames
filename = string(dirsim,fnames,".nc")
println(fname_short2,"; lat=",LAT," -------------------")

# look up this run's metadata (lat/Flux/DX/N2 source) from the master table
row = get_runs(mainnm, [runnm])[1]

# load N2 profile -----------------------------------------------------------
fnamegrid = n2_filename(row)
path_fname = string(dirforce,fnamegrid);
@load path_fname N2w zfw
N2c = N2w[1:end-1]/2 + N2w[2:end]/2;
zc  = (zfw[1:end-1] .+ zfw[2:end]) ./ 2;   # cell centers -- matches the model's z_aac
                                            # by construction (forcing built on the model grid),
                                            # so this doesn't need to open the sim netCDF

# calculation of the resonance parameter epsilon -------------------------------------------------------
# ω is a function of k (as in Sutherland papers)
# 4om2 - om(2k)2 / 4om2
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

    # using 2k wavelength find the associated frequency
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd);
    kr  = 2*kn[Nm]
    omr = itom(zfw, N2w, fcor, omi, nonhyd, nk, kr, Nm)     # first iteration
    om2 = collect(range(0.75*omr, 1.25*omr, nk))            # second iteration
    omr = itom(zfw, N2w, fcor, om2, nonhyd, nk, kr, Nm)
    return omr
end

# k is a function of ω ---------------
# get k from prescribing omega
function getkres(ω,LAT,nonhyd,Nm)
    fcor   = coriolis(LAT);
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd);
    k_k  = kn[Nm]
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, 2ω, nonhyd);
    k2_k = kn[Nm]
    return k_k, k2_k
end

# obtain the hydrostatic and nonhydrostatic epsilons
ω      = 2π / (T2*3600)
nonhyd = 1; Nm = 1;

# epsilon based on ω -----------------------------
# Sutherland epsilon
omr   = getomres(ω,LAT,nonhyd,Nm)
epsnh = ((2*ω)^2 - omr^2)/(2*ω)^2

nonhyd = 0;
omr   = getomres(ω,LAT,nonhyd,Nm)
epshy = ((2*ω)^2 - omr^2)/(2*ω)^2

# epsilon based on k -----------------------------
# ((2k)2 - k(2om))/(2k)2
# based on omega resonance: om+om=2om
nonhyd=1;
k_k, k2_k = getkres(ω,LAT,nonhyd,Nm)
epsnh_k   = ((2*k_k)^2 - k2_k^2)/(2*k_k)^2

nonhyd=0;
k_k, k2_k = getkres(ω,LAT,nonhyd,Nm)
epshy_k   = ((2*k_k)^2 - k2_k^2)/(2*k_k)^2

# get A0nl analytically ----------------------------
DX = row.DX     # this run's grid spacing
Fx = row.Flux   # this run's mode-1 flux

if DX < 500; nonhyd = 1;
else;        nonhyd = 0;
end

fcor   = coriolis(LAT);

kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
    sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd);

# use the convention that Weig(zmax) = 1
Im = 1; #mode 1
W1 = Weig[:,Im];
imax = argmax(abs.(W1))
W1n  = W1 ./ W1[imax]      # sets W1n[imax] = +1, flips sign automatically
zmax = zfw[imax]           # store for reporting / diagnostics

fig = Figure()
ax1 = Axis(fig[1,1], title=fnamegrid, xlabel="W (s⁻²)", ylabel="z [m]")
lines!(ax1, W1n, zfw, label="N²")
fig

# A₀ = √[2F/(ρ₀c_p²c_g∫Φ'²dz)]  peak displacement, metres
U1 = diff(W1n) ./ diff(zfw)
U2int = trapz(zc,U1.^2)
A0nlana = sqrt(2*Fx / (rho0*Cn[Im]^2*Cgn[Im]*U2int))

println("analytical A0 = ",A0nlana)

# A0nl from the model time series (single x-column near the source) --------
# skips ALL netCDF I/O when calcA0mod=false -- A0nlana above tracks the
# measured A0nl well, and this block (opening the sim netCDF, Butterworth-
# filtering a whole water column, APEKFeq2) is by far the slowest part of
# this script
if calcA0mod
    ds = NCDataset(filename,"r");

    # only select data after the spinup time
    tspin = 10; #days  # for 2000 km domain
    tday0 = ds["time"][:]/24/3600;
    Isel  = findall(>=(tspin),tday0);
    tsec  = ds["time"][Isel];
    tday  = tsec/24/3600;
    dt    = tday[2]-tday[1]

    xc = ds["x_caa"][:];

    # reference density ----------------------------------------------------------
    # compute reference density profile
    # b = sum N2 * dz = sum -g/rho0*drho/dz * dz
    # b = -g/rho0*rho_pert
    # rho_pert = -b*rho0/g
    breff   = cumtrapz(zfw, N2w);                              # bottom up!
    intzc   = interpolate((zfw,), breff, Gridded(Linear()));
    rhorefc = -intzc.(zc) * rho0/grav;                         # rho0 is not added!
    rrr_shape = reshape(rhorefc,1,1,:);

    # time window for averaging (same convention as IW_total_energetics_tile.jl) -
    EXCL = 2; t1 = tday[1]+EXCL*T2/24; t2 = tday[end]-EXCL*T2/24;
    numcycles = floor((t2-t1)/(T2/24))
    t2   = t1+numcycles*(T2/24)
    Iday = findall(item -> item >= t1 && item<= t2, tday)
    Nlen = length(Iday)

    # filter settings ------------------------------------------------------------
    Nf    = 8;
    Tcut1 = 18/24           #D2+HH
    Tcut2 = (T2+T2/2)/2/24  #day; HH M2-M4

    # A0nl: max tidal-band isopycnal displacement at a single x-column just east of
    # the Gaussian source (x=xA0), instead of a domain-wide max --------------
    ixA = argmin(abs.(xc .- xA0))
    println("A0nl column: x=",@sprintf("%.1f",xc[ixA]/1e3)," km (source center=",gausW_center/1e3,
        " km, +",A0_offset_sigma,"σ)")

    bc_col = permutedims(ds["b"][ixA:ixA, :, Isel], (3,1,2));   # (Nt, 1, Nz)
    Nz = length(zc);

    passflg = "high";
    bh_col = similar(bc_col)
    for iz = 1:Nz
        bh_col[:,1,iz] = lowhighpass_butter(bc_col[:,1,iz], Tcut2, dt, Nf, passflg)
    end

    passflg = "low";
    bs_col = similar(bc_col)
    for iz = 1:Nz
        bs_col[:,1,iz] = lowhighpass_butter(bc_col[:,1,iz], Tcut1, dt, Nf, passflg)
    end

    bt_col = (bc_col .- bh_col) .- bs_col   # tidal band = total - highpass - subtidal
    rt_col = -bt_col * rho0/grav
    bc_col = nothing; bh_col = nothing; bs_col = nothing

    thresh = 1e-5;
    APEz_col, Zzt_col = APEKFeq2(rt_col[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
    A0nl = maximum(Zzt_col)/2 + abs(minimum(Zzt_col))/2

    close(ds)
else
    A0nl = NaN
end

# Ostrovsky number ---------------------------------------------------------
# Os = α η₀ / (γ λ²) = 2c α η₀ / (f² λ²)
# α = (3c/2)·∫(φ′)³dz / ∫(φ′)²dz
# β = (c/2) · ∫₋H⁰ φ² dz / ∫₋H⁰ (dφ/dz)² dz
# γ  rotational dispersion: γ = f²/(2c)
# c  linear long-wave phase speed of the mode 
# η₀ the amplitude and λ the horizontal lengthscale of the initial (internal-tide)
U3int = trapz(zc,U1.^3)
alpOS = -1* 3/2 * Cen[Im] * U3int / U2int # -1 for wave of depression
gamma = fcor^2/(2*Cen[Im])
OS = alpOS * A0nlana / (gamma * Ln[Im]^2)
# at LAT=0, fcor=0 -> gamma=0 -> OS=Inf; NaN plots/autoscales far more
# gracefully than Inf (same fix as alpepshy/alpepsnh)
isinf(OS) && (OS = NaN)

# compute non-dimensional parameters from Sutherland 2022 -------------------
# stratification e-folding depth dnl: z1 = depth of peak N2 (top of the
# pycnocline, found automatically -- no assumption about where it sits, since
# that shifts with latitude for the Mercator profiles); z2 = the interpolated
# depth below z1 where the background-subtracted N2 first decays to 1/e of its
# value at z1. dnl = z1 - z2, i.e. the literal e-folding depth. Replaces the
# old fixed z=-100/-300 m secant, which assumed the WOCE AMZ profile shape.
iord = sortperm(zc)              # ascending z: deepest first, shallowest last
zcs  = zc[iord]
N2cs = N2c[iord]

I1s = argmax(N2cs)
z1  = zcs[I1s]

# background N2: mean over the deepest 20% of THIS profile's own z-range,
# rather than a fixed z cutoff -- some profiles (e.g. higher-latitude Mercator
# ones) haven't flattened out by z=-300/-500 m yet, so a fixed cutoff still
# includes decaying-tail points and biases N2_deep high (which makes the 1/e
# crossing land too shallow and dnl come out too small)
deepfrac = 0.2
zdeep_cutoff = zcs[1] + deepfrac*(zcs[end] - zcs[1])
N2_deep = mean(N2cs[zcs .<= zdeep_cutoff])

dN2    = N2cs .- N2_deep         # background-subtracted (excess) stratification
target = dN2[I1s] / ℯ            # yes that is e-symbol, 2.7....

Isub   = 1:I1s                   # deepest point up to the N2 peak
zsub   = zcs[Isub]
dN2sub = dN2[Isub]

Icross = findlast(dN2sub .< target)
if Icross === nothing
    z2 = zsub[1]   # excess N2 never drops below the target -- fall back to the deepest point
    @warn "dnl: 1/e crossing not found above zdeep_cutoff for $(fname_short2); using deepest available point"
else
    z2 = zsub[Icross] + (target - dN2sub[Icross]) / (dN2sub[Icross+1] - dN2sub[Icross]) * (zsub[Icross+1] - zsub[Icross])
end

dnl = z1 - z2

#= old method, kept for reference -- fixed z=-100/-300 m secant, assumed the
WOCE AMZ profile shape (not appropriate for the Mercator profiles, whose
pycnocline depth shifts with latitude)
I1_old  = argmin(abs.(zc .- -100))
I2_old  = argmin(abs.(zc .- -300))
dnl_old = (zc[I1_old] - zc[I2_old]) / log(N2c[I1_old]/N2c[I2_old])
=#

# N2 profile + the fitted exponential decay used for dnl (dashed) ----------
N2exp = N2_deep .+ dN2[I1s] .* exp.(-(z1 .- zsub) ./ dnl)   # model over the fitted range [zsub[1], z1]

figN2 = Figure()
ax1 = Axis(figN2[1,1], title=fnamegrid, xlabel="N² (s⁻²)", ylabel="z [m]")
lines!(ax1, N2c, zc, label="N²")
lines!(ax1, N2exp, zsub, linestyle=:dash, color=:red, label="exp fit (dnl)")
ylims!(ax1, -2000, 0)
axislegend(ax1, position=:rb)
figN2
#display(figN2)

# nonlinearity parameter alpha/epsilon
# at LAT=0 (f=0) the hydrostatic dispersion relation is frequency-independent,
# so epshy can land on/near exact resonance (epshy≈0) and alpepshy -> Inf;
# NaN plots/autoscales far more gracefully than Inf, so replace it here once,
# rather than guarding every downstream plot
alpnl        = A0nlana/dnl
alpepshy     = alpnl/epshy
alpepsnh     = alpnl/epsnh
isinf(alpepshy) && (alpepshy = NaN)
isinf(alpepsnh) && (alpepsnh = NaN)

# same ratio using the k(2om)-based epsilon instead of the omega-based one --
# physically the more appropriate detuning for compound-tide harmonics like
# M4/M6 (frequency is exact by construction from the periodic forcing; the
# wavenumber match is what's actually detuned). Sign-flipped to match the
# omega-based convention, same as the epsilon-vs-latitude panel.
alpepshy_k   = alpnl/(-epshy_k)
alpepsnh_k   = alpnl/(-epsnh_k)
isinf(alpepshy_k) && (alpepshy_k = NaN)
isinf(alpepsnh_k) && (alpepsnh_k = NaN)

# same ratios from the model-measured A0nl instead of the analytic A0nlana
# (NaN throughout when calcA0mod=false, since A0nl itself is NaN then)
alpnl_meas        = A0nl/dnl
alpepshy_meas     = alpnl_meas/epshy
alpepsnh_meas     = alpnl_meas/epsnh
isinf(alpepshy_meas) && (alpepshy_meas = NaN)
isinf(alpepsnh_meas) && (alpepsnh_meas = NaN)

# beat period of energy exchange
Tbeatnh_days = 2π/(epsnh*ω)/(24*3600)
Tbeathy_days = 2π/(epshy*ω)/(24*3600)

println(fnames,"; A0nlana=",@sprintf("%.1f",A0nlana)," m; A0nl=",@sprintf("%.1f",A0nl)," m; alpepshy=",@sprintf("%.3f",alpepshy),
    "; alpepsnh=",@sprintf("%.3f",alpepsnh))

# save nondim terms ----------------------------------------------------------
if savefl == 1
    fnameout = string("nondim_",fname_short2,".jld2")
    jldsave(string(dirout,fnameout);
        LAT, xA0, dnl, A0nlana, A0nl, alpnl, alpepshy, alpepsnh, alpepshy_k, alpepsnh_k,
        alpnl_meas, alpepshy_meas, alpepsnh_meas, Tbeatnh_days, Tbeathy_days,
        epsnh, epshy, epsnh_k, epshy_k, OS);
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


# ============================================================================
# load the just-saved nondim_AMZexptXX.YY.jld2 files back and plot the
# non-dimensional parameters as a function of latitude, 3 rows x 2 columns
# ============================================================================
dnlP          = zeros(length(runnms))
A0nlP         = zeros(length(runnms))
A0nlanaP      = zeros(length(runnms))
alpnlP        = zeros(length(runnms))
alpepshyP     = zeros(length(runnms))
alpepsnhP     = zeros(length(runnms))
alpepshy_kP   = zeros(length(runnms))
alpepsnh_kP   = zeros(length(runnms))
epsnhP        = zeros(length(runnms))
epshyP        = zeros(length(runnms))
epsnh_kP      = zeros(length(runnms))
epshy_kP      = zeros(length(runnms))
TbeatnhP_days = zeros(length(runnms))
TbeathyP_days = zeros(length(runnms))
OSP           = zeros(length(runnms))

for (i, runnm) in enumerate(runnms)
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    @load string(dirout,"nondim_",fnames,".jld2") dnl A0nl A0nlana alpnl alpepshy alpepsnh alpepshy_k alpepsnh_k epsnh epshy epsnh_k epshy_k Tbeatnh_days Tbeathy_days OS
    dnlP[i]          = dnl
    A0nlP[i]         = A0nl
    A0nlanaP[i]      = A0nlana
    alpnlP[i]        = alpnl
    alpepshyP[i]     = alpepshy
    alpepsnhP[i]     = alpepsnh
    alpepshy_kP[i]   = alpepshy_k
    alpepsnh_kP[i]   = alpepsnh_k
    epsnhP[i]        = epsnh
    epshyP[i]        = epshy
    epsnh_kP[i]      = epsnh_k
    epshy_kP[i]      = epshy_k
    TbeatnhP_days[i] = Tbeatnh_days
    TbeathyP_days[i] = Tbeathy_days
    OSP[i]           = OS
end

fnum = string(mainnm,".",runnms[1],"-",runnms[end])

figLAT = Figure(size=(1100,1200))

# (1,1) dnl & A0 combined -- same units [m], so one panel
ax11 = Axis(figLAT[1,1], title=string("dnl & A0 vs latitude (",fnum,")"), xlabel="latitude [°]", ylabel="[m]")
lines!(ax11, LATS, dnlP,     label="dnl",                color=:blue,  linewidth=2)
lines!(ax11, LATS, A0nlP,    label="A0nl (measured)",    color=:black, linewidth=2)
lines!(ax11, LATS, A0nlanaP, label="A0nlana (analytic)", color=:red,   linewidth=2, linestyle=:dash)
axislegend(ax11, position=:rt)

# (1,2) wave-wave interaction beat period
ax12 = Axis(figLAT[1,2], title="wave-wave beat period vs latitude", xlabel="latitude [°]", ylabel="Tbeat [days]")
lines!(ax12, LATS, TbeatnhP_days, label="nonhydrostatic", color=:red, linewidth=3)
lines!(ax12, LATS, TbeathyP_days, label="hydrostatic",    color=:red,   linewidth=2, linestyle=:dash)
axislegend(ax12, position=:rt)

# (2,1) 1/epsilon, ω-based and k-based combined -- the k-based epsilon uses
# the opposite sign convention, so flip it here (plot only, the saved
# epsnh_k/epshy_k keep their raw computed sign) to compare directly.
# 1/epsilon (∝ the PSI/harmonic beat timescale) spans a wide range, hence
# log y-axis; Inf (from epsilon≈0, e.g. epshy at LAT=0) -> NaN so it doesn't
# wreck the autoscale.
nan_guard(x) = (v = copy(x); v[isinf.(v)] .= NaN; v)
inv_epshyP   = nan_guard(1 ./ epshyP)
inv_epsnhP   = nan_guard(1 ./ epsnhP)
inv_epshy_kP = nan_guard(1 ./ (-epshy_kP))
inv_epsnh_kP = nan_guard(1 ./ (-epsnh_kP))

ax21 = Axis(figLAT[2,1], title="1/epsilon vs latitude (ω- and k-based)", xlabel="latitude [°]", ylabel="1/epsilon", yscale=log10)
lines!(ax21, LATS, inv_epshyP,   label="ω, hydrostatic",    color=:red,    linewidth=4, linestyle=:dash)
lines!(ax21, LATS, inv_epsnhP,   label="ω, nonhydrostatic", color=:red,   linewidth=4, linestyle=:solid)
lines!(ax21, LATS, inv_epshy_kP, label="k, hydrostatic",    color=:dodgerblue, linewidth=2, linestyle=:dash)
lines!(ax21, LATS, inv_epsnh_kP, label="k, nonhydrostatic", color=:dodgerblue,    linewidth=2, linestyle=:solid)
axislegend(ax21, position=:rt)

# (2,2) alpha (analytic A0) on its own
ax22 = Axis(figLAT[2,2], title="alpha vs latitude (analytic A0)", xlabel="latitude [°]", ylabel="alpha")
lines!(ax22, LATS, alpnlP, color=:black, linewidth=2)

# (3,1) alpha/epsilon from the analytic A0 (the only alpha/epsilon panel --
# theory tracks the measured A0nl well, so no separate measured-A0 panel);
# omega-based and k(2om)-based epsilon both shown for comparison
ax31 = Axis(figLAT[3,1], title="alpha/epsilon vs latitude (analytic A0)", xlabel="latitude [°]", ylabel="alpha/epsilon")
lines!(ax31, LATS, alpepshyP,   label="ω, hydrostatic",    color=:red,  linewidth=4, linestyle=:dash)
lines!(ax31, LATS, alpepsnhP,   label="ω, nonhydrostatic", color=:red, linewidth=4, linestyle=:solid)
lines!(ax31, LATS, alpepshy_kP, label="k, hydrostatic",    color=:dodgerblue, linewidth=2, linestyle=:dash)
lines!(ax31, LATS, alpepsnh_kP, label="k, nonhydrostatic", color=:dodgerblue,    linewidth=2, linestyle=:solid)
axislegend(ax31, position=:rt)
ylims!(ax31, 0, 75)


# (3,2) Ostrovsky number
ax32 = Axis(figLAT[3,2], title="Ostrovsky number vs latitude", xlabel="latitude [°]", ylabel="Os")
lines!(ax32, LATS, OSP, color=:black, linewidth=2)

figLAT
display(figLAT)
if figflag==1; save(string(dirfig,"nondim_vs_lat_",fnum,".png"), figLAT); end
