#= IW_nondim_params.jl
Maarten Buijsman, USM DMS, 2026-8-28
Non-dimensional parameters and time scales for the D2 internal-tide PSI/resonance
problem (Sutherland & Dhaliwal 2022): resonance parameter epsilon (frequency- and
wavenumber-based, hydrostatic and nonhydrostatic), the nonlinearity length scale
dnl, the nonlinearity parameter alpha/epsilon, and the PSI beat period.
Split out of IW_total_energetics_tile.jl (was lines 126-183 and 385-397 there).

A0nl (max tidal-band vertical isopycnal displacement), the MEASURED companion
to the analytic A0nlana computed below, is no longer computed in this file --
that required opening the sim netCDF, Butterworth-filtering a whole water
column, and running APEKFeq2, by far the slowest part of this script (1000+ s
for a 13-run block on the 200 m grid). Since A0nl doesn't change once a run
has been simulated, that cost doesn't need to be paid on every run of this
file: claudecodes/IW_A0nl_extract.jl does it once per run and saves
a0nl_AMZexptXX.YY.jld2, which is loaded back in below (NaN + a warning if
that file doesn't exist yet for a given run).
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using Printf
using Statistics
using CairoMakie
using JLD2
using Interpolations
using Trapz

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";
    dirforce = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
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
mainnm  = 10
runnms  = collect(1:13)   # varying  N2   12.5 kW/m 
#runnms  = collect(27:39) # varying  N2   25 kW/m
#runnms  = collect(40:52) # constant N2   2.5N
#runnms  = collect(53:65) # constant N2  50N
#runnms  = collect(66:78) # varying  N2  50 kW/m

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
# at LAT=0 (f=0) the hydrostatic dispersion relation is exactly linear
# (non-dispersive), so epshy is analytically exactly 0 there -- not just
# "close to 0". getomres only reaches that value through two nested linear
# interpolation passes though, so omr and 2ω are computed via different
# arithmetic paths and don't bit-for-bit cancel: they can differ by 1 ULP,
# leaving a ~1e-16-scale residual whose SIGN is essentially arbitrary. When
# it lands negative, 1/epshy becomes a huge finite negative number (not Inf),
# which breaks any log-scale plot of 1/epshy and isn't caught by an isinf()
# guard. Fix it at the source instead of masking the symptom downstream.
if LAT == 0.0; epshy = 0.0; end

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

# A0nl (measured) -- loaded from claudecodes/IW_A0nl_extract.jl's output
# rather than recomputed here (see file header for why). Missing file just
# gives NaN + a warning, same fallback convention as the beatdist_*.jld2 load
# further down.
fnameA0nl = string(dirout,"a0nl_",fname_short2,".jld2")
if isfile(fnameA0nl)
    A0nl = load(fnameA0nl, "A0nl")
else
    A0nl = NaN
    @warn "no a0nl_$(fname_short2).jld2 found -- run claudecodes/IW_A0nl_extract.jl for this run first"
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

# old method, kept live for comparison against the new 1/e-crossing dnl --
# I1_old now sits at the depth of peak N2 (same starting point as z1 above,
# instead of the original fixed z=-100 m), I2_old stays fixed at z=-300 m
# (the original WOCE-AMZ-shaped assumption); dnl_old is the old log-ratio
# secant between those two points.
zdeep = -300
I1_old  = argmax(N2c)
I2_old  = argmin(abs.(zc .- zdeep))
dnl_old = (zc[I1_old] - zc[I2_old]) / log(N2c[I1_old]/N2c[I2_old])

# N2 profile + the fitted exponential decay used for dnl (dashed) ----------
N2exp = N2_deep .+ dN2[I1s] .* exp.(-(z1 .- zsub) ./ dnl)   # model over the fitted range [zsub[1], z1]

# old method's exponential fit, for comparison -- no background subtraction
# (the old secant just assumes N2(z) = N2(z1_old)*exp((z-z1_old)/dnl_old)),
# shown only over [z2_old, z1_old], the interval it was actually derived from
z_oldrange = range(zc[I2_old], zc[I1_old], length=50)
N2exp_old  = N2c[I1_old] .* exp.((z_oldrange .- zc[I1_old]) ./ dnl_old)

figN2 = Figure()
ax1 = Axis(figN2[1,1], title=fnamegrid, xlabel="N² (s⁻²)", ylabel="z [m]")
lines!(ax1, N2c, zc, label="N²")
lines!(ax1, N2exp, zsub, linestyle=:dash, color=:red, label="exp fit (dnl)")
lines!(ax1, N2exp_old, z_oldrange, linestyle=:dashdot, color=:green, label="exp fit (dnl_old)")
ylims!(ax1, -2000, 0)
axislegend(ax1, position=:rb)
figN2
#display(figN2)
#return

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
# (NaN throughout if a0nl_*.jld2 hasn't been extracted yet for this run)
alpnl_meas        = A0nl/dnl
alpepshy_meas     = alpnl_meas/epshy
alpepsnh_meas     = alpnl_meas/epsnh
isinf(alpepshy_meas) && (alpepshy_meas = NaN)
isinf(alpepsnh_meas) && (alpepsnh_meas = NaN)

# same ratio using dnl_old instead of dnl -- omega-based, nonhydrostatic only,
# to compare the old vs. new dnl method against each other directly
alpnl_old    = A0nlana/dnl_old
alpepsnh_old = alpnl_old/epsnh
isinf(alpepsnh_old) && (alpepsnh_old = NaN)

# beat period of energy exchange
Tbeatnh_days = 2π/(epsnh*ω)/(24*3600)
Tbeathy_days = 2π/(epshy*ω)/(24*3600)

# mode-1 group speed for this run's N2 profile (at the M2 frequency ω, same
# Cgn already computed above for A0nlana) -- used downstream to turn Tbeat
# into a beat DISTANCE
Cg1 = Cgn[Im]

println(fnames,"; A0nlana=",@sprintf("%.1f",A0nlana)," m; A0nl=",@sprintf("%.1f",A0nl)," m; alpepshy=",@sprintf("%.3f",alpepshy),
    "; alpepsnh=",@sprintf("%.3f",alpepsnh))

# save nondim terms ----------------------------------------------------------
if savefl == 1
    fnameout = string("nondim_",fname_short2,".jld2")
    jldsave(string(dirout,fnameout);
        LAT, xA0, dnl, A0nlana, A0nl, alpnl, alpepshy, alpepsnh, alpepshy_k, alpepsnh_k,
        alpnl_meas, alpepshy_meas, alpepsnh_meas, Tbeatnh_days, Tbeathy_days,
        epsnh, epshy, epsnh_k, epshy_k, OS, dnl_old, alpnl_old, alpepsnh_old, Cg1);
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
dnl_oldP      = zeros(length(runnms))
alpepsnh_oldP = zeros(length(runnms))
Cg1P          = zeros(length(runnms))   # mode-1 group speed [m/s]

for (i, runnm) in enumerate(runnms)
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    @load string(dirout,"nondim_",fnames,".jld2") dnl A0nl A0nlana alpnl alpepshy alpepsnh alpepshy_k alpepsnh_k epsnh epshy epsnh_k epshy_k Tbeatnh_days Tbeathy_days OS dnl_old alpepsnh_old Cg1
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
    dnl_oldP[i]      = dnl_old
    alpepsnh_oldP[i] = alpepsnh_old
    Cg1P[i]          = Cg1
end

fnum = string(mainnm,".",runnms[1],"-",runnms[end])

figLAT = Figure(size=(1100,1200))

# catches Inf/NaN AND ordinary non-positive values -- moved up here (was
# defined just above the (2,1) panel) so the (1,2) panel below can reuse it
# for the k-based beat period too
nan_guard(x) = (v = copy(x); v[.!isfinite.(v) .| (v .<= 0)] .= NaN; v)
ω = 2π / (T2*3600)   # local to run_analysis there, needed globally here too

# (1,1) dnl & A0 combined -- same units [m], so one panel; dnl_old (fixed
# z=-300 m secant, but I1_old now at max N2 like the new method) shown for
# comparison
ax11 = Axis(figLAT[1,1], title=string("dnl & A0 vs latitude (",fnum,")"), xlabel="latitude [°]", ylabel="[m]")
lines!(ax11, LATS, dnlP,     label="dnl",                color=:blue,  linewidth=2)
lines!(ax11, LATS, dnl_oldP, label="dnl_old",            color=:blue,  linewidth=2, linestyle=:dot)
lines!(ax11, LATS, A0nlP,    label="A0nl (measured)",    color=:black, linewidth=2)
lines!(ax11, LATS, A0nlanaP, label="A0nlana (analytic)", color=:red,   linewidth=2, linestyle=:dash)
axislegend(ax11, position=:rt)

# (1,2) wave-wave interaction beat period -- ω-based (as saved by run_analysis)
# plus k-based, computed here from epsnh_kP/epshy_kP with the same sign flip
# and Tbeat formula (2π/(eps*ω)) used for the ω-based ones
Tbeatnh_kP_days = nan_guard(2π ./ ((-epsnh_kP) .* ω) ./ (24*3600))
Tbeathy_kP_days = nan_guard(2π ./ ((-epshy_kP) .* ω) ./ (24*3600))

ax12 = Axis(figLAT[1,2], title="wave-wave beat period vs latitude", xlabel="latitude [°]", ylabel="Tbeat [days]")
lines!(ax12, LATS, TbeatnhP_days,   label="ω, nonhydrostatic", color=:red,        linewidth=3)
#lines!(ax12, LATS, TbeathyP_days,   label="ω, hydrostatic",    color=:red,        linewidth=2, linestyle=:dash)
lines!(ax12, LATS, Tbeatnh_kP_days, label="k, nonhydrostatic", color=:dodgerblue, linewidth=3)
#lines!(ax12, LATS, Tbeathy_kP_days, label="k, hydrostatic",    color=:dodgerblue, linewidth=2, linestyle=:dash)
axislegend(ax12, position=:rt)

# (2,1) 1/epsilon, ω-based and k-based combined -- the k-based epsilon uses
# the opposite sign convention, so flip it here (plot only, the saved
# epsnh_k/epshy_k keep their raw computed sign) to compare directly.
# 1/epsilon (∝ the PSI/harmonic beat timescale) spans a wide range, hence
# log y-axis; Inf (from epsilon≈0, e.g. epshy at LAT=0) -> NaN so it doesn't
# wreck the autoscale.
# catches Inf/NaN AND ordinary non-positive values -- epsilon can cross zero
# (e.g. runnms=53:65, where the N2 profile is fixed at the lat=50 shape but
# evaluated across all simulated latitudes), so 1/epsilon can land on a large
# *finite* negative number, not just Inf, which log10(y-scale) can't handle
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

#= standalone figure: epsilon itself (not inverted) for the same 4 variables
# as the 1/epsilon panel above -- exploratory only, not saved to disk
epshyP_g   = nan_guard(epshyP)
epsnhP_g   = nan_guard(epsnhP)
epshy_kP_g = nan_guard(-epshy_kP)
epsnh_kP_g = nan_guard(-epsnh_kP)

fig = Figure(size=(600,450))
axeps = Axis(fig[1,1], title="epsilon vs latitude (ω- and k-based)", xlabel="latitude [°]", ylabel="epsilon")
lines!(axeps, LATS, epshyP_g,   label="ω, hydrostatic",    color=:red,    linewidth=4, linestyle=:dash)
lines!(axeps, LATS, epsnhP_g,   label="ω, nonhydrostatic", color=:red,   linewidth=4, linestyle=:solid)
lines!(axeps, LATS, epshy_kP_g, label="k, hydrostatic",    color=:dodgerblue, linewidth=2, linestyle=:dash)
lines!(axeps, LATS, epsnh_kP_g, label="k, nonhydrostatic", color=:dodgerblue,    linewidth=2, linestyle=:solid)
hlines!(axeps, [0], color=:gray, linestyle=:dot)
axislegend(axeps, position=:rt)
display(fig)
=#

# (2,2) alpha (analytic A0) on its own
ax22 = Axis(figLAT[2,2], title="alpha vs latitude (analytic A0)", xlabel="latitude [°]", ylabel="alpha")
lines!(ax22, LATS, alpnlP, color=:black, linewidth=2)

# (3,1) alpha/epsilon from the analytic A0 (the only alpha/epsilon panel --
# theory tracks the measured A0nl well, so no separate measured-A0 panel);
# omega-based and k(2om)-based epsilon both shown for comparison, plus the
# omega/nonhydrostatic case using dnl_old instead of dnl (old vs. new dnl)
ax31 = Axis(figLAT[3,1], title="alpha/epsilon vs latitude (analytic A0)", xlabel="latitude [°]", ylabel="alpha/epsilon")
lines!(ax31, LATS, alpepshyP,      label="ω, hydrostatic",           color=:red,  linewidth=4, linestyle=:dash)
lines!(ax31, LATS, alpepsnhP,      label="ω, nonhydrostatic",        color=:red, linewidth=4, linestyle=:solid)
lines!(ax31, LATS, alpepshy_kP,    label="k, hydrostatic",           color=:dodgerblue, linewidth=2, linestyle=:dash)
lines!(ax31, LATS, alpepsnh_kP,    label="k, nonhydrostatic",        color=:dodgerblue,    linewidth=2, linestyle=:solid)
lines!(ax31, LATS, alpepsnh_oldP,  label="ω, nonhydrostatic (old dnl)", color=:green, linewidth=2, linestyle=:dot)
axislegend(ax31, position=:rt)
ylims!(ax31, 0, 75)


#= (3,2) Ostrovsky number
ax32 = Axis(figLAT[3,2], title="Ostrovsky number vs latitude", xlabel="latitude [°]", ylabel="Os")
lines!(ax32, LATS, OSP, color=:black, linewidth=2)
=#

# (3,2) beat distance = Tbeat * mode-1 group speed Cg1P (same Cg1P for all
# four, since group speed is a property of the N2 profile, not of which
# epsilon convention set the beat period)
BeatDistnh_km   = TbeatnhP_days   .* 86400 .* Cg1P ./ 1e3
BeatDisthy_km   = TbeathyP_days   .* 86400 .* Cg1P ./ 1e3
BeatDistnh_kkm  = Tbeatnh_kP_days .* 86400 .* Cg1P ./ 1e3
BeatDisthy_kkm  = Tbeathy_kP_days .* 86400 .* Cg1P ./ 1e3

# measured beat distance from the simulated KEt(x) profiles, for comparison
# with the analytical Tbeat*Cg1 above. Produced by
# claudecodes/IW_KEt_beat_distance.jl (2*median spacing between successive
# extrema of the 150 km-lowpassed KEt over x in [100,1800] km -- see that
# file's header for the method and why each step is needed). Missing files
# just leave NaNs, so this panel still draws if that script hasn't been run
# for the current run block yet.
BeatDistMeas_km = fill(NaN, length(runnms))
for (i, runnm) in enumerate(runnms)
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    fbeat  = string(dirout,"beatdist_",fnames,".jld2")
    if isfile(fbeat)
        BeatDistMeas_km[i] = load(fbeat, "BeatDist_km")
    end
end
nmeas = count(!isnan, BeatDistMeas_km)
nmeas == 0 && @warn "no beatdist_*.jld2 found for mainnm=$mainnm runnms=$(runnms[1]):$(runnms[end]) -- run claudecodes/IW_KEt_beat_distance.jl for this block first"

ax32 = Axis(figLAT[3,2], title="beat distance vs latitude", xlabel="latitude [°]", ylabel="beat distance [km]")
lines!(ax32, LATS, BeatDistnh_km,  label="ω, nonhydrostatic", color=:red,        linewidth=3)
#lines!(ax32, LATS, BeatDisthy_km,  label="ω, hydrostatic",    color=:red,        linewidth=2, linestyle=:dash)
lines!(ax32, LATS, BeatDistnh_kkm, label="k, nonhydrostatic", color=:dodgerblue, linewidth=3)
#lines!(ax32, LATS, BeatDisthy_kkm, label="k, hydrostatic",    color=:dodgerblue, linewidth=2, linestyle=:dash)
lines!(ax32, LATS, BeatDistMeas_km, label="measured (KEt)", color=:black, linewidth=2)
scatter!(ax32, LATS, BeatDistMeas_km, color=:black, markersize=8)
axislegend(ax32, position=:rt)

figLAT
display(figLAT)
if figflag==1; save(string(dirfig,"nondim_vs_lat_",fnum,".png"), figLAT); end
