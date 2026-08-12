#= IW_flux_LAT_2000km_bash_cuda.jl
Maarten Buijsman, USM DMS, 2026-8-10  (nudging generator, generated with Claude Code)
    for mode 1 M2, dx=200m, 2000km, 20days
    IW generation by RELAXATION (nudging) of u toward the analytic mode over the
    Gaussian source patch. Forcing amplitude set from an energy metric per mode:
    forcing_metric = "flux" (constant F vs latitude, default) | "KE" | "usurf".
    Amplitude:  a = sqrt(2F/(rho0*Cg*∫Ueig2²dz))  [flux]  -> nudge u -> a*Ueig2*cos(kx-wt)
    v,w,b develop dynamically (v via Coriolis). N2source: amz1 | zonalmean | zonalmeanfixed.

use realistic N(z) and eigenfunctions generated in testing_sturmL.jl
adopt closed L and R BCOs with sponge layers
based on IW_test3.jl / IW_KE_200m_2000km_bash_cuda.jl

uses GPU
GPU FIX (2026): All Vector/Matrix fields in `pm` converted to CuArray so that

FunctionField and ContinuousForcing parameters are isbits-compatible on the GPU.
Interpolation (linear_interpolation, cumtrapz) pre-computed on CPU and stored
as CuArray lookup tables  GPU kernels can only index arrays, not call CPU functions.

Run modes (set `runmode` below):
    "batch"  : parameters come from 6 command-line ARGS, as supplied by
               run_batch.sh from a params_XX.jl file.
    "manual" : ARGS are ignored; parameters are hardcoded directly in the
               "MANUAL MODE PARAMETERS" block below. Use this for
               interactive debugging (REPL / IDE "run file") where there
               are no command-line ARGS to parse.

Command-line usage (batch mode):
    julia IW_Amz_bash_cuda.jl <mainnm> <runnm> <lat> <numM> <Usur1> <Usur2>

Example:
    julia IW_Amz_bash_cuda.jl 4 3 0.0 1 0.4 0.0

N2 forcing source (set `N2source` below):
    "amz1"           : original WOCE-based N2_amz1.jld2 (single fixed profile).
    "zonalmean"      : Mercator-based N2_ZonalMeanAtl_lat<lat>.jld2, matched to
                       this run's `lat` -- only exists at lat = 0, 2.5, 5,
                       10, 15, ..., 60.
    "zonalmeanfixed" : Mercator-based profile at fixed `latfix`, same for all runs.
=#

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2
using Printf
using CUDA

println("number of threads is ", Threads.nthreads())

# -------------------------------------------------------
# RUN MODE: "batch" (parse command-line ARGS, via run_batch.sh) or
# "manual" (edit the parameters below directly, for debugging)
# -------------------------------------------------------
runmode = "batch"      # "batch" or "manual" -- NOTE: run_batch.sh always passes
#runmode = "manual"       # 6 ARGS regardless of this flag, so leave this on
                        # "batch" for real batch runs; only set to "manual"
                        # when running this file directly (REPL/IDE) to debug

# N2 forcing source: "amz1" (original WOCE-based, single fixed profile) or
# "zonalmean" (Mercator-based, matched to this run's `lat`) -- NOTE: zonalmean
# files only exist at lat = 0,2.5,5,10,15,...,60; several existing
# input_params/*.jl use other latitudes (e.g. 7.5, 12.5, 28.8), so this
# defaults to "amz1" to preserve existing batch behavior everywhere
# zonalmeanfixed = constant N2 everywhere

#N2source = "amz1"        
N2source = "zonalmean"
#N2source = "zonalmeanfixed"
#latfix = 2.5;
latfix = 50.0;

# forcing metric: how the two per-mode targets are interpreted
#   "flux"  : energy flux F [W/m]  -> constant-flux-vs-latitude (a=sqrt(2F/(rho0*Cg*∫Ueig2²)))
#   "KE"    : depth-integrated KE [J/m2]
#   "usurf" : surface velocity [m/s]
forcing_metric = "flux"

if runmode == "batch"

    if length(ARGS) != 6
        error("Usage: julia <script> <mainnm> <runnm> <lat> <numM> <target1> <target2>\n" *
              "  mainnm  : experiment number (integer)\n" *
              "  runnm   : run number       (integer)\n" *
              "  lat     : latitude         (float)\n" *
              "  numM    : mode selection, e.g. 1 or 1,2\n" *
              "  target1 : mode-1 forcing target (units per forcing_metric)\n" *
              "  target2 : mode-2 forcing target")
    end

    mainnm = parse(Int,     ARGS[1])
    runnm  = parse(Int,     ARGS[2])
    lat    = parse(Float64, ARGS[3])
    numM   = parse.(Int,    split(ARGS[4], ","))   # e.g. "1" → [1]; "1,2" → [1,2]
    target1 = parse(Float64, ARGS[5])   # mode-1 target (units per forcing_metric)
    target2 = parse(Float64, ARGS[6])   # mode-2 target

elseif runmode == "manual"

    # ---- MANUAL MODE PARAMETERS: edit these directly for debugging ----
    mainnm    = 9           # 200 m production series
    runnm     = 99          # scratch slot
    lat       = 40.0        # debug latitude
    numM      = [1]         # e.g. [1] or [1,2]
    target1   = 10_000.0    # mode-1 target (forcing_metric="flux" -> 10 kW/m)
    target2   = 0.0         # mode-2 target
    # ---------------------------------------------------------------

else
    error("runmode must be \"batch\" or \"manual\", got: ", runmode)
end

println("runmode = $runmode, mainnm = $mainnm, runnm = $runnm, lat = $lat, numM = $numM, metric = $forcing_metric, target1 = $target1, target2 = $target2")

# -------------------------------------------------------

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname, "include_functions.jl"));

#pathout = "/data3/mbui/ModelOutput/IW/"
pathout = "/home/mbui/ModelOutput/IW/"

###########------ OUTPUT FILE NAME ------#############

fid = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
println("running ", fid)

###########------ INPUT PARAMETERS ------#############

TM2 = (12 + 25.2 / 60) * 3600   # M2 tidal period

# mode 1 and 2 surface velocities
# numM = [1];
# Usur1, Usur2 = 0.4, 0.0  

#high resolution: 100/200 m
L   = 2_000_000;                   # domain length
DX  = 200;                         # 11. series 200 m production grid (Nx = 10000)
#DX  = 100;                        # 
#DX  = 4000;                       # 10. series
max_Δt = 2minutes  
Δt     = 15seconds   # nonhyd

println("domain length = ",L/1e3," km")


# run duration and output frequency
#dtoutput   = 15minutes 
start_time = 0days
mid_time   = 10days      # old output regime
stop_time  = 20days      # old output regime

#println("stop_time: ", stop_time, "; lat: ", lat, "; select mode: ", numM)

t_coarse = range(start_time,  mid_time, step=60minutes)  # 0 → day 10, every 60 min
t_fine   = range(mid_time,   stop_time, step=5minutes)   # day 10 → 20, every 5 min

output_times = vcat(collect(t_coarse), collect(t_fine)[2:end])  # avoid duplicate at day 4

###########------ LOAD N and grid params ------#############

dirin = string(pathout, "forcingfiles/");

if N2source == "amz1"
    fnamegrid = "N2_amz1.jld2"
elseif N2source == "zonalmean"
    fnamegrid = @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", lat)
elseif N2source == "zonalmeanfixed"
    fnamegrid = @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", latfix)
else
    error("N2source must be \"amz1\" or \"zonalmean\" or \"zonalmeanfixed\", got: ", N2source)
end

path_fname = string(dirin, fnamegrid);

isfile(path_fname) || error("N2 forcing file not found: ", path_fname,
      N2source in ("zonalmean","zonalmeanfixed") ? "\n(zonalmean files only exist at lat = 0, 2.5, 5, 10, 15, ..., 60)" : "")

println("N2source = $N2source, fnamegrid = $fnamegrid --------------------------------")

gridfile = jldopen(path_fname, "r")
println(keys(gridfile))
close(gridfile)

@load path_fname N2w zfw

#= plot N2
fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, N2w, zfw)
ylims!(ax1, -500, 0)
fig
=#

###########------ SIMULATION PARAMETERS ------#############
Nz = length(zfw) - 1;
Nx = Integer(L / DX);
H  = abs(round(minimum(zfw)));
dx = L / Nx
const rho0 = 1020.0     # reference density for KE-target inversion [kg/m3]

# constants for nudging/relaxation layers
const fnudl           = 0.002
const fnudr           = 0.0001
const Sp_Region_right = 200_000
const Sp_Region_left  =  40_000
const Sp_extra        = 0
const gausW_width     = 16_000
const gausW_center    = 80_000     # moved out of the left sponge (0-40 km) for nudging

pm = (lat=lat, Nz=Nz, Nx=Nx, H=H, L=L, numM=numM, gausW_center=gausW_center,
      gausW_width=gausW_width);

pm = merge(pm, (T=TM2,));

#= play with KE strength
rhos     = 1034
KEs1     = 15.0
KEs2     = 10.0
Usur1tst = sqrt(4 * KEs1 / rhos);
Usur2tst = sqrt(4 * KEs2 / rhos);
@printf("Based on KE, U1 = %.2f, U2 = %.2f\n", Usur1tst, Usur2tst)
=#

println("Δx = ", pm.L / pm.Nx / 1e3, " km")
println("Δz = ", pm.H / pm.Nz, " m on average")

grid = RectilinearGrid(GPU(), size=(pm.Nx, pm.Nz),
                       x=(0, pm.L),
                       z=zfw,
                       topology=(Bounded, Flat, Bounded))

###########-------- Parameters ----------------#############

ω    = 2 * π / pm.T
fcor = FPlane(latitude=pm.lat);
pm   = merge(pm, (f=fcor.f, ω=ω))

nonhyd = 1;
kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
    sturm_liouville_noneqDZ_norm(zfw, N2w, pm.f, pm.ω, nonhyd);

fnameEIG = @sprintf("EIG_AMZexpt%02i.%02i_LAT_%04.1f.jld2", mainnm, runnm, lat)
f_save   = copy(fcor.f);
om2      = copy(ω);
jldsave(string(dirin, fnameEIG); f=f_save, om2, zfw, N2w, nonhyd, kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2, lat);
println(string(fnameEIG), " Ueig data saved ........ ")

Lstr = @sprintf("%5.1f", Ln[1] / 1e3)
println("Mode 1 wavelength is ", Lstr, " km")
println("fraction gauss_width/L1 is ", @sprintf("%5.3f", gausW_width / Ln[1]))

zcw = zfw[1:end-1] / 2 + zfw[2:end] / 2;

#= plot Ueig
fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, Weig[:, 1], zfw)
lines!(ax1, Weig[:, 2], zfw)
ax2 = Axis(fig[1, 2])
lines!(ax2, Ueig[:, 1], zcw)
lines!(ax2, Ueig[:, 2], zcw)
fig
=#

###########------ PRE-COMPUTE INTERPOLATED PROFILES ON CPU ------#############
#=
  GPU kernels cannot call linear_interpolation(), cumtrapz(), or any other
  CPU/heap-allocating function. We pre-compute all interpolated profiles here
  on the CPU, then upload them to the GPU as CuArrays. The kernel functions
  below use only indexing (no interpolation at runtime).

  Profiles stored:
    Ueig_cu[iz, imode]  : u-eigenfunction at cell centres, scaled to Usur
    Weig_cu[iz, imode]  : w-eigenfunction at cell faces,   scaled to Usur
    bb_cu[iz]           : background buoyancy at cell faces (cumtrapz of N2w)
=#

using Interpolations

nModes = 2          # we keep modes 1 and 2 (indices into Ueig/Weig columns)

# --- forcing target -> modal amplitude a_m -----------------------------------
# Mode m, linear, depth-integrated, time-mean (Ueig2: ⟨Φ²⟩=1, ∫Ueig2²dz≈H):
#   KE = 1/4*rho0*(1+f²/w²)*a²*∫Ueig2²dz ;  E=KE+APE=KE*2/(1+f²/w²) ;  F=E*Cg
#   "flux" : a = sqrt( 2*F  / (rho0*Cg*∫Ueig2²dz) )   (1+f²/w² cancels → const-flux)
#   "KE"   : a = sqrt( 4*KE / (rho0*(1+f²/w²)*∫Ueig2²dz) )
#   "usurf": a = usurf / Ueig2(0)
target  = (target1, target2)
dzf     = diff(zfw)
polar   = 1 + (pm.f / pm.ω)^2
a_modal = zeros(nModes)
for i in 1:nModes
    Iphi = sum(Ueig2[:, i].^2 .* dzf)          # ∫Ueig2_i^2 dz (≈ H)
    Cgi  = Cgn[i]
    if     forcing_metric == "flux"  ; a_modal[i] = target[i] <= 0 ? 0.0 : sqrt(2*target[i]/(rho0*Cgi*Iphi))
    elseif forcing_metric == "KE"    ; a_modal[i] = sqrt(4*target[i]/(rho0*polar*Iphi))
    elseif forcing_metric == "usurf" ; a_modal[i] = target[i]/Ueig2[end, i]
    else   error("forcing_metric must be flux | KE | usurf, got: ", forcing_metric)
    end
    KEi  = 0.25*rho0*polar*a_modal[i]^2*Iphi                       # implied diagnostics
    APEi = KEi*(1 - (pm.f/pm.ω)^2)/(1 + (pm.f/pm.ω)^2)
    Fi   = (KEi + APEi)*Cgi
    @printf("mode %d [%s target=%.4g]: a=%.5f, usurf=%.4f m/s | KE=%.1f APE=%.1f E=%.1f J/m2, F=%.1f W/m (Cg=%.3f, ∫Ueig2²=%.0f)\n",
            i, forcing_metric, target[i], a_modal[i], a_modal[i]*Ueig2[end,i],
            KEi, APEi, KEi+APEi, Fi, Cgi, Iphi)
end

# --- nudging (relaxation) parameters -----------------------------------------
nudge_tau  = 300.0                 # relaxation timescale [s] inside the source patch
nudge_rate = 1.0 / nudge_tau       # [1/s]
@printf("nudging: tau=%.0f s (rate=%.2e 1/s); est. amplitude deficit tau*Cg/width=%.3f\n",
        nudge_tau, nudge_rate, nudge_tau * Cgn[1] / gausW_width)

# --- u target eigenfunction (modal-amplitude-scaled) onto cell centres -------
Ueig_scaled_cpu = zeros(Nz, nModes)
for i in 1:nModes
    col     = a_modal[i] * Ueig2[:, i]           # nudging target amplitude a_m * Ueig2(z)
    itp     = linear_interpolation(zcw, col, extrapolation_bc=Line())
    Ueig_scaled_cpu[:, i] = itp.(zcw)
end

# --- w eigenfunctions (NOT nudged; kept only for optional diagnostics) -------
Weig_scaled_cpu = zeros(Nz + 1, nModes)
for i in 1:nModes
    col     = a_modal[i] * Weig[:, i]
    itp     = linear_interpolation(zfw, col, extrapolation_bc=Line())
    Weig_scaled_cpu[:, i] = itp.(zfw)
end

# --- background buoyancy at cell faces (cumulative trapezoid of N2w) ---------
bb_cpu = cumtrapz(zfw, N2w)   # length Nz+1, on zfw grid

# --- upload everything to the GPU -------------------------------------------
Ueig_cu = CuArray(Ueig_scaled_cpu)   # (Nz,   nModes)
Weig_cu = CuArray(Weig_scaled_cpu)   # (Nz+1, nModes)
bb_cu   = CuArray(bb_cpu)            # (Nz+1,)
zcw_cu  = CuArray(zcw)               # (Nz,)   cell-centre z values
zfw_cu  = CuArray(zfw)               # (Nz+1,) cell-face  z values
kn_cu   = CuArray(kn[1:nModes])      # (nModes,)
numM_cu = CuArray(Int32.(numM))      # GPU-friendly integer array

# --- pack all scalar + CuArray params into a single NamedTuple --------------
#     Every field must be isbits (scalars) or a CuArray (GPU pointer = isbits).
pm_gpu = (
    lat          = Float64(lat),        # Float64 to preserve decimal latitudes
    Nz           = Int32(Nz),
    Nx           = Int32(Nx),
    H            = Float64(H),
    L            = Float64(L),
    numM         = numM_cu,             # CuArray{Int32,1}  ✓
    gausW_center = Float64(gausW_center),
    gausW_width  = Float64(gausW_width),
    nudge_rate   = Float64(nudge_rate), # u-nudging relaxation rate [1/s]
    T            = Float64(TM2),
    f            = Float64(fcor.f),
    ω            = Float64(ω),
    zcw          = zcw_cu,              # CuArray{Float64,1} ✓
    zfw          = zfw_cu,              # CuArray{Float64,1} ✓
    kn           = kn_cu,              # CuArray{Float64,1} ✓
    Ueig         = Ueig_cu,            # CuArray{Float64,2} ✓
    Weig         = Weig_cu,            # CuArray{Float64,2} ✓
    bb           = bb_cu,              # CuArray{Float64,1} ✓  (background b)
    N2w          = CuArray(N2w),       # CuArray{Float64,1} ✓
)

###########------ GPU-COMPATIBLE KERNEL FUNCTIONS ------#############
#=
  Rules for functions called inside FunctionField / ContinuousForcing on GPU:
    * No heap allocation (no [], no interpolation objects, no push!)
    * No CPU function calls (linear_interpolation, cumtrapz, etc.)
    * Only indexing into CuDeviceArrays and scalar arithmetic
    * Use @inline so the compiler can inline them into the kernel

  Strategy for z-lookup (replaces linear_interpolation):
    We find the bracketing index in the pre-computed z-grid and linearly
    interpolate between the two neighbouring stored values.
=#

# Linear interpolation helper: given sorted array zarr and value z0,
# return interpolated value from varr.  Safe to call inside GPU kernels.
@inline function gpu_interp1(zarr, varr, z0)
    n  = length(zarr)
    # clamp to bounds
    z0 = max(zarr[1], min(zarr[n], z0))
    # binary-search-free scan (fine for small Nz ~ O(100))
    idx = 1
    for k in 1:n-1
        if zarr[k] <= z0
            idx = k
        end
    end
    # linear interpolation
    dz  = zarr[idx+1] - zarr[idx]
    t   = dz == 0 ? 0.0 : (z0 - zarr[idx]) / dz
    return varr[idx] * (1 - t) + varr[idx+1] * t
end

# ramp-up and Gaussian envelope (scalars only  fine on GPU)
@inline framp(t, p)  = 1 - exp(-1 / (p.T / 2) * t)
@inline gaus(x, p)   = exp(-(x - p.gausW_center)^2 / (2 * p.gausW_width^2))

# --- u target (analytic mode, modal-amplitude scaled) + nudging --------------
# p.Ueig now holds a_m * Ueig2_m(z) (the KE-derived target), so u_target is the
# wave velocity we relax toward, ramped in time and localized by the Gaussian.
@inline function u_target(x, z, t, p)
    ut  = 0.0
    phi = (0.0, Float64(π))
    for i in 1:length(p.numM)
        mi = p.numM[i]                                 # mode index (1-based)
        Ai = gpu_interp1(p.zcw, view(p.Ueig, :, mi), z)  # = a_mi * Ueig2_mi(z)
        ut += Ai * cos(p.kn[mi] * (x - p.gausW_center) - p.ω * t - phi[i])
    end
    return ut * framp(t, p)
end

# relaxation of u toward the target inside the Gaussian source patch
@inline u_nudge(x, z, t, u, p) = -p.nudge_rate * (u - u_target(x, z, t, p)) * gaus(x, p)

# v target = tidal polarization v = (f/ω) u, 90° out of phase (v ∝ sin)
# nudging v to this damps the spurious subtidal/near-inertial v that pools at
# the source when only u is nudged.
@inline function v_target(x, z, t, p)
    vt  = 0.0
    phi = (0.0, Float64(π))
    for i in 1:length(p.numM)
        mi = p.numM[i]
        Ai = gpu_interp1(p.zcw, view(p.Ueig, :, mi), z)   # a_mi * Ueig2_mi(z)
        vt += (p.f / p.ω) * Ai * sin(p.kn[mi] * (x - p.gausW_center) - p.ω * t - phi[i])
    end
    return vt * framp(t, p)
end

@inline v_nudge(x, z, t, v, p) = -p.nudge_rate * (v - v_target(x, z, t, p)) * gaus(x, p)

# --- v body forcing -----------------------------------------------------------
@inline function Fv_wave(x, z, t, p)
    dv  = 0.0
    phi = (0.0, Float64(π))
    for i in 1:length(p.numM)
        mi    = p.numM[i]
        Ueigi = gpu_interp1(p.zcw, view(p.Ueig, :, mi), z)
        dv   += p.f * Ueigi * sin(p.kn[mi] * (x - p.gausW_center) - p.ω * t - phi[i])
    end
    return dv * framp(t, p) * gaus(x, p)
end

# --- w body forcing -----------------------------------------------------------
@inline function Fw_wave(x, z, t, p)
    dw  = 0.0
    phi = (0.0, Float64(π))
    for i in 1:length(p.numM)
        mi    = p.numM[i]
        Weigi = gpu_interp1(p.zfw, view(p.Weig, :, mi), z)
        dw   += Weigi * p.ω * sin(p.kn[mi] * (x - p.gausW_center) - p.ω * t - phi[i])
    end
    return dw * framp(t, p) * gaus(x, p)
end

# --- background buoyancy (GPU-safe: index into pre-computed bb_cu) -----------
@inline function B_func(x, z, t, p)
    return gpu_interp1(p.zfw, p.bb, z)
end

###########------ SPONGE LAYERS ------#############

@inline heaviside(X)      = ifelse(X < 0, 0.0, 1.0)
@inline mask2nd(X)        = heaviside(X) * X^2
@inline right_mask(x, p)  = mask2nd((x - p.L + Sp_Region_right + Sp_extra) / (Sp_Region_right + Sp_extra))
@inline left_mask(x, p)   = mask2nd(((Sp_Region_left + Sp_extra) - x) / (Sp_Region_left + Sp_extra))

@inline u_sponge(x, z, t, u, p) = -(fnudl * left_mask(x, p) + fnudr * right_mask(x, p)) * u
@inline v_sponge(x, z, t, v, p) = -(fnudl * left_mask(x, p) + fnudr * right_mask(x, p)) * v
@inline w_sponge(x, z, t, w, p) = -(fnudl * left_mask(x, p) + fnudr * right_mask(x, p)) * w
@inline b_sponge(x, z, t, b, p) = -(fnudl * left_mask(x, p) + fnudr * right_mask(x, p)) * b

@inline force_u(x, z, t, u, p) = u_sponge(x, z, t, u, p) + u_nudge(x, z, t, u, p)
# v-nudging tested (run 93) but REJECTED: it relocates the source artifact to the
# patch edge and worsens near-source APE. u-only is cleaner; v develops via Coriolis.
@inline force_v(x, z, t, v, p) = v_sponge(x, z, t, v, p)
@inline force_w(x, z, t, w, p) = w_sponge(x, z, t, w, p)
@inline force_b(x, z, t, b, p) = b_sponge(x, z, t, b, p)

###########------ BUILD OCEANANIGANS MODEL ------#############

B         = BackgroundField(B_func, parameters=pm_gpu)

u_forcing = Forcing(force_u, field_dependencies=:u, parameters=pm_gpu)
v_forcing = Forcing(force_v, field_dependencies=:v, parameters=pm_gpu)
w_forcing = Forcing(force_w, field_dependencies=:w, parameters=pm_gpu)
b_forcing = Forcing(force_b, field_dependencies=:b, parameters=pm_gpu)

model = NonhydrostaticModel(grid;
    coriolis          = fcor,
#    advection         = Centered(order=4),
#    closure           = ScalarDiffusivity(ν=1e-2),    # blow up
#    closure           = ScalarDiffusivity(ν=1e-5),     # blow up   
#    closure           = ScalarDiffusivity(ν=1e-2, κ=1e-2),
    advection         = WENO(),
    closure           = ScalarDiffusivity(ν=1e-5),
#    closure           = ScalarDiffusivity(ν=1e-2),
    tracers           = :b,
    buoyancy          = BuoyancyTracer(),
    background_fields = (; b=B),
    forcing           = (u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing))

println(model)

###########------ SIMULATION ------#############

simulation = Simulation(model; Δt, stop_time)

progress(sim) = @printf(
    "Iter: % 6d, sim time: % s , wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
    iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
    sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

fields = Dict("u"    => model.velocities.u,
              "v"    => model.velocities.v,
              "w"    => model.velocities.w,
              "b"    => model.tracers.b,
              "pNHS" => model.pressures.pNHS,
              "pHY"  => model.pressures.pHY′) #′ is not an error!

filenameout = string(pathout, fid, ".nc")
simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filenameout,
                 schedule = SpecifiedTimes(output_times),
                 overwrite_existing=true)

conjure_time_step_wizard!(simulation, cfl=0.8, max_Δt=max_Δt)

model.clock.iteration = 0
model.clock.time      = 0

run!(simulation)
