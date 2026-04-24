#= IW_Amz2_cuda.jl
Maarten Buijsman, USM DMS, 2026-4-24
use realistic N(z) and eigenfunctions generated in testing_sturmL.jl
adopt closed L and R BCOs with sponge layers
IW forcing is with body Gaussian force
based on IW_test3.jl

uses GPU
GPU FIX (2026): All Vector/Matrix fields in `pm` converted to CuArray so that

FunctionField and ContinuousForcing parameters are isbits-compatible on the GPU.
Interpolation (linear_interpolation, cumtrapz) pre-computed on CPU and stored
as CuArray lookup tables  GPU kernels can only index arrays, not call CPU functions.

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

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname, "include_functions.jl"));

#pathout = "/data3/mbui/ModelOutput/IW/"
pathout = "/home/mbui/ModelOutput/IW/"

###########------ OUTPUT FILE NAME ------#############

mainnm = 1
runnm  = 49
fid    = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
println("running ", fid)

numM   = [1];
Usur1, Usur2 = 0.4, 0.0

DX  = 4000;
lat = 35

lats   = [0, 5, 10, 20, 30, 40, 50]
ftot   = [5.88, 5.89, 5.95, 6.19, 6.6, 7.19, 7.81]
fracs2 = [0.151, 0.150, 0.148, 0.141, 0.129, 0.112, 0.091];
lines(lats, fracs2)
lines(lats, ftot)
lines(fracs2, ftot)

max_Δt     = 10minutes
Δt         = 1minutes
start_time = 0days
stop_time  = 15days

println("stop_time: ", stop_time, "; lat: ", lat, "; select mode: ", numM)

###########------ LOAD N and grid params ------#############

dirin      = string(pathout, "forcingfiles/");
fnamegrid  = "N2_amz1.jld2";
path_fname = string(dirin, fnamegrid);

gridfile = jldopen(path_fname, "r")
println(keys(gridfile))
close(gridfile)

@load path_fname N2w zfw

fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, N2w, zfw)
ylims!(ax1, -500, 0)
fig

###########------ SIMULATION PARAMETERS ------#############

Nz = length(zfw) - 1;
L  = 700_000;
Nx = Integer(L / DX);
H  = abs(round(minimum(zfw)));

TM2 = (12 + 25.2 / 60) * 3600   # M2 tidal period
dx  = L / Nx

const fnudl          = 0.002
const fnudr          = 0.0001
const Sp_Region_right = 200_000
const Sp_Region_left  = 40_000
const Sp_extra        = 0
const gausW_width     = 16_000
const gausW_center    = 40_000

pm = (lat=lat, Nz=Nz, Nx=Nx, H=H, L=L, numM=numM, gausW_center=gausW_center,
      gausW_width=gausW_width);

pm = merge(pm, (Usur=[Usur1, Usur2], T=TM2));

rhos     = 1034
KEs1     = 15.0
KEs2     = 10.0
Usur1tst = sqrt(4 * KEs1 / rhos);
Usur2tst = sqrt(4 * KEs2 / rhos);
@printf("Based on KE, U1 = %.2f, U2 = %.2f\n", Usur1tst, Usur2tst)
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

fnameEIG = @sprintf("EIG_amz_%04.1f.jld2", lat)
f_save   = copy(fcor.f);
om2      = copy(ω);
jldsave(string(dirin, fnameEIG); f=f_save, om2, zfw, N2w, nonhyd, kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2);
println(string(fnameEIG), " Ueig data saved ........ ")

Lstr = @sprintf("%5.1f", Ln[1] / 1e3)
println("Mode 1 wavelength is ", Lstr, " km")
println("fraction gauss_width/L1 is ", @sprintf("%5.3f", gausW_width / Ln[1]))

zcw = zfw[1:end-1] / 2 + zfw[2:end] / 2;

fig = Figure()
ax1 = Axis(fig[1, 1])
lines!(ax1, Weig[:, 1], zfw)
lines!(ax1, Weig[:, 2], zfw)
ax2 = Axis(fig[1, 2])
lines!(ax2, Ueig[:, 1], zcw)
lines!(ax2, Ueig[:, 2], zcw)
fig

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

# --- u eigenfunctions interpolated onto cell centres (length Nz) -------------
Ueig_scaled_cpu = zeros(Nz, nModes)
for i in 1:nModes
    col     = Ueig[:, i] * pm.Usur[i] / Ueig[end, i]   # scale to surface vel
    itp     = linear_interpolation(zcw, col, extrapolation_bc=Line())
    Ueig_scaled_cpu[:, i] = itp.(zcw)
end

# --- w eigenfunctions interpolated onto cell faces (length Nz+1) -------------
Weig_scaled_cpu = zeros(Nz + 1, nModes)
for i in 1:nModes
    col     = Weig[:, i] * pm.Usur[i] / Ueig[end, i]
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
    lat          = Int32(lat),
    Nz           = Int32(Nz),
    Nx           = Int32(Nx),
    H            = Float64(H),
    L            = Float64(L),
    numM         = numM_cu,           # CuArray{Int32,1}  ✓
    gausW_center = Float64(gausW_center),
    gausW_width  = Float64(gausW_width),
    Usur1        = Float64(Usur1),    # scalars instead of Vector
    Usur2        = Float64(Usur2),
    T            = Float64(TM2),
    f            = Float64(fcor.f),
    ω            = Float64(ω),
    zcw          = zcw_cu,            # CuArray{Float64,1} ✓
    zfw          = zfw_cu,            # CuArray{Float64,1} ✓
    kn           = kn_cu,             # CuArray{Float64,1} ✓
    Ueig         = Ueig_cu,           # CuArray{Float64,2} ✓
    Weig         = Weig_cu,           # CuArray{Float64,2} ✓
    bb           = bb_cu,             # CuArray{Float64,1} ✓  (background b)
    N2w          = CuArray(N2w),      # CuArray{Float64,1} ✓
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

# --- u body forcing (du/dt from wave) ----------------------------------------
@inline function Fu_wave(x, z, t, p)
    du  = 0.0
    phi = (0.0, Float64(π))
    for i in 1:length(p.numM)
        mi    = p.numM[i]                              # mode index (1-based)
        Ueigi = gpu_interp1(p.zcw, view(p.Ueig, :, mi), z)
        du   += Ueigi * p.ω * cos(p.kn[mi] * (x - p.gausW_center) - p.ω * t - phi[i])
    end
    return du * framp(t, p) * gaus(x, p)
end

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

@inline force_u(x, z, t, u, p) = u_sponge(x, z, t, u, p) + Fu_wave(x, z, t, p)
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
    advection         = Centered(order=4),
    closure           = ScalarDiffusivity(ν=1e-2, κ=1e-2),
    tracers           = :b,
    buoyancy          = BuoyancyTracer(),
    background_fields = (; b=B),
    forcing           = (u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing))

###########------ SIMULATION ------#############

simulation = Simulation(model; Δt, stop_time)

progress(sim) = @printf(
    "Iter: % 6d, sim time: % s , wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
    iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
    sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

fields = Dict("u"   => model.velocities.u,
              "v"   => model.velocities.v,
              "w"   => model.velocities.w,
              "b"   => model.tracers.b,
              "pNHS" => model.pressures.pNHS,
              "pHY"  => model.pressures.pHY′)

filenameout = string(pathout, fid, ".nc")
simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filenameout,
                 schedule=TimeInterval(15minutes),
                 overwrite_existing=true)

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=max_Δt)

model.clock.iteration = 0
model.clock.time      = 0

run!(simulation)