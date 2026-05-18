#= IW_Amz2_cuda.jl => IW_Amz_bash_cuda.jl
Maarten Buijsman, USM DMS, 2026-4-24
use realistic N(z) and eigenfunctions generated in testing_sturmL.jl
adopt closed L and R BCOs with sponge layers
IW forcing is with body Gaussian force
based on IW_test3.jl

uses GPU

Command-line usage:
    julia IW_Amz2_cuda.jl <mainnm> <runnm> <lat>

Example:
    julia IW_Amz2_cuda.jl 1 49 35.0

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
# Parse command-line arguments
# -------------------------------------------------------
if length(ARGS) != 3
    error("Usage: julia IW_Amz2_cuda.jl <mainnm> <runnm> <lat>\n" *
          "  mainnm : experiment number (integer)\n" *
          "  runnm  : run number       (integer)\n" *
          "  lat    : latitude         (float)")
end

mainnm = parse(Int,     ARGS[1])
runnm  = parse(Int,     ARGS[2])
lat    = parse(Float64, ARGS[3])

println("mainnm = $mainnm, runnm = $runnm, lat = $lat")

# -------------------------------------------------------
pathname = "/home/mbui/Documents/julia-codes/functions/"
include(string(pathname, "include_functions.jl"))

pathout = "/home/mbui/ModelOutput/IW/"

###########------ OUTPUT FILE NAME ------#############

fid = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
println("running ", fid)

numM = [1]
Usur1, Usur2 = 0.4, 0.0

# dx grid size
DX = 4000

lats  = [0, 5, 10, 20, 30, 40, 50]
ftot  = [5.88, 5.89, 5.95, 6.19, 6.6, 7.19, 7.81]
fracs2 = [0.151, 0.150, 0.148, 0.141, 0.129, 0.112, 0.091]

# simulation time stepping
max_Δt     = 10minutes
Δt         = 1minutes
start_time = 0days
stop_time  = 15days

println("stop_time: ", stop_time, "; lat: ", lat, "; select mode: ", numM)

###########------ LOAD N and grid params ------#############

dirin      = string(pathout, "forcingfiles/")
fnamegrid  = "N2_amz1.jld2"
path_fname = string(dirin, fnamegrid)

gridfile = jldopen(path_fname, "r")
println(keys(gridfile))
close(gridfile)

@load path_fname N2w zfw

###########------ SIMULATION PARAMETERS ------#############

Nz = length(zfw) - 1
L  = 700_000
Nx = Integer(L / DX)
H  = abs(round(minimum(zfw)))

TM2 = (12 + 25.2/60) * 3600   # M2 tidal period
dx  = L / Nx

const fnudl           = 0.002
const fnudr           = 0.0001
const Sp_Region_right = 200_000
const Sp_Region_left  =  40_000
const Sp_extra        = 0
const gausW_width     = 16_000
const gausW_center    = 40_000

pm = (lat=lat, Nz=Nz, Nx=Nx, H=H, L=L, numM=numM,
      gausW_center=gausW_center, gausW_width=gausW_width)
pm = merge(pm, (Usur=[Usur1, Usur2], T=TM2))

rhos     = 1034
KEs1     = 15.0
KEs2     = 10.0
Usur1tst = sqrt(4*KEs1/rhos)
Usur2tst = sqrt(4*KEs2/rhos)
@printf("Based on KE, U1 = %.2f, U2 = %.2f\n", Usur1tst, Usur2tst)
println("Δx = ", pm.L/pm.Nx/1e3, " km")
println("Δz = ", pm.H/pm.Nz, " m on average")

grid = RectilinearGrid(GPU(), size=(pm.Nx, pm.Nz),
                       x=(0, pm.L),
                       z=zfw,
                       topology=(Bounded, Flat, Bounded))

###########-------- Parameters ----------------#############

ω    = 2*π/pm.T
fcor = FPlane(latitude=pm.lat)
pm   = merge(pm, (f=fcor.f, ω=ω))

nonhyd = 1
kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
    sturm_liouville_noneqDZ_norm(zfw, N2w, pm.f, pm.ω, nonhyd)

fnameEIG = @sprintf("EIG_amz_%04.1f.jld2", lat)
f   = copy(fcor.f)
om2 = copy(ω)
jldsave(string(dirin, fnameEIG); f, om2, zfw, N2w, nonhyd, kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2)
println(string(fnameEIG), " Ueig data saved ........ ")

Lstr = @sprintf("%5.1f", Ln[1]/1e3)
println("Mode 1 wavelength is ", Lstr, " km")
println("fraction gauss_width/L1 is ", @sprintf("%5.3f", gausW_width/Ln[1]))

zcw = zfw[1:end-1]/2 + zfw[2:end]/2
pm  = merge(pm, (zcw=zcw, zfw=zfw, kn=kn[1:2], Ueig=Ueig[:,1:2], Weig=Weig[:,1:2], N2w=N2w))

###########------ FORCING ------#############

using Interpolations

function fdun(x, z, t, p)
    du = 0.0
    for i in p.numM
        phi  = [0, π]
        Ueig = p.Ueig[:,i] * p.Usur[i] / p.Ueig[end,i]
        intzc = linear_interpolation(p.zcw, Ueig, extrapolation_bc=Line())
        Ueigi = intzc.(z)
        du   = du .+ Ueigi * p.ω * cos(p.kn[i]*(x-p.gausW_center) - p.ω*t - phi[i])
    end
    return du
end

function fdwn(x, z, t, p)
    dw = 0.0
    for i in p.numM
        phi  = [0, π]
        Weig = p.Weig[:,i] * p.Usur[i] / p.Ueig[end,i]
        intzc = linear_interpolation(p.zfw, Weig, extrapolation_bc=Line())
        Weigi = intzc.(z)
        dw   = dw .+ Weigi * p.ω * sin(p.kn[i]*(x-p.gausW_center) - p.ω*t - phi[i])
    end
    return dw
end

function fdvn(x, z, t, p)
    dv = 0.0
    for i in p.numM
        phi  = [0, π]
        Ueig = p.Ueig[:,i] * p.Usur[i] / p.Ueig[end,i]
        intzc = linear_interpolation(p.zcw, Ueig, extrapolation_bc=Line())
        Ueigi = intzc.(z)
        dv   = dv .+ p.f .* Ueigi * sin(p.kn[i]*(x-p.gausW_center) - p.ω*t - phi[i])
    end
    return dv
end

function B_func(x, z, t, p)
    bb    = cumtrapz(p.zfw, p.N2w)
    intzc = linear_interpolation(p.zfw, bb, extrapolation_bc=Line())
    bbi   = intzc.(z)
    return bbi
end

B = BackgroundField(B_func, parameters=pm)

@inline heaviside(X)       = ifelse(X < 0, 0.0, 1.0)
@inline mask2nd(X)         = heaviside(X) * X^2
@inline right_mask(x, p)   = mask2nd((x - p.L + Sp_Region_right + Sp_extra) / (Sp_Region_right + Sp_extra))
@inline left_mask(x, p)    = mask2nd(((Sp_Region_left + Sp_extra) - x) / (Sp_Region_left + Sp_extra))

@inline u_sponge(x, z, t, u, p) = -(fnudl * left_mask(x,p) + fnudr * right_mask(x,p)) * u
@inline v_sponge(x, z, t, v, p) = -(fnudl * left_mask(x,p) + fnudr * right_mask(x,p)) * v
@inline w_sponge(x, z, t, w, p) = -(fnudl * left_mask(x,p) + fnudr * right_mask(x,p)) * w
@inline b_sponge(x, z, t, b, p) = -(fnudl * left_mask(x,p) + fnudr * right_mask(x,p)) * b

@inline framp(t, p)              = 1 - exp(-1/(p.T/2) * t)
@inline gaus(x, p)               = exp(-(x - p.gausW_center)^2 / (2 * p.gausW_width^2))

@inline Fu_wave(x, z, t, p) = fdun(x, z, t, p) * framp(t, p) * gaus(x, p)
@inline Fv_wave(x, z, t, p) = fdvn(x, z, t, p) * framp(t, p) * gaus(x, p)
@inline Fw_wave(x, z, t, p) = fdwn(x, z, t, p) * framp(t, p) * gaus(x, p)

@inline force_u(x, z, t, u, p) = u_sponge(x, z, t, u, p) + Fu_wave(x, z, t, p)
@inline force_v(x, z, t, v, p) = v_sponge(x, z, t, v, p)
@inline force_w(x, z, t, w, p) = w_sponge(x, z, t, w, p)
@inline force_b(x, z, t, b, p) = b_sponge(x, z, t, b, p)

u_forcing = Forcing(force_u, field_dependencies=:u, parameters=pm)
v_forcing = Forcing(force_v, field_dependencies=:v, parameters=pm)
w_forcing = Forcing(force_w, field_dependencies=:w, parameters=pm)
b_forcing = Forcing(force_b, field_dependencies=:b, parameters=pm)

model = NonhydrostaticModel(grid; coriolis=fcor,
                            advection=Centered(order=4),
                            closure=ScalarDiffusivity(ν=1e-2, κ=1e-2),
                            tracers=:b,
                            buoyancy=BuoyancyTracer(),
                            background_fields=(; b=B),
                            forcing=(u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing))

simulation = Simulation(model; Δt, stop_time)

progress(sim) = @printf(
    "Iter: % 6d, sim time: % s, wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
    iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
    sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))

fields = Dict("u"    => model.velocities.u,
              "v"    => model.velocities.v,
              "w"    => model.velocities.w,
              "b"    => model.tracers.b,
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
