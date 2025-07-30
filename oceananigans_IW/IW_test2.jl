# IW_test2.jl
# Maarten Buijsman, DMS, 2025-7-28
# Set up IW test case for Oceananigans
# instead of open boundaries try a body force as in Whitley paper
# also include a sponge layer
# .json setting:    "julia.NumThreads": "auto",

# sims
# 1 IW_fields_an0.0003.nc: velocity grows larger away from the wall
# 2 IW_fields_an0.0003_adv_visc.nc: very little effect
# 3 inclusion of v causes u to become really large
#   maybe I need a different bco for v velocity?

println("number of threads is ",Threads.nthreads())

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie

###########------ SIMULATION PARAMETERS ------#############

# file ID
#fid = "_adv_visc_u_w_lat45"
#fid = "_adv_visc_u_w_sameloc_lat0"
#fid = "_adv_visc_u_w_lat0"
#fid = "_adv_visc_u_w_lat0_bodyforce_sponge_closure"
#fid = "_lat0_bodyforce_sponge_noclosure_ampx4"   #true U = 0.01 amplitude
fid = "_lat0_bodyforce_sponge_noclosure_ampx400"   #true U = 0.1  amplitude

# grid parameters
const lat = 0
const Nz, Nx = 50, 100
const H, L = 1000meters, 500_000meters

println("Δx = ",L/Nx/1e3," km")
println("Δz = ",H/Nz," m")

# internal wave parameters
const gausW_width = 500/3
const gausW_center = 250_000  # x position of Gaussian of forced wave

# sponge regions
#const Sp_Region_right = 500                               # size of sponge region on RHS
#const Sp_Region_left = 500
const Sp_Region_right = 20000                              # size of sponge region on RHS
const Sp_Region_left = 20000
const Sp_extra = 0                                         # not really needed

coriolis = FPlane(latitude = lat);    # Coriolis

grid = RectilinearGrid(size=(Nx, Nz), 
                       x=(0,L), 
                       z=(-H, 0), 
                       topology=(Bounded, Flat, Bounded))

###########-------- Parameters ----------------#############

#=
struct params
    n::Int64
    an::Float64
    f::Float64
    ω::Float64
    N::Float64
end
=#

const n=1
#const an = 0.0003
const an = 0.0003*40
const f = coriolis.f
const N = 0.005
const T = (12+25.2/60)*3600   # period in seconds

ω = 2*π/T
kn = n*π/H*sqrt( (ω^2-f^2)/(N^2-ω^2))

Ln = 2*π/kn
cn = ω/kn

println(" cn = ",cn)
println(" Ln = ",Ln)

#pmi = params(n,an,f,ω,N)  
#pmi = merge(pmi, (;kn = pmi.n*π/H*sqrt( (pmi.ω^2-pmi.f^2)/(pmi.N^2-pmi.ω^2) )))


###########------ FORCING ------#############

# background 
B_func(x, z, t, N) = N^2 * z
B = BackgroundField(B_func, parameters=N)

# IW forcing

# sponge regions
@inline heaviside(X)  = ifelse(X <0, 0.0, 1.0)
@inline mask2nd(X)    = heaviside(X)* X^2
@inline right_mask(x) = mask2nd((x-L+Sp_Region_right+Sp_extra)/(Sp_Region_right+Sp_extra))
@inline left_mask(x)  = mask2nd(((Sp_Region_left+Sp_extra)-x)/(Sp_Region_left+Sp_extra))

#= plot function 
heavisidef(X)  = ifelse(X <0, 0.0, 1.0)
mask2ndf(X)    = heavisidef(X)* X^2
right_maskf(x) = mask2ndf((x-L+Sp_Region_right+Sp_extra)/(Sp_Region_right+Sp_extra))
left_maskf(x)  = mask2ndf(((Sp_Region_left+Sp_extra)-x)/(Sp_Region_left+Sp_extra))

xx = xnodes(grid, Center())
#xx = range(0,L,1000)
asw1 = right_maskf.(xx)
asw2 = left_maskf.(xx)

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax,xx/1e3,asw1,color=:red)
scatter!(ax,xx/1e3,asw2,color=:blue)
fig
=#

@inline u_sponge(x, z, t, u) = - 0.001 * right_mask(x) * u - 0.001 * left_mask(x) * u
@inline v_sponge(x, z, t, v) = - 0.001 * right_mask(x) * v - 0.001 * left_mask(x) * v
@inline w_sponge(x, z, t, w) = - 0.001 * right_mask(x) * w - 0.001 * left_mask(x) * w
@inline b_sponge(x, z, t, b) = - 0.001 * right_mask(x) * b - 0.001 * left_mask(x) * b
#@inline b_sponge(x, z, t, b) =   0.001 * right_mask(x) * (N^2 * z - b) + 0.001 * left_mask(x) * (N^2 * z - b)

#try @inline b_sponge(x, y, z, t, b, p) =   0.001 * right_mask(y, p) * (p.Ñ^2 * z - b)


@inline gaus(x) = exp( -(x - gausW_center)^2 / (2 * gausW_width^2))
@inline Fu_wave(x, z, t) = ω * an * ( n * π / ( kn* H)) * cos( n * π * z /  H ) *
                                 sin( kn * x -  ω * t) * gaus(x)

@inline force_u(x, z, t, u) = u_sponge(x, z, t, u) + Fu_wave(x, z, t)
@inline force_v(x, z, t, v) = v_sponge(x, z, t, v) 
@inline force_w(x, z, t, w) = w_sponge(x, z, t, w) 
@inline force_b(x, z, t, b) = b_sponge(x, z, t, b) 

u_forcing = Forcing(force_u, field_dependencies = :u)
v_forcing = Forcing(force_v, field_dependencies = :v)
w_forcing = Forcing(force_w, field_dependencies = :w)
b_forcing = Forcing(force_b, field_dependencies = :b)

# test function
gausf(x) = exp( -(x - gausW_center)^2 / (2 * gausW_width^2))
#Fu_wavef(x, z, t) = ω * an * ( n * π / ( kn* H)) * cos( n * π * z /  H ) *
#                                 sin( kn * x -  ω * t) * gausf(x)
Fu_wavef2(x, z) = an * ( n * π / ( kn* H)) * cos( n * π * z /  H ) * gausf(x)
println("surface velocity amplitude = ",Fu_wavef2(gausW_center, 0))
println("fractional value = ",gausf(gausW_center-gausW_width))

#=
gausf(gausW_center-500)
zz = range(-H,0,21)                           
uu = Fu_wavef.(gausW_center, 0, range(0,T,21))  
=# 

#@inline force_u(x, y, z, t, v, p) = Fu_wave(x, y, z, t, p) + v_sponge(x, y, z, t, v, p) + vel_sponge_corner(y, z, v)
#@inline force_u(x, y, z, t, u, p) = Fu_wave(x, y, z, t, p) 
#@inline force_u(x, y, z, t p) = Fu_wave(x, y, z, t, p) 

#u_forcing = Forcing(force_u, field_dependencies = :u, parameters=(an=an, n=n, ω=ω, kn=kn, H=H))

model = NonhydrostaticModel(; grid, coriolis,
                advection = WENO(),
#                closure = SmagorinskyLilly(),
#                closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                buoyancy = BuoyancyTracer(),
                timestepper = :RungeKutta3,
                tracers = :b,
                background_fields = (; b=B),
                forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing))

#=
model = NonhydrostaticModel(; grid, coriolis,
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B),
                            forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing))  
=#                           

                            model.forcing.u
                            methods(force_u)




# generate screen output
wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))
    wall_clock[] = time_ns()
    @info msg
    return nothing
end

#Δt = 30seconds
Δt = 1minutes
start_time = 0days
stop_time = 4days
simulation = Simulation(model; Δt, stop_time)

add_callback!(simulation, progress, name=:progress, IterationInterval(400))

fields = Dict("u" => model.velocities.u, "w" => model.velocities.w, "c" => model.tracers.b)

pthnm = "/data3/mbui/ModelOutput/IW/"
fname = "IW_fields_an"
filename=string(pthnm,fname,an,fid,".nc")

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filename, 
    schedule=TimeInterval(30minutes),
    overwrite_existing = true)


#conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minutes)

# integrate the mofo
model.clock.iteration = 0
model.clock.time = 0




run!(simulation)

# read the NC fields
# plot the velocity as a function of time


#= alternative               
u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(+0.1),
                                       bottom = ValueBoundaryCondition(-0.1));

model = NonhydrostaticModel(; grid, coriolis,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B)) 

# mmm also an error
model = NonhydrostaticModel(grid=grid, boundary_conditions=(u=u_bcs,), tracers=:c)

topology = (Periodic, Periodic, Bounded);

grid = RectilinearGrid(size=(16, 16, 16), extent=(1, 1, 1), topology=topology);

u_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(+0.1),
                                       bottom = ValueBoundaryCondition(-0.1));

c_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(20.0),
                                       bottom = GradientBoundaryCondition(0.01));

model = NonhydrostaticModel(grid=grid, boundary_conditions=(u=u_bcs, c=c_bcs), tracers=:c)

=#