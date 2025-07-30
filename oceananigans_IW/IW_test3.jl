#= IW_test3.jl
Maarten Buijsman, USM DMS, 2025-7-29
Set up IW test case for Oceananigans
use open boundary forcing w and u
also include a sponge layer on the right boundary

todo: 1) include parameters in functions, 2) realistic N, 3) add a mode 2
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie

###########------ SIMULATION PARAMETERS ------#############

# file ID
#fid = "_lat0_boundaryforce_sponge_closure"  #true U = 0.1  amplitude
#fid = "_lat0_boundaryforce_closure"  #true U = 0.1  amplitude
#fid = "_lat0_boundaryforce_closure_IW1" 
fid = "_lat0_boundforce_advc4_sponge_8d_2min" 

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
const U0n = 0.1
const an = 0.003
const f = coriolis.f
const N = 0.005
const T = (12+25.2/60)*3600   # period in seconds

ω = 2*π/T
kn = n*π/H*sqrt( (ω^2-f^2)/(N^2-ω^2))

# reverse engineer an
const an = U0n/(n*π/(kn*H))
#U0n = an*(n*π/(kn*H))

ueig(z) = cos(n*π*z/H)
weig(z) = sin(n*π*z/H)

Ln = 2*π/kn
cn = ω/kn

println(" cn = ",cn)
println(" Ln = ",Ln)

println(" velocity u at 0: ",-an*n*π/(kn*H)*ueig(0))
println(" velocity u at -H: ",-an*n*π/(kn*H)*ueig(-H))
println(" velocity w at -H: ",an*weig(-H/2))


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

# nudging layer ∂x/∂t = F(x) + K(x_target - x) 
# K has units [1/time]
@inline u_sponge(x, z, t, u) = - 0.001 * right_mask(x) * u 
@inline v_sponge(x, z, t, v) = - 0.001 * right_mask(x) * v 
@inline w_sponge(x, z, t, w) = - 0.001 * right_mask(x) * w 
@inline b_sponge(x, z, t, b) = - 0.001 * right_mask(x) * b 
#@inline b_sponge(x, z, t, b) =   0.001 * right_mask(x) * (N^2 * z - b) + 0.001 * left_mask(x) * (N^2 * z - b)

#= body force
@inline gaus(x) = exp( -(x - gausW_center)^2 / (2 * gausW_width^2))
@inline Fu_wave(x, z, t) = ω * an * ( n * π / ( kn* H)) * cos( n * π * z /  H ) *
                                 sin( kn * x -  ω * t) * gaus(x)
=#

# additional body forcing can be added
@inline force_u(x, z, t, u) = u_sponge(x, z, t, u) 
@inline force_v(x, z, t, v) = v_sponge(x, z, t, v) 
@inline force_w(x, z, t, w) = w_sponge(x, z, t, w) 
@inline force_b(x, z, t, b) = b_sponge(x, z, t, b) 

u_forcing = Forcing(force_u, field_dependencies = :u)
v_forcing = Forcing(force_v, field_dependencies = :v)
w_forcing = Forcing(force_w, field_dependencies = :w)
b_forcing = Forcing(force_b, field_dependencies = :b)

# boundary forcing
# u at face, w at center offset by -Δx/2
@inline umod(z,t) = -an*n*π/(kn*H)*ueig(z)*sin(-ω*t)

Dx = xspacings(grid, Center())[1]
@inline wmod(z,t) = an*weig(z)*cos(-kn*Dx/2-ω*t)
#@inline vmod(z,t) = f/ω*an*n*π/(kn*H)*ueig(z)*cos(-kn*Dx/2-ω*t)

u_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(umod))
#v_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(vmod))
w_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(wmod))

#= WENO causes diffusion near the boundaries
model = NonhydrostaticModel(; grid, coriolis,
# try order = 9                advection = WENO(order=9),
                advection = Centered(order=4),
                buoyancy = BuoyancyTracer(),
                timestepper = :RungeKutta3,
                tracers = :b,
                background_fields = (; b=B),
                boundary_conditions=(u=u_bcs, w=w_bcs))
#                forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing),
=#

# this model does not cause diffusion near the botom amd surface boundaries
model = NonhydrostaticModel(; grid, coriolis,
                advection = Centered(order=4),
                closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                tracers = :b,
                buoyancy = BuoyancyTracer(),
                background_fields = (; b=B),
                forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing),
                boundary_conditions=(u=u_bcs, w=w_bcs))                 


#model.forcing.u
#methods(force_u)

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

# simulation time stepping
#Δt = 30seconds
Δt = 2minutes
start_time = 0days
stop_time = 8days
simulation = Simulation(model; Δt, stop_time)

add_callback!(simulation, progress, name=:progress, IterationInterval(400))

# write output
fields = Dict("u" => model.velocities.u, "w" => model.velocities.w, "c" => model.tracers.b)
pthnm = "/data3/mbui/ModelOutput/IW/"
fname = "IW_fields_U0n"
filename=string(pthnm,fname,U0n,fid,".nc")

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