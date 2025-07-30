# IW_test1.jl
# Maarten Buijsman, DMS, 2025-7-21
# Set up IW test case for Oceananigans

# sims
# 1 IW_fields_an0.0003.nc: velocity grows larger away from the wall
# 2 IW_fields_an0.0003_adv_visc.nc: very little effect
# 3 inclusion of v causes u to become really large
#   maybe I need a different bco for v velocity?

Threads.nthreads()

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf

#fid = "_adv_visc_u_w_lat45"
#fid = "_adv_visc_u_w_sameloc_lat0"
#fid = "_adv_visc_u_w_lat0"
fid = "_adv_visc_u_w_lat0_nosponge"

lat = 0

Nz, Nx = 50, 100
H, L = 1000meters, 500_000meters


grid = RectilinearGrid(size=(Nx, Nz), 
                       x=(0,L), 
                       z=(-H, 0), 
                       topology=(Bounded, Flat, Bounded))


println("Δx = ",L/Nx/1e3," km")
println("Δz = ",H/Nz," m")

# buoyancy ====================================
N = 0.005       # buoyancy frequency [s⁻¹]
#N = 0.001       # buoyancy frequency [s⁻¹]
B_func(x, z, t, N) = N^2 * z
B = BackgroundField(B_func, parameters=N)

# and now the forcing =======================================

# add some boundary conditions
n=1 
an = 0.003
#an = 0.0003
T = (12+25/60)*3600   #seconds
ω = 2*π/T  #rad/s

coriolis = FPlane(latitude = lat);
f = coriolis.f

kn = n*π/H*sqrt((ω^2-f^2)/(N^2-ω^2) )

Ln = 2*π/kn
cn = ω/kn

ueig(z) = cos(n*π*z/H)
weig(z) = sin(n*π*z/H)

println(" velocity u at 0: ",-an*n*π/(kn*H)*ueig(0))
println(" velocity u at -H: ",-an*n*π/(kn*H)*ueig(-H))
println(" velocity w at -H: ",an*weig(-H/2))


# boundary forcing
# u at face, w at center offset by -Δx/2
@inline umod(z,t) = -an*n*π/(kn*H)*ueig(z)*sin(-ω*t)

Dx = xspacings(grid, Center())[1]
#Dx = 0.0 # w is assumed at same location as x; no improvement here
@inline wmod(z,t) = an*weig(z)*cos(-kn*Dx/2-ω*t)

#@inline vmod(z,t) = f/ω*an*n*π/(kn*H)*ueig(z)*cos(-kn*Dx/2-ω*t)

# HOW TO create heat map of buoyancy field ====================== 

#=
   
z = znodes(grid, Center())
#Δx = xspacings(grid, Center())


bb = B_func(0, z, 0, N)

# make 2D array
bbb = repeat(bb,1,Nx)

heatmap(x/1e3,z,bbb')

# apply to vectors
uu = ueig.(z)

Figure
lines(uu,z)
=#

u_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(umod))
#v_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(vmod))
w_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(wmod))


model = NonhydrostaticModel(; grid, coriolis,
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B),
                            boundary_conditions=(u=u_bcs, w=w_bcs)) 

#                            boundary_conditions=(; w=w_bcs)) 

#                            boundary_conditions=(u=u_bcs,v=v_bcs,w=w_bcs)) 

#=
model = NonhydrostaticModel(; grid, coriolis,
                            advection = Centered(order=4),
                            closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B),
                            boundary_conditions=(; u=u_bcs)) 
=#

                            #=
model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                            tracers = :b,
                            buoyancy = BuoyancyTracer(),
                            background_fields = (; b=B),
                            boundary_conditions=(; u=u_bcs)) 
                            =#

model.velocities.u.boundary_conditions
#model.velocities.v.boundary_conditions
model.velocities.w.boundary_conditions


# integrate the mofo
#Δt = 30seconds
Δt = 1minutes
start_time = 0days
stop_time = 4days
simulation = Simulation(model; Δt, stop_time)

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
add_callback!(simulation, progress, name=:progress, IterationInterval(400))


fields = Dict("u" => model.velocities.u, "w" => model.velocities.w, "c" => model.tracers.b)

pthnm = "/data3/mbui/ModelOutput/IW/"
fname = "IW_fields_an"
filename=string(pthnm,fname,an,fid,".nc")

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filename, 
    schedule=TimeInterval(30minutes),
    overwrite_existing = true)


conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minutes)

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