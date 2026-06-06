#=
Seamount example
https://clima.github.io/OceananigansDocumentation/stable/literated/internal_tide/
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie

Nx, Nz = 256, 128
H, L = 2kilometers, 1000kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = (-H, 0),
                                  topology = (Periodic, Flat, Bounded))

h₀ = 250meters
width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
bottom(x) = - H + hill(x)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

x = xnodes(grid, Center())
bottom_boundary = interior(grid.immersed_boundary.bottom_height, :, 1, 1)
top_boundary = 0 * x

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1],
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((-grid.Lx/2e3, grid.Lx/2e3), (-grid.Lz, 0)))

band!(ax, x/1e3, bottom_boundary, top_boundary, color = :mediumblue)

fig                                  

coriolis = FPlane(latitude = -45)

T₂ = 12.421hours
ω₂ = 2π / T₂ # radians/sec
ϵ = 0.1 # excursion parameter
U₂ = ϵ * ω₂ * width
A₂ = U₂ * (ω₂^2 - coriolis.f^2) / ω₂

@inline tidal_forcing(x, z, t, p) = p.A₂ * sin(p.ω₂ * t)
u_forcing = Forcing(tidal_forcing, parameters=(; A₂, ω₂))

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                      buoyancy = BuoyancyTracer(),
                                      tracers = :b,
                                      momentum_advection = WENO(),
                                      tracer_advection = WENO(),
                                      forcing = (; u = u_forcing))

Nᵢ² = 1e-4  # [s⁻²] initial buoyancy frequency / stratification
bᵢ(x, z) = Nᵢ² * z
set!(model, u=U₂, b=bᵢ)

#  Run it --------------------------------------------
model.clock.iteration = 0
model.clock.time = 0

Δt = 5minutes
stop_time = 4days
simulation = Simulation(model; Δt, stop_time)

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

add_callback!(simulation, progress, name=:progress, IterationInterval(25))

b = model.tracers.b
u, v, w = model.velocities
U = Field(Average(u))
u′ = u - U
N² = ∂z(b)

#=
save_fields_interval = 30minutes
simulation.output_writers[:fields] = JLD2Writer(model, (; u, u′, w, b, N²); filename,
                                                schedule = TimeInterval(save_fields_interval),
                                                overwrite_existing = true)
=#

filename = "/data3/mbui/ModelOutput/seamount/internal_tide"
#fields = Dict("u" => model.velocities.u, "w" => model.velocities.w, "b" => model.tracers.b)
fields = Dict("u" => model.velocities.u, "w" => model.velocities.w, "b" => model.tracers.b)

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filename, 
    schedule=TimeInterval(30minutes),
    overwrite_existing = true)


run!(simulation)                                                

#= load output
saved_output_filename = filename * ".jld2"

u′_t = FieldTimeSeries(saved_output_filename, "u′")
 w_t = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")

umax = maximum(abs, u′_t[end])
wmax = maximum(abs, w_t[end])

times = u′_t.times
=#
