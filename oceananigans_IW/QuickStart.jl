# https://clima.github.io/OceananigansDocumentation/stable/quick_start/

using Oceananigans
using CairoMakie

grid = RectilinearGrid(size = (128, 128),
                       x = (0, 2π),
                       y = (0, 2π),
                       topology = (Periodic, Periodic, Flat))

model = NonhydrostaticModel(; grid, advection=WENO())

ϵ(x, y) = 2rand() - 1
set!(model, u=ϵ, v=ϵ)

simulation = Simulation(model; Δt=0.01, stop_iteration=100)
run!(simulation)

# visualization =====================

u, v, w = model.velocities
ζ = Field(∂x(v) - ∂y(u))

heatmap(ζ, axis=(; aspect=1))

# a few more iterations =====================

simulation.stop_iteration += 400
run!(simulation)

heatmap(ζ, axis=(; aspect=1))