using Plots # Required for visualization

"""
    stretched_grid(H, n, alpha)

Generates a stretched grid from -H to 0 using a hyperbolic tangent function.

# Arguments
- `H`: The total depth of the grid.
- `n`: The number of grid points.
- `alpha`: The stretching parameter. A larger value increases clustering near -H and 0.

# Returns
- A vector of stretched grid points.
"""
function stretched_grid(H, n, alpha)
    # 1. Create a uniform grid from -1 to 1
    xi = LinRange(-1, 1, n)

    # 2. Apply the hyperbolic tangent stretching
    #   The mapping tanh(alpha*xi) maps [-1, 1] to [-tanh(alpha), tanh(alpha)]
    #   The division normalizes the domain to [-1, 1]
    #   The -H * ... shifts and scales the domain to [-H, 0]
    z = -H .* tanh.(alpha .* xi) ./ tanh(alpha)

    return z
end

# --- Example usage ---
H = 100.0 # Total depth
n = 50    # Number of grid points
alpha = 3.0 # Stretching parameter

# Generate the stretched grid
z_stretched = stretched_grid(H, n, alpha)

# Verify the endpoints
println("First grid point: $(z_stretched[1])") # Should be approximately -H
println("Last grid point: $(z_stretched[end])") # Should be approximately 0

# Plot the result to visualize the stretching
uniform_grid = LinRange(-H, 0, n)
plot(uniform_grid, label="Uniform Grid", line=:dot, marker=:x,
    title="Comparison of Stretched and Uniform Grids",
    xlabel="Grid Point Index", ylabel="Depth (z)")
plot!(z_stretched, label="Stretched Grid", marker=:circle, markersize=3,
    line=:solid, color=:red)
