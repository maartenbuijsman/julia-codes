# gridding_functions.jl
# Maarten Buijsman, USM, 2025-8-8
# This file contains various gridding functions:
# meshgrid, cumtrapz
"""
    meshgrid(x,y)

This method creates the coordinate matrices X and Y based on vectors x and y.

# Arguments
x, y coordinate vectors

# Returns
- X, Y coordinate matrices

# Info
Maarten Buijsman, USM, 2025-8-8
Based on MATLAB's meshgrid
"""
function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

"""
    cumtrapz(X::T, Y::T) where {T <: AbstractVector}

Cumulative intergration

# Arguments
Coordinate vector X
Vector Y that needs to be integrated 

# Returns
Cumulative integrated value vector with same length as X 

# Info
Maarten Buijsman, USM, 2025-9-3
https://stackoverflow.com/questions/58139195/cumulative-integration-options-with-julia
"""
function cumtrapz(X::T, Y::T) where {T <: AbstractVector}
  # Check matching vector length
  @assert length(X) == length(Y)
  # Initialize Output
  out = similar(X)
  out[1] = 0
  # Iterate over arrays
  for i in 2:length(X)
    out[i] = out[i-1] + 0.5*(X[i] - X[i-1])*(Y[i] + Y[i-1])
  end
  # Return output
  return out
end
