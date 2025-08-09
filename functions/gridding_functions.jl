# gridding_functions.jl
# Maarten Buijsman, USM, 2025-8-8
# This file contains various gridding functions
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