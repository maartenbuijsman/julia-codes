"""
    coriolis(lat)

Returns f for latitude (degrees)

# Arguments
- `lat`: Latitude (degrees)

# Returns
- `f`: Coriolis frequency (radians/s)
"""
function coriolis(lat)
    # BigOmega = (2*pi)/86164.09  # 2pi / seconds in sidereal day 
    BigOmega = 7.292115e-5  # Earth rotation rate (rad/s) Kantha & Clayson p.842
    f = 2 * BigOmega .* sin.(lat * pi / 180)
    return f
end