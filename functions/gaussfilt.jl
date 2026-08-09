#= gaussfilt.jl
Maarten Buijsman, USM DMS, 2026-08-07  (generated with Claude Code)
1-D Gaussian smoothing for vectors, with edge-aware renormalisation and a
physical-length interface. A better replacement for a fixed N-point boxcar
running mean: smooth spectral response, cutoff set by a real length scale.
=#

"""
    gaussfilt(y, σ; truncate=4.0)

Smooth vector `y` with a Gaussian kernel of standard deviation `σ` measured in
**samples** (grid points). The kernel is truncated at `±truncate·σ`.

At the array ends the (truncated) kernel is **renormalised over the points that
actually exist**, so the output is unbiased at the boundaries — no energy loss,
no pull toward zero (equivalent to a symmetric/Neumann edge condition).

Returns a `Float64` vector the same length as `y`.
"""
function gaussfilt(y::AbstractVector{<:Real}, σ::Real; truncate::Real = 4.0)
    n = length(y)
    yf = float.(y)
    σ <= 0 && return yf                        # no smoothing
    hw = ceil(Int, truncate * σ)               # half-width in samples
    offs = -hw:hw
    k = @. exp(-0.5 * (offs / σ)^2)            # unnormalised Gaussian weights
    out = similar(yf)
    @inbounds for i in 1:n
        num = 0.0
        den = 0.0
        for (jj, j) in enumerate(offs)
            ii = i + j
            (1 <= ii <= n) || continue         # skip out-of-range → renormalise
            num += k[jj] * yf[ii]
            den += k[jj]
        end
        out[i] = num / den
    end
    return out
end

"""
    gaussfilt(x, y, L; fwhm=false, truncate=4.0)

Same as above but the smoothing scale `L` is given in the **physical units of
`x`** (assumes uniform spacing, `dx = x[2]-x[1]`).

- `fwhm=false` (default): `L` is the Gaussian standard deviation.
- `fwhm=true`: `L` is the full width at half maximum (σ = L / 2.3548).

Example: smooth on a 20 km scale with `xc` in metres → `gaussfilt(xc, cge, 20e3)`.
"""
function gaussfilt(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, L::Real;
                   fwhm::Bool = false, truncate::Real = 4.0)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    dx = x[2] - x[1]
    σsamples = (fwhm ? L / (2 * sqrt(2 * log(2))) : L) / dx
    return gaussfilt(y, σsamples; truncate = truncate)
end