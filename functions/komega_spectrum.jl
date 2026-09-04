"""
    freq, k, power = komega_spectrum(data, dt, dx; taper=:hann, tukeycf=0.2, demean=true)

`taper` may also be a `(taper_t, taper_x)` tuple to use a different taper per
dimension -- e.g. `taper=(:none, :hann)` when the time window is an exact
whole number of the dominant period (no taper needed there, see below) but
the space window isn't.

2D (space-time) FFT wavenumber-frequency power spectrum of a real space-time
field, with the physically correct sign convention: a rightward-propagating
wave (positive phase speed) shows up at matching-sign (k,freq), i.e. positive
k at positive freq. A plain 2D FFT (same exp(-i...) kernel on both axes) does
NOT have this property on its own -- the two axes have opposite effective
sign for a traveling wave -- so the k-axis DATA (not its axis labels) is
reversed here to correct it. Verified against a reference F-K implementation
using two synthetic rightward-moving pulses (fk_reference.jl); see also the
derivation/history in claudecodes/IW_komega_spectrum.jl.

# Arguments
- `data::AbstractMatrix{<:Real}`: real-valued field, size (Nt,Nx)
  (rows=time, columns=space)
- `dt::Real`: time step (any unit; `freq` comes back in cycles per that unit)
- `dx::Real`: space step (any unit; `k` comes back in cycles per that unit)
- `taper`: `:hann`, `:tukey`, or `:none` (boxcar), or a `(taper_t, taper_x)`
  tuple for different tapers per dimension. Hann has lower sidelobes than
  Tukey (less leakage/striping, see comparison in
  claudecodes/IW_komega_spectrum.jl), at some cost to central resolution.
  Use `:none` when the record length is an exact whole number of the
  dominant period in that dimension (no edge discontinuity to suppress, so
  no taper is needed there).
- `tukeycf::Float64`: Tukey cosine-taper fraction, only used if `taper=:tukey`
- `demean::Bool`: if true (default), remove BOTH marginal means -- the
  per-x time-mean AND the per-t space-mean -- before tapering, not just the
  single global mean. A steady mean-flow profile (constant in time, varying
  in x) or a barotropic/transect-mean signal (constant in x, varying in
  time) each sit entirely at freq=0 or k=0 respectively until a taper leaks
  them into nearby bins; subtracting only the global scalar mean leaves both
  residual patterns intact.

# Units, following the same normalization as `fft_spectra` (fft_spectra_vectorized.jl):
# that function scales `fft(seg)*dt` before squaring (approximating the
# continuous Fourier transform) and multiplies the result by `df` to get a
# proper one-sided power spectral density (units data_unit^2 * dt_unit,
# e.g. m^2/s^2*s = m^2/s for a velocity time series in seconds) -- verified
# there via Parseval: sum(y.^2*dt) == sum(power*df... already folded in).
# This function does the exact 2D analog: scale by (dt*dx), square, then
# multiply by (df*dk) -- so `power` comes back in units
# data_unit^2 * dt_unit * dx_unit (e.g. m^2/s^2 * day * km if you pass dt in
# days and dx in km, which also makes `freq`/`k` come back directly in cpd /
# cycles-per-km with NO separate rescaling needed afterward -- pass dt/dx
# already in whatever units you want the output in, exactly as you would
# with `fft_spectra`). The redundant negative-frequency half is NOT folded
# here (this function returns the full two-sided range, by design, so +k/-k
# content at a fixed frequency -- independent rightward/leftward propagation,
# not a redundant mirror -- is never lost); fold that yourself (double every
# kept row except freq=0) after cropping to freq>=0, the same "select left
# side and double the power" step `fft_spectra` does internally.

# Returns
- `freq::Vector{Float64}`: frequency axis [cycles per unit of `dt`],
  ascending, full negative-to-positive range (length Nt)
- `k::Vector{Float64}`: wavenumber axis [cycles per unit of `dx`],
  ascending, full negative-to-positive range (length Nx)
- `power::Matrix{Float64}`: 2D power spectral density, units
  (data_unit)^2 * (dt_unit) * (dx_unit), size (length(freq), length(k))

# Info
Maarten Buijsman, USM DMS, 2026-9-3
"""
function komega_spectrum(data::AbstractMatrix{<:Real}, dt::Real, dx::Real;
    taper = :hann, tukeycf::Float64 = 0.2, demean::Bool = true)

    Nt, Nx = size(data)
    d = Float64.(data)

    if demean
        dbar_x = mean(d, dims=1)   # time-mean at each x,     (1,Nx)
        dbar_t = mean(d, dims=2)   # space-mean at each t,    (Nt,1)
        d = d .- dbar_x .- dbar_t .+ mean(d)
    end

    taper_t, taper_x = taper isa Tuple ? taper : (taper, taper)
    mkwin(sym, N) = sym == :hann  ? hanning(N) :
                     sym == :tukey ? tukey(N, tukeycf) :
                     sym == :none  ? ones(N) :
                     error("taper must be :hann, :tukey, or :none")
    wt = mkwin(taper_t, Nt)
    wx = mkwin(taper_x, Nx)
    d = d .* wt .* wx'

    freq = fftshift(fftfreq(Nt, 1/dt))
    k    = fftshift(fftfreq(Nx, 1/dx))
    df = freq[2] - freq[1]
    dk = k[2] - k[1]

    Ufull = fftshift(fft(d)) .* (dt*dx)   # scale by dt*dx, approximating the continuous 2D FT (units data_unit*dt_unit*dx_unit)
    U = reverse(Ufull, dims=2)            # fix rightward-propagation sign convention
    power = abs2.(U) .* (df*dk)           # -> proper PSD, units data_unit^2*dt_unit*dx_unit
    # Parseval check (verified 2026-9-3, both on random data with
    # taper=:none/demean=false and on real demeaned+Hann-tapered data):
    #   sum(d.^2)*dt*dx == sum(power)   (power already has df*dk folded in,
    #   don't multiply by df*dk again when checking -- same convention as
    #   fft_spectra's returned `P1.*df`)

    return freq, k, power
end
