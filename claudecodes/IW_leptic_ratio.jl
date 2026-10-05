#= IW_leptic_ratio.jl
Maarten Buijsman, USM DMS, 2026-9-29

Grid lepticity and the ratio of numerical to physical dispersion,
Vitousek & Fringer (2011, Ocean Modelling 40, 72-86), eqs. 39-41:

    Gamma = K * lambda_e^2,   lambda_e = Dx / h_e,
    h_e   = sqrt( 3 * int phi^2 dz / int (dphi/dz)^2 dz )

phi(z) = mode-1 vertical-displacement (= w) eigenfunction of the long-wave
(hydrostatic, non-rotating) problem phi'' + N^2/c^2 phi = 0, i.e. the phi of
the KdV coefficients (their eq. 19, Liu 1988). h_e is the depth that makes the
physical dispersion (k h_e)^2/6 -- the same integral ratio as the KdV
dispersion coefficient beta = (c/2) int phi^2 / int phi'^2 in
oceananigans_IW/IW_nondim_params.jl, so h_e = sqrt(6 beta / c). For uniform
N, h_e = sqrt(3) d / pi = 0.55 d.

K for Oceananigans NonhydrostaticModel (not in the paper):
- WENO() only discretizes ADVECTION. The linear terms that set the leading
  numerical dispersion -- horizontal pressure gradient and divergence -- are
  2nd-order centred on the staggered C-grid (the FFT Poisson solver inverts
  the 2nd-order discrete Laplacian). So the paper's 2nd-order analysis holds.
- Staggered C-grid: omega = c (2/Dx) sin(k Dx/2) = c k [1 - (k Dx)^2/24],
  vs physical omega = c k [1 - (k h_e)^2/6]  ->  K = (1/24)/(1/6) = 1/4.
  (The paper's unstaggered leap-frog KdV scheme has sin(k Dx)/Dx -> K = 1 - C^2.)
- Default timestepper RungeKutta3: phase error is O((omega Dt)^4), so it does
  not change K at O((k Dx)^2) (leap-frog's -C^2 has no RK3 counterpart).
  The paper's own SUNTANS fit (C-grid) gave K = 0.075.
=#

using Printf, JLD2, Trapz

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

const T2 = 12 + 25.2/60
const ω  = 2π / (T2*3600)
const Dt = 15.0                           # s; wizard Δt while NH solitary waves form (Maarten); 120 s cap otherwise
const KS = (Cgrid = 0.25, KdV = 1.0, SUNTANS = 0.075)

rows = get_runs(15, collect(27:39))       # zonal-mean N2, 0-50 N, Dx = 200 m

# h_e from the first-mode W eigenfunction on the model's own (WKB) z-faces
function h_equiv(zfw, N2w, f, om, nonhyd)
    _, _, _, _, Ce, Weig, _, _ = sturm_liouville_noneqDZ_norm(zfw, N2w, f, om, nonhyd)
    φ  = Weig[:, 1]
    zc = (zfw[1:end-1] .+ zfw[2:end]) ./ 2
    φz = diff(φ) ./ diff(zfw)
    he = sqrt(3 * abs(trapz(zfw, φ.^2)) / abs(trapz(zc, φz.^2)))
    return he, Ce[1]
end

@printf("%6s %6s %6s %7s %7s %6s %6s %8s %8s %8s %6s\n",
        "lat", "H", "Dx", "h_e", "h_e/H", "he_D2", "c1", "lam_e", "G(1/4)", "G(1)", "C")
for r in rows
    @load string(dirforce, n2_filename(r)) N2w zfw
    zfw = Float64.(zfw); N2w = Float64.(N2w)
    H  = maximum(zfw) - minimum(zfw)
    he, c1 = h_equiv(zfw, N2w, 0.0, ω, 0)                  # KdV long-wave limit
    heD2, _ = h_equiv(zfw, N2w, coriolis(r.lat), ω, 1)     # D2, rotating, NH (sensitivity)
    λe = r.DX / he
    @printf("%6.1f %6.0f %6.0f %7.0f %7.3f %6.0f %6.2f %8.3f %8.4f %8.4f %6.2f\n",
            r.lat, H, r.DX, he, he/H, heD2, c1, λe, KS.Cgrid*λe^2, KS.KdV*λe^2, c1*Dt/r.DX)
end

# sanity check of the h_e integral: uniform N -> sqrt(3)/pi * d
zt = collect(range(0.0, -4000.0, length = 201))
het, _ = h_equiv(zt, fill(1e-5, length(zt)), 0.0, ω, 0)
@printf("\nuniform-N check: h_e/d = %.4f (theory sqrt(3)/pi = %.4f)\n", het/4000, sqrt(3)/π)
