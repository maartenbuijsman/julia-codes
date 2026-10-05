#= gm81_ic.jl
Maarten Buijsman, USM DMS, 2026-9-30

Garrett-Munk internal-wave initial condition in the version of Munk (1981)
("GM81", same B, H, E as https://github.com/joernc/GM81), for the 2-D (x,z)
AMZ runs from series 16 on. Shared by the simulation script
(IW_GM81_flux_LAT_2000km_bash_cuda.jl) and the CPU check (IW_GM81_IC_check.jl)
so both build exactly the same field.

SPECTRUM (Munk 1981; b = 1300 m, N0 = 5.24e-3 rad/s, E0 = 6.3e-5, j* = 3)
    E(ω,j) = E0 B(ω) H(j),  B(ω) = (2/π) f_B/(ω sqrt(ω² - f_B²)),
    H(j)   = (j² + j*²)⁻¹ / Σ_j' (j'² + j*²)⁻¹
    local energy per unit mass (WKB):  e(z) = b² N0 N(z) E0
    depth-integrated reference:        E_GM81 = rho0 b² N0 E0 ∫N dz
f_B = max(|f|, f(2.5°)) is used ONLY in B(ω) and for the lower end of the
frequency range: GM76/81's B(ω) collapses to zero as f -> 0 (it is a
mid-latitude parameterization built around the near-inertial peak), so it
is floored there. This is a pragmatic approximation, not a rigorous
equatorial spectrum.

MODES AND AMPLITUDES
At each of Nfreq log-spaced frequencies ω_n (f_B < ω < N_max, bin width
Δω_n) the non-hydrostatic Sturm-Liouville solver gives, for every vertical
mode j, k_j(ω_n), the horizontal-velocity eigenfunction U_j(z) (Ueig2,
depth-mean-square = 1) and W_j(z) (W_j' = k_j U_j). The modes and the
polarization use the MODEL's f (fmod), so at the equator v = 0 -- in
series 13-15 the floored f was used here too, which left a v field that the
f = 0 model cannot carry as a wave. The exact modes already carry the WKB
depth scaling (U_j² ~ N(z)/<N>), so the amplitude only supplies the depth
mean of Munk's energy density:
    A_nj² = 2 b² N0 <N> E0 B(ω_n) H(j) Δω_n,   <N> = (1/H) ∫N dz
For a hydrostatic linear wave the depth-mean KE + APE per unit mass of a
component is A²/2, split as Munk's <u²+v²> ~ (1 + f²/ω²) and
N²<ζ²> ~ (1 - f²/ω²). (Series 13-15 used A² = 2 b² N0² (1 + f²/ω²) E0 B H Δω:
N0 instead of <N> puts the thermocline energy level over the whole column,
and the (1 + f²/ω²) put all of <u²+v²> into u with v added on top -- about
3x GM81 in total.)
Components are kept only if 6 DX <= 2π/k_j <= L: shorter ones are not
resolved on the grid, longer ones do not complete a cycle in this bounded
domain. The energy they carry is not synthesized; the caller restores the
target level with one global factor alpha (see below), which is identical to
the old proportional "redistribution" boost of every surviving component.

FIELDS (random phase φ and random direction σ = ±1 per component, seeded)
    u  =  τ(x) Σ A U_j(z) cos θ,              θ = σ k_j x + φ
    v  =  τ(x) Σ (f/ω) A U_j(z) sin θ
    w  =  τ(x) Σ σ A W_j(z) sin θ             (continuity)
    b' = -τ(x) Σ (σ A/ω) N²(z) W_j(z) cos θ    (b' = -N² ζ, ζ = ∫w dt)
τ(x) tapers the field to zero over the production sponges.

ENERGY NORMALIZATION (done by the caller on the actual model fields)
    alpha = sqrt(s E_GM81 / E_IC),  E_IC = < ∫ rho0/2 (u²+v²+w²+b'²/N²) dz >_x
over x = 100-1800 km, so the run starts at exactly s x GM81.
=#

using Random, Trapz, Statistics
import Interpolations   # import, not using: Interpolations exports `Flat`, which would clash
                        # with Oceananigans.Flat if this file is included before the grid is built

const GM81_E0 = 6.3e-5      # GM energy parameter (dimensionless)
const GM81_JS = 3.0         # mode-number scale j*
const GM81_N0 = 5.24e-3     # reference buoyancy frequency [rad/s] (3 cph)
const GM81_B  = 1300.0      # stratification e-folding scale [m]
const GM81_JSUM = (π*GM81_JS/tanh(π*GM81_JS) - 1)/(2*GM81_JS^2)   # Σ_{j=1}^∞ (j²+j*²)⁻¹
const GM81_OMEGA_EARTH = 7.292115e-5                             # as Oceananigans' FPlane

gm81_B(om, f) = 2/π * f/om / sqrt(om^2 - f^2)
gm81_H(j)     = 1.0 / (j^2 + GM81_JS^2) / GM81_JSUM

"depth-integrated GM81 reference energy [J/m²] for the profile N2w(zfw)"
gm81_EGM(zfw, N2w; rho0 = 1020.0) =
    rho0 * GM81_B^2 * GM81_N0 * GM81_E0 * abs(trapz(zfw, sqrt.(max.(N2w, 0.0))))

"""
    gm81_energy(u, v, w, b, dz, N2c, Ix; rho0=1020.0)

Mean over the x-indices `Ix` of the depth-integrated KE + APE [J/m²] of
fields on the model grid: `u` on x-faces (Nx+1, Nz), `v`, `b` at centres
(Nx, Nz), `w` on z-faces (Nx, Nz+1); `dz` and `N2c` at cell centres, same
vertical order as the fields. Same definition as IW_GM15_decay_vs_GM76.jl.
Returns (E, KE, APE).
"""
function gm81_energy(u, v, w, b, dz, N2c, Ix; rho0 = 1020.0)
    uc  = (u[1:end-1, :] .+ u[2:end, :]) ./ 2
    wc  = (w[:, 1:end-1] .+ w[:, 2:end]) ./ 2
    dzz = reshape(dz, 1, :)
    KE  = mean(0.5rho0 .* vec(sum((uc.^2 .+ v.^2 .+ wc.^2) .* dzz, dims = 2))[Ix])
    APE = mean(0.5rho0 .* vec(sum(b.^2 ./ max.(reshape(N2c, 1, :), 1e-12) .* dzz, dims = 2))[Ix])
    return KE + APE, KE, APE
end

"""
    ic = build_gm81_ic(zfw, N2w, fmod, L, DX, Sp_left, Sp_right; nonhyd=1, seed=1,
                       Nfreq=60, minwavelenfac=6, LAT_f_floor=2.5, rho0=1020.0)

Build the GM81 u, v, w, b' initial-condition functions of (x, z) (unit
normalization, alpha = 1) plus metadata. `zfw`, `N2w`: faces and N² from the
N2 forcing file; `fmod`: the model's Coriolis parameter. Returns a NamedTuple
(u, v, w, b, ncomp, Emiss_frac, f_B, Nmean, EGM81, Eexp); `Eexp` is the
expected depth-integrated KE + APE [J/m²] of the synthesized field (sum of the
component energies, cross terms average out).
"""
function build_gm81_ic(zfw, N2w, fmod, L, DX, Sp_left, Sp_right; nonhyd = 1, seed = 1,
                       Nfreq = 60, minwavelenfac = 6, LAT_f_floor = 2.5, rho0 = 1020.0)
    zfw = Float64.(zfw); N2w = Float64.(N2w)
    zcw = (zfw[1:end-1] .+ zfw[2:end]) ./ 2
    dzf = abs.(diff(zfw)); H = sum(dzf)

    fm  = abs(fmod)                                           # modes + polarization
    f_B = max(fm, 2GM81_OMEGA_EARTH*sind(LAT_f_floor))        # B(ω) + lower frequency bound
    Nw    = sqrt.(max.(N2w, 0.0))
    Nmean = abs(trapz(zfw, Nw)) / H
    N_max = sqrt(maximum(N2w))
    EGM81 = gm81_EGM(zfw, N2w; rho0 = rho0)

    Random.seed!(seed)
    edges = exp.(range(log(f_B*1.001), log(N_max*0.999), length = Nfreq + 1))
    omid  = sqrt.(edges[1:end-1] .* edges[2:end])
    domg  = diff(edges)

    comp_k = Float64[]; comp_A = Float64[]; comp_om = Float64[]
    comp_U = Vector{Vector{Float64}}(); comp_W = Vector{Vector{Float64}}()
    E_miss = 0.0; E_pres = 0.0
    for n in 1:Nfreq
        om = omid[n]
        kn, Lw, _, _, _, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fm, om, nonhyd)
        for j in 1:length(kn)
            Lw[j] < minwavelenfac*DX && continue             # unresolved on the grid -- dropped
            Ecomp = GM81_B^2 * GM81_N0 * Nmean * GM81_E0 * gm81_B(om, f_B) * gm81_H(j) * domg[n]
            if Lw[j] > L                                      # too long for the domain -- dropped;
                E_miss += Ecomp                               # alpha restores the level
                continue
            end
            E_pres += Ecomp
            # re-derive the solver's own Ueig normalization factor (Ueig2 = Ueig/norm_factor,
            # see sturm_liouville_noneqDZ_norm.jl) so W is on the SAME amplitude scale as Ueig2
            nf = sqrt(sum(Ueig[:, j].^2 .* dzf) / H)
            push!(comp_k, kn[j]); push!(comp_A, sqrt(2Ecomp)); push!(comp_om, om)
            push!(comp_U, Ueig2[:, j])
            push!(comp_W, nf == 0 ? zero(Weig[:, j]) : Weig[:, j] ./ nf)
        end
    end
    ncomp = length(comp_k)
    phase = 2π .* rand(ncomp)
    dsgn  = rand([-1.0, 1.0], ncomp)

    # expected depth-integrated energy of the synthesized field [J/m²]:
    # rho0 A²/4 ∫[U²(1 + f²/ω²) + W² + N² W²/ω²] dz per component
    Eexp = 0.0
    for i in 1:ncomp
        IU = sum(comp_U[i].^2 .* dzf)
        IW = abs(trapz(zfw, comp_W[i].^2)); INW = abs(trapz(zfw, N2w .* comp_W[i].^2))
        Eexp += rho0 * comp_A[i]^2 / 4 * (IU*(1 + (fm/comp_om[i])^2) + IW + INW/comp_om[i]^2)
    end

    # per-component z-interpolants; Ueig2 lives on cell centres, W and N2 on faces
    iz  = sortperm(zcw); izf = sortperm(zfw)
    Uitp  = [Interpolations.linear_interpolation(zcw[iz], comp_U[i][iz], extrapolation_bc = Interpolations.Line()) for i in 1:ncomp]
    Witp  = [Interpolations.linear_interpolation(zfw[izf], comp_W[i][izf], extrapolation_bc = Interpolations.Line()) for i in 1:ncomp]
    N2itp = Interpolations.linear_interpolation(zfw[izf], N2w[izf], extrapolation_bc = Interpolations.Line())

    # sponge-zone taper: 1 in the interior, 0 at x = 0 and x = L, over the
    # production sponge widths
    mask2(X)  = X > 0 ? X^2 : 0.0
    taper(x)  = 1.0 - mask2((Sp_left - x)/Sp_left) - mask2((x - L + Sp_right)/Sp_right)
    fsgn = fmod                                               # signed model f for v

    # Flat-y grid: Oceananigans' set! calls these with (x, z)
    function u_ic(x, z)
        s = 0.0
        @inbounds for i in 1:ncomp
            s += comp_A[i] * Uitp[i](z) * cos(dsgn[i]*comp_k[i]*x + phase[i])
        end
        return s * taper(x)
    end
    function v_ic(x, z)
        s = 0.0
        @inbounds for i in 1:ncomp
            s += comp_A[i] * (fsgn/comp_om[i]) * Uitp[i](z) * sin(dsgn[i]*comp_k[i]*x + phase[i])
        end
        return s * taper(x)
    end
    function w_ic(x, z)
        s = 0.0
        @inbounds for i in 1:ncomp
            s += comp_A[i] * dsgn[i] * Witp[i](z) * sin(dsgn[i]*comp_k[i]*x + phase[i])
        end
        return s * taper(x)
    end
    function b_ic(x, z)
        s = 0.0
        N2z = N2itp(z)
        @inbounds for i in 1:ncomp
            s += -(comp_A[i]*dsgn[i]/comp_om[i]) * N2z * Witp[i](z) * cos(dsgn[i]*comp_k[i]*x + phase[i])
        end
        return s * taper(x)
    end

    return (u = u_ic, v = v_ic, w = w_ic, b = b_ic, ncomp = ncomp,
            Emiss_frac = E_miss/(E_miss + E_pres), f_B = f_B, Nmean = Nmean,
            EGM81 = EGM81, Eexp = Eexp)
end
