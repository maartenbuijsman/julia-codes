#= IW_GM81_IC_check.jl
Maarten Buijsman, USM DMS, 2026-9-30

CPU check of the series-16 GM81 initial condition (claudecodes/gm81_ic.jl)
before any GPU time is spent: for every latitude, build the field exactly as
IW_GM81_flux_LAT_2000km_bash_cuda.jl does, evaluate it on a subsampled model
grid (every 2 km in x = 100-1800 km, all model levels) and compare its
depth-integrated KE + APE with the GM81 reference E_GM81 = rho0 b² N0 E0 ∫N dz.

Printed per latitude:
  Eexp/GM   expected energy of the synthesized field (sum over components)
  Esmp/GM   energy of the actual random-phase realization on the subsampled grid
  alpha     the factor the simulation will apply, sqrt(s0 E_GM81 / E_IC), for
            the first-guess s0 = 1/R15 (series-15 retention, see #65)
Esmp/GM should be close to (a little below) 1: the corrected amplitude gives
1x GM81 before the cutoffs (6 DX <= λ <= L) remove some energy.
=#

using Printf, JLD2, Statistics, Trapz

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))
include("/home/mbui/Documents/julia-codes/claudecodes/gm81_ic.jl")

const rho0 = 1020.0
const L, DX = 2_000_000.0, 200.0
const Sp_left, Sp_right = 40_000.0, 200_000.0
const xs = collect(100e3:2e3:1800e3)        # subsampled x [m]

lats = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0]
s0   = [2.06, 2.16, 1.92, 2.13, 2.05, 2.34, 2.25, 2.38, 2.37, 2.29, 2.43, 2.52]
# (0 N: 1/R15 with the inert v removed from both E(0) and the day-10-20 mean)

@printf("%5s %6s %7s %8s %8s %8s %7s %7s %6s %6s\n", "lat", "ncomp", "Emiss", "EGM kJ", "Eexp/GM",
        "Esmp/GM", "KE/APE", "v2/u2", "s0", "alpha")
for (lat, s) in zip(lats, s0)
    @load string(dirforce, @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", lat)) N2w zfw
    zfw = Float64.(zfw); N2w = Float64.(N2w)
    fmod = 2GM81_OMEGA_EARTH*sind(lat)
    ic = build_gm81_ic(zfw, N2w, fmod, L, DX, Sp_left, Sp_right)

    zc  = (zfw[1:end-1] .+ zfw[2:end]) ./ 2; dz = abs.(diff(zfw))
    N2c = max.((N2w[1:end-1] .+ N2w[2:end]) ./ 2, 1e-12)
    U2 = zeros(length(xs)); V2 = zeros(length(xs)); W2 = zeros(length(xs)); P2 = zeros(length(xs))
    Threads.@threads for i in eachindex(xs)
        x = xs[i]
        u = ic.u.(x, zc); v = ic.v.(x, zc); b = ic.b.(x, zc)
        wf = ic.w.(x, zfw); w = (wf[1:end-1] .+ wf[2:end]) ./ 2
        U2[i] = sum(u.^2 .* dz); V2[i] = sum(v.^2 .* dz); W2[i] = sum(w.^2 .* dz)
        P2[i] = sum(b.^2 ./ N2c .* dz)
    end
    KE  = 0.5rho0*(mean(U2) + mean(V2) + mean(W2)); APE = 0.5rho0*mean(P2)
    Esmp = KE + APE
    @printf("%5.1f %6d %7.3f %8.2f %8.3f %8.3f %7.2f %7.3f %6.2f %6.3f\n", lat, ic.ncomp, ic.Emiss_frac,
            ic.EGM81/1e3, ic.Eexp/ic.EGM81, Esmp/ic.EGM81, KE/APE, mean(V2)/mean(U2), s, sqrt(s*ic.EGM81/Esmp))
end
