#= IW_GM16_calibration_eval.jl
Maarten Buijsman, USM DMS, 2026-9-30

Evaluate the series-16 GM81 calibration runs 16.91-93 (GM only, lat 0, 5,
28.8 N, hourly output) and set the per-latitude initial level GMs for the
production runs 16.1-12 (GM only) and 16.27-38 (GM + tide).

For each calibration run (same diagnostic as IW_GM15_decay_vs_GM76.jl:
depth-integrated KE + APE, zonal mean over x = 100-1800 km, every 12 h):
    achieved = mean E(day 10-20) / E_GM81       (target 1)
    R16      = mean E(day 10-20) / E(0)
    c        = R16 / R15,  R15 = 1/s0 (the first guess used in params_16_cal.jl)
c is interpolated linearly in latitude between 0, 5 and 28.8 N (held constant
beyond 28.8 N) and the production level is GMs = s0 / c for every latitude.
=#

using NCDatasets, Printf, Statistics, JLD2, Trapz

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirsim    = string(pth0, "IW/")
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))
include("/home/mbui/Documents/julia-codes/claudecodes/gm81_ic.jl")

const rho0 = 1020.0
const xlo, xhi = 100e3, 1800e3

LATS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0]
S0   = [2.06, 2.16, 1.92, 2.13, 2.05, 2.34, 2.25, 2.38, 2.37, 2.29, 2.43, 2.52]   # = params_16_cal / IW_GM81_IC_check

function energy_timeseries(r)
    @load string(dirforce, n2_filename(r)) N2w zfw
    zord = sortperm(zfw)
    N2c  = (N2w[zord][1:end-1] .+ N2w[zord][2:end]) ./ 2
    ds   = NCDataset(string(dirsim, @sprintf("AMZexpt%02i.%02i", r.mainnm, r.runnm), ".nc"), "r")
    tday = ds["time"][:] ./ 86400
    dz   = ds["Δz_aac"][:]
    Ix   = findall(xlo .<= ds["x_caa"][:] .<= xhi)
    Isel = unique([argmin(abs.(tday .- d)) for d in 0:0.5:floor(tday[end])])
    E = [gm81_energy(ds["u"][:, :, it], ds["v"][:, :, it], ds["w"][:, :, it], ds["b"][:, :, it],
                     dz, N2c, Ix; rho0 = rho0)[1] for it in Isel]
    t = tday[Isel]
    close(ds)
    return t, E, gm81_EGM(zfw, N2w; rho0 = rho0)
end

rows = get_runs(16, [91, 92, 93])
latc = [r.lat for r in rows]; cc = zeros(3)
@printf("%5s %6s %9s %9s %8s %8s %6s\n", "lat", "s0", "E(0)/GM", "E10-20/GM", "R16", "R15", "c")
for (i, r) in enumerate(rows)
    t, E, EGM = energy_timeseries(r)
    s0 = S0[findfirst(==(r.lat), LATS)]
    I = findall(10 .<= t .<= 20)
    R16 = mean(E[I]) / E[1]; cc[i] = R16 * s0
    @printf("%5.1f %6.2f %9.3f %9.3f %8.3f %8.3f %6.3f\n", r.lat, s0, E[1]/EGM, mean(E[I])/EGM, R16, 1/s0, cc[i])
end

cint(lat) = lat >= latc[end] ? cc[end] :
            (k = findlast(latc .<= lat); cc[k] + (cc[k+1] - cc[k]) * (lat - latc[k]) / (latc[k+1] - latc[k]))
GMs = round.(S0 ./ cint.(LATS), digits = 2)
println("\nproduction GMs (lat ", LATS, "):")
println("GMs = ", GMs)
