#= IW_KE_zonalavg_13gm_vs_lat.jl
Maarten Buijsman, USM DMS, 2026-9-6

Zonal-average (x=100-1800km) total KE and APE vs latitude for the 13.1-13
block (GM only, no tide, Flux=0), plus their ratio KE/APE. x-window
matches the "clean domain" convention used elsewhere in this project
(away from the source and the sponge).
=#

using NCDatasets, Printf, CairoMakie, Statistics, JLD2

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

const LAT13 = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
const xlo, xhi = 100e3, 1800e3
const fcKE = 1e-3   # J/m^2 -> kJ/m^2

runnms = collect(1:13)
KEavg  = zeros(length(runnms))
APEavg = zeros(length(runnms))

for (i, runnm) in enumerate(runnms)
    fnames = @sprintf("AMZexpt13.%02i", runnm)
    @load string(dirout, "energetics_", fnames, ".jld2") xc KE APE
    Ix = findall(xlo .<= xc .<= xhi)
    KEavg[i]  = mean(KE[Ix])  * fcKE
    APEavg[i] = mean(APE[Ix]) * fcKE
end
ratio = KEavg ./ APEavg

fig = Figure(size=(600,400))
ax = Axis(fig[1,1], title="zonal-mean KE & APE (x=100-1800 km) — 13.1-13 (GM only, no tide)",
    xlabel="latitude [°]", ylabel="energy [kJ/m2]")
lines!(ax, LAT13, KEavg,  color=:black, linewidth=2, label="KE")
scatter!(ax, LAT13, KEavg,  color=:black, markersize=10)
lines!(ax, LAT13, APEavg, color=:red,   linewidth=2, label="APE")
scatter!(ax, LAT13, APEavg, color=:red,   markersize=10)
axislegend(ax, position=:rt, framevisible=false)
display(fig)
save(string(dirfig,"KE_APE_zonalavg_13.1-13.png"), fig)
println("saved KE_APE_zonalavg_13.1-13.png")

figR = Figure(size=(600,400))
axR = Axis(figR[1,1], title="zonal-mean KE/APE ratio (x=100-1800 km) — 13.1-13 (GM only, no tide)",
    xlabel="latitude [°]", ylabel="KE/APE")
lines!(axR, LAT13, ratio, color=:black, linewidth=2)
scatter!(axR, LAT13, ratio, color=:black, markersize=10)
display(figR)
save(string(dirfig,"KE_APE_ratio_13.1-13.png"), figR)
println("saved KE_APE_ratio_13.1-13.png")

for (i,lat) in enumerate(LAT13)
    println("lat=",lat,": KE=",@sprintf("%.3f",KEavg[i])," APE=",@sprintf("%.3f",APEavg[i])," kJ/m2  KE/APE=",@sprintf("%.3f",ratio[i]))
end
