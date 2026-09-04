#= IW_APEt_x_check.jl
Maarten Buijsman, USM DMS, 2026-9-4

Quick diagnostic: plot APEt(x) for one run over x=80-150km, to see what the
single nearest-grid-point sample (x~100km, used in
claudecodes/IW_mode1_theory_vs_lat.jl for the measured-vs-theory comparison)
actually looks like in context -- no averaging was done there, just the
nearest grid point (x=98km on this 4km grid).
=#

using JLD2, Printf, CairoMakie

dirout = "/home/mbui/ModelOutput/diagout/"
dirfig = "/home/mbui/ModelOutput/figs/"

mainnm, runnm = 10, 59   # lat=25, "F=25kW/m fixed N2 @50N" block (53:65)
fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
d = load(string(dirout, "energetics_", fnames, ".jld2"))
xc = d["xc"] ./ 1e3   # km

Ix = findall(80.0 .<= xc .<= 150.0)
xmeas_km = 98.0   # the single point actually used (nearest grid point to 100km)
imeas = argmin(abs.(xc .- xmeas_km))

fig = Figure(size=(700,450))
ax = Axis(fig[1,1], title=string("APEt(x), ", fnames, " (lat=25)"),
    xlabel="x [km]", ylabel="APEt [J/m²]")
lines!(ax, xc[Ix], d["APEt"][Ix], color=:black, linewidth=2)
scatter!(ax, [xc[imeas]], [d["APEt"][imeas]], color=:darkorange, markersize=14,
    strokecolor=:black, strokewidth=1, label="point used (x=98km)")
axislegend(ax, position=:rt)
display(fig)
save(string(dirfig, "APEt_x_check_", fnames, ".png"), fig)
println("saved APEt_x_check_", fnames, ".png")
