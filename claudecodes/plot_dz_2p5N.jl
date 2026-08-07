# plot_dz_2p5N.jl
# MCB/Claude, USM, 2026-8-6
# Scatter plot of depth vs. grid spacing (dz) for the 2.5 N WKB-scaled
# forcing grid (the shared reference grid used for all latitudes in
# build_N2_forcing_Mercator_zonalmean_40W.jl).

using JLD2
using CairoMakie

dirforce = "/home/mbui/ModelOutput/IW/forcingfiles/"
dirfig   = "/home/mbui/ModelOutput/figs/"

f = jldopen(string(dirforce,"N2_ZonalMeanAtl_lat02.5.jld2"),"r")
zfw = f["zfw"]
close(f)

dz = diff(zfw)
zmid = (zfw[1:end-1] .+ zfw[2:end]) ./ 2

fig = Figure(size = (900, 700))

ax1 = Axis(fig[1,1], xlabel = "Δz (m)", ylabel = "Depth (m)", title = "Full water column")
scatter!(ax1, dz, zmid)

ax2 = Axis(fig[1,2], xlabel = "Δz (m)", ylabel = "Depth (m)", title = "Upper 400 m")
scatter!(ax2, dz, zmid)
ylims!(ax2, -400, 0)

Label(fig[0,1:2], "WKB grid spacing at 2.5°N (reference grid, Nz=$(length(zfw)-1))", fontsize = 16)

save(string(dirfig,"dz_2p5N_scatter.png"), fig)
println("saved: ", string(dirfig,"dz_2p5N_scatter.png"))
