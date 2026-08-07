# plot_range2_extent_map.jl
# MCB/Claude, USM, 2026-8-5
# Map of Range 2 (coastline-anchored, off-shelf >500m) west/east extent at
# each latitude, alongside Range 1 (coastline) for context.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using CairoMakie

dirmerc = "/home/mbui/ModelOutput/IW/mercator/"
dirfig  = "/home/mbui/ModelOutput/figs/"

ds = NCDataset(string(dirmerc,"MERC_N2_zonalmean_Atl_offshelf_monthly.nc"),"r")
lat = ds["latitude"][:]
range1_west = ds["range1_west"][:]
range1_east = ds["range1_east"][:]
range2_west = ds["range2_west"][:]
range2_east = ds["range2_east"][:]
close(ds)

nlat = length(lat)
lonsearch = (-95.0, 25.0)

# coarse land/sea background for context
dsa = NCDataset(string(dirmerc,"gl12_mean_1993_2016_allmonths.nc"),"r")
longitude = dsa["longitude"][:]
latitude  = dsa["latitude"][:]
ilo, _ = nearest_index(longitude, lonsearch[1])
ihi, _ = nearest_index(longitude, lonsearch[2])
latlo, _ = nearest_index(latitude, -5.0)
lathi, _ = nearest_index(latitude, 65.0)
lonbg = longitude[ilo:4:ihi]
latbg = latitude[latlo:4:lathi]
Tsurfbg = dsa["thetao"][ilo:4:ihi, latlo:4:lathi, 1, 1]
landmask = ismissing.(Tsurfbg)
close(dsa)

# ---------------------------------------------------------------------------
# plot

fig = Figure(size = (1100, 900))
ax = Axis(fig[1,1], xlabel = "Longitude", ylabel = "Latitude",
          title = "Range 2 (off-shelf, >500 m) extent at each target latitude")

heatmap!(ax, lonbg, latbg, Float64.(landmask), colormap = [:transparent, (:gray, 0.5)])

for k in 1:nlat
    lines!(ax, [range1_west[k], range1_east[k]], [lat[k], lat[k]], color = (:blue, 0.4), linewidth = 4, label = k==1 ? "Range 1 (coastline)" : nothing)
    lines!(ax, [range2_west[k], range2_east[k]], [lat[k], lat[k]], color = :black, linewidth = 2, label = k==1 ? "Range 2 (off-shelf)" : nothing)
end
scatter!(ax, range2_west, lat, color = :firebrick, markersize = 10, label = "Range 2 west")
scatter!(ax, range2_east, lat, color = :seagreen, markersize = 10, label = "Range 2 east")

xlims!(ax, lonsearch...)
ylims!(ax, -5, 65)
axislegend(ax, position = :lt, framevisible = true, unique = true)

save(string(dirfig,"range2_extent_map.png"), fig)
println("saved: ", string(dirfig,"range2_extent_map.png"))
