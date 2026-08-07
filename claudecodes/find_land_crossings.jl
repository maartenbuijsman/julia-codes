# find_land_crossings.jl
# MCB/Claude, USM, 2026-8-5
#
# At each of the 14 target latitudes, walk outward (west and east) from 40 W
# along the Mercator land/sea mask (missing T,S = land) until first hitting
# land -- i.e. the true coastline (Americas to the west, Africa/Europe to the
# east), as opposed to find_shelf_2000m_crossings.jl's 2000 m depth
# threshold, which can catch Mid-Atlantic Ridge/seamount topography instead
# of a real coast. Plots these coastline crossings on a lon-lat map next to
# the previous 2000 m crossings and the fixed 60W-20W band.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using CairoMakie

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
dirfig = "/home/mbui/ModelOutput/figs/"

lonsearch = (-95.0, 25.0)
loncenter = -40.0
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat = length(latsels)
oldband = (-60.0, -20.0)

ds = NCDataset(string(dirin,"gl12_mean_1993_2016_allmonths.nc"),"r")
longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]

ilo, _ = nearest_index(longitude, lonsearch[1])
ihi, _ = nearest_index(longitude, lonsearch[2])
ic0, _ = nearest_index(longitude, loncenter)
lonvec = longitude[ilo:ihi]

lonwest_land = fill(NaN, nlat)
loneast_land = fill(NaN, nlat)

for k in 1:nlat
    js, _ = nearest_index(latitude, latsels[k])
    Tsurf = ds["thetao"][ilo:ihi,js,1,1]   # surface level only, (lon,)
    island = ismissing.(Tsurf)
    n = length(island)

    ic = ic0 - ilo + 1
    @assert !island[ic] "loncenter is land at lat=$(latsels[k])"

    iw = ic
    while iw > 1 && !island[iw-1]
        iw -= 1
    end
    lonwest_land[k] = lonvec[iw]

    ie = ic
    while ie < n && !island[ie+1]
        ie += 1
    end
    loneast_land[k] = lonvec[ie]

    println("lat=",latsels[k]," west coast=",lonwest_land[k]," east coast=",loneast_land[k],
            " (basin width=",round(loneast_land[k]-lonwest_land[k],digits=1)," deg)")
end

# ---------------------------------------------------------------------------
# manual override: at 15N the raw land-crossing walk threads through the
# Caribbean/Central America (see conversation) rather than stopping at the
# edge of the open Atlantic; user-specified open-ocean bounds used instead
overrides = Dict(15.0 => (-60.0, -17.0))
for (latv, (w,e)) in overrides
    k = findfirst(==(latv), latsels)
    println("overriding lat=",latv," west=",lonwest_land[k],"->",w," east=",loneast_land[k],"->",e)
    lonwest_land[k] = w
    loneast_land[k] = e
end

# ---------------------------------------------------------------------------
# coarse background land/sea mask for map context

lonbg = longitude[ilo:4:ihi]
latlo, _ = nearest_index(latitude, -5.0)
lathi, _ = nearest_index(latitude, 65.0)
latbg = latitude[latlo:4:lathi]

Tsurfbg = ds["thetao"][ilo:4:ihi, latlo:4:lathi, 1, 1]
landmask = ismissing.(Tsurfbg)

close(ds)

# ---------------------------------------------------------------------------
# plot

fig = Figure(size = (1100, 900))
ax = Axis(fig[1,1], xlabel = "Longitude", ylabel = "Latitude",
          title = "Coastline crossings (west/east of 40°W) at each target latitude")

heatmap!(ax, lonbg, latbg, Float64.(landmask), colormap = [:transparent, (:gray, 0.5)])

vlines!(ax, [oldband[1], oldband[2]], color = :blue, linestyle = :dot, linewidth = 1.5,
        label = "fixed 60W-20W band used")

for k in 1:nlat
    lines!(ax, [lonwest_land[k], loneast_land[k]], [latsels[k], latsels[k]], color = :black, linewidth = 1)
end
scatter!(ax, lonwest_land, latsels, color = :firebrick, markersize = 10, label = "west coast")
scatter!(ax, loneast_land, latsels, color = :seagreen, markersize = 10, label = "east coast")

xlims!(ax, lonsearch...)
ylims!(ax, -5, 65)
axislegend(ax, position = :lt, framevisible = true)

save(string(dirfig,"land_crossings_map.png"), fig)
println("saved: ", string(dirfig,"land_crossings_map.png"))
