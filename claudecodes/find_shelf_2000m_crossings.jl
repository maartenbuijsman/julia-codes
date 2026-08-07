# find_shelf_2000m_crossings.jl
# MCB/Claude, USM, 2026-8-5
#
# At each of the 14 target latitudes, walk outward (west and east) from a
# reference longitude (-40 W, the original transect) along the Mercator
# bottom-depth mask (deepest valid/unmasked T,S level, same proxy used
# elsewhere in this project) until the sea floor first shoals to <= 2000 m --
# i.e. the westernmost/easternmost edge of the open-ocean corridor around
# 40 W at each latitude. Plots these crossing points on a lon-lat map, with a
# coarse land/sea background for context, and the fixed 60W-20W band used in
# extract_TS_N2_Atlantic_zonalmean_40W.jl for comparison.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using CairoMakie

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
dirfig = "/home/mbui/ModelOutput/figs/"

lonsearch = (-95.0, 25.0)   # wide search window
loncenter = -40.0
threshold = 2000.0
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat = length(latsels)
oldband = (-60.0, -20.0)

ds = NCDataset(string(dirin,"gl12_mean_1993_2016_allmonths.nc"),"r")
longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])

ilo, _ = nearest_index(longitude, lonsearch[1])
ihi, _ = nearest_index(longitude, lonsearch[2])
ic0, _ = nearest_index(longitude, loncenter)
lonvec = longitude[ilo:ihi]

lonwest = fill(NaN, nlat)
loneast = fill(NaN, nlat)

for k in 1:nlat
    js, _ = nearest_index(latitude, latsels[k])
    Tb = ds["thetao"][ilo:ihi,js,:,1]   # (lon, depth)
    n = size(Tb,1)
    bottom = fill(NaN, n)
    for c in 1:n
        igood = findall(!ismissing, Tb[c,:])
        isempty(igood) || (bottom[c] = depth[igood[end]])
    end

    ic = ic0 - ilo + 1   # index of loncenter within this row's local array
    @assert !isnan(bottom[ic]) && bottom[ic] > threshold "loncenter is not deep water at lat=$(latsels[k])"

    iw = ic
    while iw > 1 && !isnan(bottom[iw-1]) && bottom[iw-1] > threshold
        iw -= 1
    end
    lonwest[k] = lonvec[iw]

    ie = ic
    while ie < n && !isnan(bottom[ie+1]) && bottom[ie+1] > threshold
        ie += 1
    end
    loneast[k] = lonvec[ie]

    println("lat=",latsels[k]," west crossing=",lonwest[k]," east crossing=",loneast[k],
            " (corridor width=",round(loneast[k]-lonwest[k],digits=1)," deg)")
end

# ---------------------------------------------------------------------------
# coarse background land/sea mask for map context (surface level only,
# subsampled for speed)

lonbg = longitude[ilo:4:ihi]
latlo, _ = nearest_index(latitude, -5.0)
lathi, _ = nearest_index(latitude, 65.0)
latbg = latitude[latlo:4:lathi]

Tsurf = ds["thetao"][ilo:4:ihi, latlo:4:lathi, 1, 1]   # (lon, lat)
landmask = ismissing.(Tsurf)

close(ds)

# ---------------------------------------------------------------------------
# plot

fig = Figure(size = (1100, 900))
ax = Axis(fig[1,1], xlabel = "Longitude", ylabel = "Latitude",
          title = "2000 m shelf-break crossings (west/east of 40°W) at each target latitude")

heatmap!(ax, lonbg, latbg, Float64.(landmask), colormap = [:transparent, (:gray, 0.5)])

vlines!(ax, [oldband[1], oldband[2]], color = :blue, linestyle = :dot, linewidth = 1.5,
        label = "fixed 60W-20W band used")

for k in 1:nlat
    lines!(ax, [lonwest[k], loneast[k]], [latsels[k], latsels[k]], color = :black, linewidth = 1)
end
scatter!(ax, lonwest, latsels, color = :firebrick, markersize = 10, label = "west crossing")
scatter!(ax, loneast, latsels, color = :seagreen, markersize = 10, label = "east crossing")

xlims!(ax, lonsearch...)
ylims!(ax, -5, 65)
axislegend(ax, position = :lt, framevisible = true)

save(string(dirfig,"shelf_2000m_crossings_map.png"), fig)
println("saved: ", string(dirfig,"shelf_2000m_crossings_map.png"))
