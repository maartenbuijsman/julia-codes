# plot_N2_zonal_transect_2p5N.jl
# MCB/Claude, USM, 2026-8-5
# Zonal (longitude-depth) transect of Mercator N2 at 2.5 N, upper 500 m, over
# a wide longitude span, to visually check whether the 60W-20W band used for
# the open-Atlantic zonal mean (extract_TS_N2_Atlantic_zonalmean_40W.jl) is a
# reasonable choice -- i.e. actually open ocean, not clipping land/shelf.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater
using CairoMakie

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
dirfig = "/home/mbui/ModelOutput/figs/"

latsel = 2.5
lonrange = (-80.0, 20.0)   # wide span: South America coast to West Africa coast
maxdepth_plot = 500.0
band = (-60.0, -20.0)      # the zonal-mean band actually used

ds = NCDataset(string(dirin,"gl12_mean_1993_2016_allmonths.nc"),"r")
longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])

js, _ = nearest_index(latitude, latsel)
ilo, _ = nearest_index(longitude, lonrange[1])
ihi, _ = nearest_index(longitude, lonrange[2])

iz = findall(depth .<= maxdepth_plot)
nz_plot = length(iz)

println("lat used = ", latitude[js], ", lon range used = ", longitude[ilo], " to ", longitude[ihi],
        " (", ihi-ilo+1, " columns)")

Tblock = ds["thetao"][ilo:ihi,js,iz,1]   # (lon, depth)
Sblock = ds["so"][ilo:ihi,js,iz,1]
ncol = size(Tblock,1)

depth_mid = (depth[iz[1:end-1]] .+ depth[iz[2:end]]) ./ 2
N2grid = fill(NaN, nz_plot-1, ncol)
bottom_depth = fill(NaN, ncol)

for c in 1:ncol
    Ts = Tblock[c,:]; Ss = Sblock[c,:]
    igood = findall(!ismissing, Ts .+ Ss)
    isempty(igood) && continue
    bottom_depth[c] = depth[iz[igood[end]]]
    length(igood) < 2 && continue

    Tsg = Float64.(Ts[igood]); Ssg = Float64.(Ss[igood]); dpk = depth[iz[igood]]
    nzk = length(dpk)

    p  = gsw_p_from_z.(-dpk, latitude[js])
    SA = gsw_sa_from_sp.(Ssg, p, longitude[ilo+c-1], latitude[js])
    CT = gsw_ct_from_pt.(SA, Tsg)

    N2k = zeros(nzk-1); Pmidk = zeros(nzk-1)
    Lats = fill(latitude[js], nzk)
    gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

    N2grid[1:nzk-1,c] = N2k
end

close(ds)

lonvec = longitude[ilo:ihi]

fig = Figure(size = (1300, 600))
ax = Axis(fig[1,1], xlabel = "Longitude", ylabel = "Depth (m)",
          title = string("Mercator N² zonal transect at ", latitude[js], "°N, upper ", Int(maxdepth_plot), " m"))

hm = heatmap!(ax, lonvec, -depth_mid, permutedims(N2grid), colormap = :viridis)
Colorbar(fig[1,2], hm, label = "N² (s⁻²)")

lines!(ax, lonvec, -bottom_depth, color = :black, linewidth = 1.5)

vlines!(ax, [band[1], band[2]], color = :red, linestyle = :dash, linewidth = 2)

ylims!(ax, -maxdepth_plot, 0)

save(string(dirfig,"N2_zonal_transect_2p5N_upper500m.png"), fig)
println("saved: ", string(dirfig,"N2_zonal_transect_2p5N_upper500m.png"))
