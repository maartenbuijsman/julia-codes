# plot_N2_zonalmean_Atl_40W.jl
# MCB/Claude, USM, 2026-8-4
# Plot the open-Atlantic (60W-20W), sea-floor>2000m zonal-mean Mercator N2
# profiles at the requested latitudes, 3 latitude-group subplots, full depth.

using NCDatasets
using CairoMakie
using ColorSchemes

dirin   = "/home/mbui/ModelOutput/IW/mercator/"
dirwoce = "/home/mbui/ModelOutput/IW/stratification/"
dirfig  = "/home/mbui/ModelOutput/figs/"

ds = NCDataset(string(dirin,"MERC_N2_zonalmean_Atl_40W.nc"),"r")
depth_mid = ds["depth_mid"][:]
lat       = ds["latitude"][:]
N2        = coalesce.(ds["N2_zonalmean"][:,:], NaN)   # (depth_mid, latitude)
ncolumns  = ds["ncolumns"][:]
close(ds)

nlat = length(lat)

# shared x-axis range with the WOCE zonal-mean plot, upper 400 m only
dsw = NCDataset(string(dirwoce,"WOCE_N2_zonalmean_Atl_40W.nc"),"r")
depth_mid_w = dsw["depth_mid"][:]
N2_woce     = coalesce.(dsw["N2_zonalmean"][:,:], NaN)
close(dsw)

im = findall(depth_mid   .<= 400)
iw = findall(depth_mid_w .<= 400)
xmax_shared = 1.05*max(maximum(filter(!isnan, N2[im,:])), maximum(filter(!isnan, N2_woce[iw,:])))

groups = [1:5, 6:10, 11:nlat]   # 0-15N, 20-40N, 45-60N
grouptitles = ["0-15°N", "20-40°N", "45-60°N"]

fig = Figure(size = (1500, 800))

for (g,rng) in enumerate(groups)
    ax = Axis(fig[1,g], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = grouptitles[g])
    ng = length(rng)
    cols = get(ColorSchemes.viridis, range(0,1,length=ng))
    for (i,k) in enumerate(rng)
        ivalid = findall(!isnan, N2[:,k])
        lines!(ax, N2[ivalid,k], -depth_mid[ivalid], color = cols[i],
               label = string(lat[k],"°N (n=",ncolumns[k],")"))
    end
    xlims!(ax, 0, xmax_shared)
    ylims!(ax, -400, 0)
    axislegend(ax, position = :rb, framevisible = false, labelsize = 10)
end

Label(fig[0,1:3], "Mercator N² along 40°W: open-Atlantic zonal mean, sea floor > 2000 m columns only, upper 400 m", fontsize = 15)

save(string(dirfig,"N2_zonalmean_Atl_40W_3panels_upper400m.png"), fig)
println("saved: ", string(dirfig,"N2_zonalmean_Atl_40W_3panels_upper400m.png"))
