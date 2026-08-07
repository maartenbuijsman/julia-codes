# plot_N2_monthly_annualmean_40W.jl
# MCB/Claude, USM, 2026-8-4
# Plot the annual mean of the 12 monthly Mercator N2 profiles (averaging N2
# itself, not T/S first) along 40 W, upper 500 m, 3 latitude-group subplots.

using NCDatasets
using CairoMakie
using ColorSchemes

dirin    = "/home/mbui/ModelOutput/IW/mercator/"
dirwoce  = "/home/mbui/ModelOutput/IW/stratification/"
dirfig   = "/home/mbui/ModelOutput/figs/"

ds = NCDataset(string(dirin,"MERC_N2_monthly_40W.nc"),"r")
depth_mid = ds["depth_mid"][:]
lat       = ds["latitude"][:]
N2_annual = coalesce.(ds["N2_annual_mean"][:,:], NaN)   # (depth_mid, latitude)
close(ds)

nlat = length(lat)

# shared x-axis range with the WOCE annual-mean-of-monthly N2 plot, for
# visual comparability (upper 500 m only, WOCE side clamped N2<0 -> zeroval
# to match plot_WOCE_N2_monthly_annualmean_40W.jl's default)
dsw = NCDataset(string(dirwoce,"WOCE_N2_monthly_40W.nc"),"r")
depth_mid_w = dsw["depth_mid"][:]
N2_woce     = coalesce.(dsw["N2_annual_mean"][:,:], NaN)
close(dsw)
N2_woce[.!isnan.(N2_woce) .& (N2_woce .< 0)] .= 1e-12

im = findall(depth_mid   .<= 500)
iw = findall(depth_mid_w .<= 500)
xmax_shared = 1.05*max(maximum(filter(!isnan, N2_annual[im,:])), maximum(filter(!isnan, N2_woce[iw,:])))

groups = [1:5, 6:10, 11:nlat]   # 0-15N, 20-40N, 45-60N
grouptitles = ["0-15°N", "20-40°N", "45-60°N"]

fig = Figure(size = (1500, 800))

for (g,rng) in enumerate(groups)
    ax = Axis(fig[1,g], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = grouptitles[g])
    ng = length(rng)
    cols = get(ColorSchemes.viridis, range(0,1,length=ng))
    for (i,k) in enumerate(rng)
        lines!(ax, N2_annual[:,k], -depth_mid, color = cols[i], label = string(lat[k],"°N"))
    end
    xlims!(ax, 0, xmax_shared)
    ylims!(ax, -500, 0)
    axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
end

Label(fig[0,1:3], "Mercator N² along 40°W: annual mean of the 12 monthly N2 profiles, upper 500 m", fontsize = 16)

save(string(dirfig,"N2_monthly_annualmean_40W_upper500m.png"), fig)
println("saved: ", string(dirfig,"N2_monthly_annualmean_40W_upper500m.png"))
