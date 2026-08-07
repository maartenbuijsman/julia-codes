# plot_WOCE_N2_monthly_annualmean_40W.jl
# MCB/Claude, USM, 2026-8-4
# Plot the annual mean of the 12 monthly WOCE N2 profiles along 40 W, upper
# 500 m, 3 subplots. Set clamp_negative = true to replace N2<0 with zeroval
# (matching the AMZ/ATL_stratification_profile.jl convention) instead of
# leaving the raw (possibly negative) annual mean as-is.

using NCDatasets
using CairoMakie
using ColorSchemes

dirin   = "/home/mbui/ModelOutput/IW/stratification/"
dirmerc = "/home/mbui/ModelOutput/IW/mercator/"
dirfig  = "/home/mbui/ModelOutput/figs/"

clamp_negative = true
zeroval = 1e-12

ds = NCDataset(string(dirin,"WOCE_N2_monthly_40W.nc"),"r")
depth_mid = ds["depth_mid"][:]
lat       = ds["latitude"][:]
N2_annual = coalesce.(ds["N2_annual_mean"][:,:], NaN)   # (depth_mid, latitude)
close(ds)

nlat = length(lat)

if clamp_negative
    nneg = count(x -> !isnan(x) && x < 0, N2_annual)
    println("clamping ", nneg, " negative annual-mean N2 point(s) to zeroval = ", zeroval)
    N2_annual[.!isnan.(N2_annual) .& (N2_annual .< 0)] .= zeroval
end

# shared x-axis range with the Mercator annual-mean-of-monthly N2 plot, for
# visual comparability (upper 500 m only)
dsm = NCDataset(string(dirmerc,"MERC_N2_monthly_40W.nc"),"r")
depth_mid_m = dsm["depth_mid"][:]
N2_merc     = coalesce.(dsm["N2_annual_mean"][:,:], NaN)
close(dsm)

iw = findall(depth_mid   .<= 500)
im = findall(depth_mid_m .<= 500)
xmax_shared = 1.05*max(maximum(filter(!isnan, N2_annual[iw,:])), maximum(filter(!isnan, N2_merc[im,:])))

groups = [1:5, 6:10, 11:nlat]   # 0-15N, 20-40N, 45-60N
grouptitles = ["0-15°N", "20-40°N", "45-60°N"]

fig = Figure(size = (1500, 800))

for (g,rng) in enumerate(groups)
    ax = Axis(fig[1,g], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = grouptitles[g])
    ng = length(rng)
    cols = get(ColorSchemes.viridis, range(0,1,length=ng))
    for (i,k) in enumerate(rng)
        ivalid = findall(!isnan, N2_annual[:,k])
        lines!(ax, N2_annual[ivalid,k], -depth_mid[ivalid], color = cols[i], label = string(lat[k],"°N"))
    end
    vlines!(ax, 0.0, color = :gray, linestyle = :dot)
    xlims!(ax, 0, xmax_shared)
    ylims!(ax, -500, 0)
    axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
end

titlesuffix = clamp_negative ? "N2<0 -> zeroval" : "N2<0 not clamped"
fname = clamp_negative ? "WOCE_N2_monthly_annualmean_40W_upper500m_zeroval.png" : "WOCE_N2_monthly_annualmean_40W_upper500m.png"

Label(fig[0,1:3], string("WOCE N² along 40°W: annual mean of the 12 monthly N2 profiles, upper 500 m (",titlesuffix,")"), fontsize = 15)

save(string(dirfig,fname), fig)
println("saved: ", string(dirfig,fname))
