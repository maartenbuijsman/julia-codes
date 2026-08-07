# plot_WOCE_TS_N2_40W.jl
# MCB/Claude, USM, 2026-8-4
# Plot WOCE annual-mean T, S, N2 profiles extracted along 40 W (extract_WOCE_TS_N2_40W.jl)

using NCDatasets
using CairoMakie
using ColorSchemes

dirin  = "/home/mbui/ModelOutput/IW/stratification/"
dirfig = "/home/mbui/ModelOutput/figs/"

# ---------------------------------------------------------------------------
# load data

ds = NCDataset(string(dirin,"WOCE_TS_profiles_40W.nc"),"r")
depth = ds["depth"][:]
lat   = ds["latitude"][:]
T     = ds["temp"][:,:]   # (depth, latitude)
S     = ds["salt"][:,:]
close(ds)

ds2 = NCDataset(string(dirin,"WOCE_N2_profiles_40W.nc"),"r")
depth_mid = ds2["depth_mid"][:]
N2        = ds2["N2"][:,:] # (depth_mid, latitude)
close(ds2)

nlat = length(lat)
cols = get(ColorSchemes.viridis, range(0,1,length=nlat))

# ---------------------------------------------------------------------------
# plot: full depth range of the WOCE data (~0-2000m, deeper at 50-60N)

fig = Figure(size = (1100, 800))

axT  = Axis(fig[1,1], xlabel = "Temperature (°C)", ylabel = "Depth (m)", title = "T (in-situ)")
axS  = Axis(fig[1,2], xlabel = "Salinity (PSU)",    ylabel = "Depth (m)", title = "S")
axN2 = Axis(fig[1,3], xlabel = "N² (s⁻²)",          ylabel = "Depth (m)", title = "N²")

for k in 1:nlat
    lines!(axT,  T[:,k],  -depth,     color = cols[k], label = string(lat[k],"°N"))
    lines!(axS,  S[:,k],  -depth,     color = cols[k], label = string(lat[k],"°N"))
    lines!(axN2, N2[:,k], -depth_mid, color = cols[k], label = string(lat[k],"°N"))
end

Legend(fig[1,4], axT, "Latitude", framevisible = false)

Label(fig[0,1:3], "WOCE (Gregg Jacobs / WOD) T, S, N² profiles at 40°W", fontsize = 18)

save(string(dirfig,"WOCE_TS_N2_profiles_40W.png"), fig)
println("saved: ", string(dirfig,"WOCE_TS_N2_profiles_40W.png"))

# ---------------------------------------------------------------------------
# zoom on upper 300 m, with markers to show the native WOCE depth levels
# (coarse near-surface spacing is a likely source of the noisy-looking N2)

fig2 = Figure(size = (1100, 800))

axT2  = Axis(fig2[1,1], xlabel = "Temperature (°C)", ylabel = "Depth (m)", title = "T (in-situ), upper 300 m")
axS2  = Axis(fig2[1,2], xlabel = "Salinity (PSU)",    ylabel = "Depth (m)", title = "S, upper 300 m")
axN22 = Axis(fig2[1,3], xlabel = "N² (s⁻²)",          ylabel = "Depth (m)", title = "N², upper 300 m")

for k in 1:nlat
    scatterlines!(axT2,  T[:,k],  -depth,     color = cols[k], markersize = 6, label = string(lat[k],"°N"))
    scatterlines!(axS2,  S[:,k],  -depth,     color = cols[k], markersize = 6, label = string(lat[k],"°N"))
    scatterlines!(axN22, N2[:,k], -depth_mid, color = cols[k], markersize = 6, label = string(lat[k],"°N"))
end

ylims!(axT2,  -300, 0)
ylims!(axS2,  -300, 0)
ylims!(axN22, -300, 0)

Legend(fig2[1,4], axT2, "Latitude", framevisible = false)

Label(fig2[0,1:3], "WOCE T, S, N² profiles at 40°W, upper 300 m (markers = native WOCE depth levels)", fontsize = 16)

save(string(dirfig,"WOCE_TS_N2_profiles_40W_upper300m.png"), fig2)
println("saved: ", string(dirfig,"WOCE_TS_N2_profiles_40W_upper300m.png"))

println("shallowest 10 WOCE depth levels (m): ", depth[1:10])

# ---------------------------------------------------------------------------
# zoom on upper 50 m

fig3 = Figure(size = (1100, 800))

axT3  = Axis(fig3[1,1], xlabel = "Temperature (°C)", ylabel = "Depth (m)", title = "T (in-situ), upper 50 m")
axS3  = Axis(fig3[1,2], xlabel = "Salinity (PSU)",    ylabel = "Depth (m)", title = "S, upper 50 m")
axN23 = Axis(fig3[1,3], xlabel = "N² (s⁻²)",          ylabel = "Depth (m)", title = "N², upper 50 m")

for k in 1:nlat
    scatterlines!(axT3,  T[:,k],  -depth,     color = cols[k], markersize = 8, label = string(lat[k],"°N"))
    scatterlines!(axS3,  S[:,k],  -depth,     color = cols[k], markersize = 8, label = string(lat[k],"°N"))
    scatterlines!(axN23, N2[:,k], -depth_mid, color = cols[k], markersize = 8, label = string(lat[k],"°N"))
end

ylims!(axT3,  -50, 0)
ylims!(axS3,  -50, 0)
ylims!(axN23, -50, 0)

Legend(fig3[1,4], axT3, "Latitude", framevisible = false)

Label(fig3[0,1:3], "WOCE T, S, N² profiles at 40°W, upper 50 m (markers = native WOCE depth levels)", fontsize = 16)

save(string(dirfig,"WOCE_TS_N2_profiles_40W_upper50m.png"), fig3)
println("saved: ", string(dirfig,"WOCE_TS_N2_profiles_40W_upper50m.png"))
