# plot_TS_N2_40W.jl
# MCB/Claude, USM, 2026-8-2
# Plot T, S, and N2 profiles extracted along 40 W (extract_TS_N2_40W.jl)

using NCDatasets
using CairoMakie
using ColorSchemes

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
dirfig = "/home/mbui/ModelOutput/figs/"

# ---------------------------------------------------------------------------
# load data

ds = NCDataset(string(dirin,"TS_profiles_40W.nc"),"r")
depth = ds["depth"][:]
lat   = ds["latitude"][:]
T     = ds["thetao"][:,:]   # (depth, latitude)
S     = ds["so"][:,:]
close(ds)

ds2 = NCDataset(string(dirin,"N2_profiles_40W.nc"),"r")
depth_mid = ds2["depth_mid"][:]
N2        = ds2["N2"][:,:] # (depth_mid, latitude)
close(ds2)

nlat = length(lat)
cols = get(ColorSchemes.viridis, range(0,1,length=nlat))

# ---------------------------------------------------------------------------
# plot

fig = Figure(size = (1100, 800))

axT  = Axis(fig[1,1], xlabel = "Temperature (°C)", ylabel = "Depth (m)", title = "T")
axS  = Axis(fig[1,2], xlabel = "Salinity (PSU)",    ylabel = "Depth (m)", title = "S")
axN2 = Axis(fig[1,3], xlabel = "N² (s⁻²)",          ylabel = "Depth (m)", title = "N²")

for k in 1:nlat
    lines!(axT,  T[:,k],  -depth,     color = cols[k], label = string(lat[k],"°N"))
    lines!(axS,  S[:,k],  -depth,     color = cols[k], label = string(lat[k],"°N"))
    lines!(axN2, N2[:,k], -depth_mid, color = cols[k], label = string(lat[k],"°N"))
end

Legend(fig[1,4], axT, "Latitude", framevisible = false)

Label(fig[0,1:3], "T, S, N² profiles at 40°W (GLORYS12V1 1993-2016 mean)", fontsize = 18)

save(string(dirfig,"TS_N2_profiles_40W_full.png"), fig)
println("saved: ", string(dirfig,"TS_N2_profiles_40W_full.png"))

# ---------------------------------------------------------------------------
# zoom on upper 500 m, where most of the vertical structure lives

fig2 = Figure(size = (1100, 800))

axT2  = Axis(fig2[1,1], xlabel = "Temperature (°C)", ylabel = "Depth (m)", title = "T (upper 500 m)")
axS2  = Axis(fig2[1,2], xlabel = "Salinity (PSU)",    ylabel = "Depth (m)", title = "S (upper 500 m)")
axN22 = Axis(fig2[1,3], xlabel = "N² (s⁻²)",          ylabel = "Depth (m)", title = "N² (upper 500 m)")

for k in 1:nlat
    lines!(axT2,  T[:,k],  -depth,     color = cols[k], label = string(lat[k],"°N"))
    lines!(axS2,  S[:,k],  -depth,     color = cols[k], label = string(lat[k],"°N"))
    lines!(axN22, N2[:,k], -depth_mid, color = cols[k], label = string(lat[k],"°N"))
end

ylims!(axT2,  -500, 0)
ylims!(axS2,  -500, 0)
ylims!(axN22, -500, 0)

Legend(fig2[1,4], axT2, "Latitude", framevisible = false)

Label(fig2[0,1:3], "T, S, N² profiles at 40°W, upper 500 m (GLORYS12V1 1993-2016 mean)", fontsize = 18)

save(string(dirfig,"TS_N2_profiles_40W_upper500m.png"), fig2)
println("saved: ", string(dirfig,"TS_N2_profiles_40W_upper500m.png"))

# ---------------------------------------------------------------------------
# depth grid spacing (same fixed z-levels at every latitude)

dz = diff(depth)

fig3 = Figure(size = (900, 500))

ax1 = Axis(fig3[1,1], xlabel = "Δz (m)", ylabel = "Depth (m)", title = "Full water column")
scatterlines!(ax1, dz, -depth_mid)

ax2 = Axis(fig3[1,2], xlabel = "Δz (m)", ylabel = "Depth (m)", title = "Upper 500 m")
scatterlines!(ax2, dz, -depth_mid)
ylims!(ax2, -500, 0)

Label(fig3[0,1:2], "Mercator/GLORYS12V1 depth grid spacing (40°W)", fontsize = 18)

save(string(dirfig,"depth_grid_spacing_40W.png"), fig3)
println("saved: ", string(dirfig,"depth_grid_spacing_40W.png"))

println("shallowest 10 levels: depth (m) = ", depth[1:10])
println("shallowest 9 spacings: dz (m)   = ", dz[1:9])
