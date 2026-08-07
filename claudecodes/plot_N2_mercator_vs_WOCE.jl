# plot_N2_mercator_vs_WOCE.jl
# MCB/Claude, USM, 2026-8-4
# Compare the Mercator/GLORYS12V1 N2 profile at 0 N, 40 W against the WOCE-based
# N2 profile (N2_amz1.jld2) used as forcing in IW_Amz_200m_2000km_bash_cuda.jl

using NCDatasets
using JLD2
using CairoMakie

dirmerc = "/home/mbui/ModelOutput/IW/mercator/"
dirwoce = "/home/mbui/ModelOutput/IW/forcingfiles/"
dirfig  = "/home/mbui/ModelOutput/figs/"

# ---------------------------------------------------------------------------
# Mercator N2 at 0 N, 40 W

ds = NCDataset(string(dirmerc,"N2_profiles_40W.nc"),"r")
depth_mid = ds["depth_mid"][:]
lat_merc  = ds["latitude"][:]
N2_merc   = ds["N2"][:,:]   # (depth_mid, latitude)
close(ds)

k0 = findfirst(==(0.0), lat_merc)
N2_merc0 = N2_merc[:,k0]

# ---------------------------------------------------------------------------
# WOCE-based N2 (amz1) used in IW_Amz_200m_2000km_bash_cuda.jl

fwoce = jldopen(string(dirwoce,"N2_amz1.jld2"),"r")
N2w    = fwoce["N2w"]
zfw    = fwoce["zfw"]
lonsel = fwoce["lonsel"]
latsel = fwoce["latsel"]
close(fwoce)

lon_woce = lonsel > 180 ? lonsel - 360 : lonsel
println("WOCE profile location: lon = ", lon_woce, " E, lat = ", latsel, " N")

# ---------------------------------------------------------------------------
# plot: N2 and N side by side

fig = Figure(size = (900, 700))

axN2 = Axis(fig[1,1], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = "N²")
lines!(axN2, N2_merc0, -depth_mid, color = :dodgerblue, label = "Mercator (0°N, 40°W)")
lines!(axN2, N2w,      zfw,        color = :firebrick,  label = string("WOCE amz1 (",round(latsel,digits=1),"°N, ",round(lon_woce,digits=1),"°W)"))
ylims!(axN2, -1500, 0)

axN = Axis(fig[1,2], xlabel = "N (s⁻¹)", ylabel = "Depth (m)", title = "N")
lines!(axN, sqrt.(max.(N2_merc0,0)), -depth_mid, color = :dodgerblue, label = "Mercator (0°N, 40°W)")
lines!(axN, sqrt.(max.(N2w,0)),      zfw,        color = :firebrick,  label = string("WOCE amz1 (",round(latsel,digits=1),"°N, ",round(lon_woce,digits=1),"°W)"))
ylims!(axN, -1500, 0)

Legend(fig[2,1:2], axN2, framevisible = false, orientation = :horizontal)

Label(fig[0,1:2], "Mercator vs. WOCE (amz1) buoyancy frequency", fontsize = 18)

save(string(dirfig,"N2_N_mercator_vs_WOCE_amz1.png"), fig)
println("saved: ", string(dirfig,"N2_N_mercator_vs_WOCE_amz1.png"))
