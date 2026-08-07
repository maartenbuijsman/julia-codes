# plot_N2_zonalmean_Atl_offshelf_monthly.jl
# MCB/Claude, USM, 2026-8-5
# Plot the Range-2 (coastline-anchored, off-shelf >500m), monthly-then-
# annual-then-zonal-mean Mercator N2 profiles, 3 latitude-group subplots,
# full depth + upper 400 m.

using NCDatasets
using CairoMakie
using ColorSchemes

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
dirfig = "/home/mbui/ModelOutput/figs/"

ds = NCDataset(string(dirin,"MERC_N2_zonalmean_Atl_offshelf_monthly.nc"),"r")
depth_mid = ds["depth_mid"][:]
lat       = ds["latitude"][:]
N2        = coalesce.(ds["N2_zonalmean"][:,:], NaN)   # (depth_mid, latitude)
ncolumns  = ds["ncolumns"][:]
close(ds)

nlat = length(lat)

groups = [1:5, 6:10, 11:nlat]   # 0-15N, 20-40N, 45-60N
grouptitles = ["0-15°N", "20-40°N", "45-60°N"]

for (ylim, suffix, ttl) in [((-5500,0), "", "full depth"), ((-400,0), "_upper400m", "upper 400 m")]
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
        ylims!(ax, ylim...)
        xlims!(ax,[0 0.00052])
        
        axislegend(ax, position = :rb, framevisible = false, labelsize = 10)

        # MBUI added
        pathout = "/home/mbui/ModelOutput/IW/"
        dirin      = string(pathout, "forcingfiles/");
        fnamegrid  = "N2_amz1.jld2";
        path_fname = string(dirin, fnamegrid);
        @load path_fname N2w zfw

        #plot N2
        lines!(ax, N2w, zfw,color = :black, linestyle = :dash)    

    end
    Label(fig[0,1:3], "Mercator N²: Range-2 (coastline-anchored, off-shelf) monthly->annual->zonal mean ($ttl)", fontsize = 14)
    save(string(dirfig,"N2_zonalmean_Atl_offshelf_monthly_3panels$suffix.png"), fig)
    println("saved: ", string(dirfig,"N2_zonalmean_Atl_offshelf_monthly_3panels$suffix.png"))
    display(fig)

end


