# plot_N2_merged_vs_amz1.jl
# MCB/Claude, USM, 2026-8-4
# Compare the new merged (WOCE+Mercator) N2 profiles at 0, 2.5, 5 N, 40 W
# against the original WOCE-only N2_amz1.jld2 profile (~7 N, 42 W) currently
# used in IW_Amz_200m_2000km_bash_cuda.jl.

using JLD2
using CairoMakie
using Printf

dirforce = "/home/mbui/ModelOutput/IW/forcingfiles/"
dirfig   = "/home/mbui/ModelOutput/figs/"

lats = [0.0, 2.5, 5.0]
cols = [:dodgerblue, :seagreen, :darkorange]

fig = Figure(size = (900, 700))

axN2 = Axis(fig[1,1], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = "N²")
axN  = Axis(fig[1,2], xlabel = "N (s⁻¹)",  ylabel = "Depth (m)", title = "N")

for (i,lt) in enumerate(lats)
    fname = @sprintf("N2_40W_lat%04.1f.jld2", lt)
    f = jldopen(string(dirforce,fname),"r")
    N2w = f["N2w"]; zfw = f["zfw"]
    close(f)

    lines!(axN2, N2w,               zfw, color = cols[i], label = string(lt,"°N, 40°W (merged)"))
    lines!(axN,  sqrt.(max.(N2w,0)), zfw, color = cols[i], label = string(lt,"°N, 40°W (merged)"))
end

famz = jldopen(string(dirforce,"N2_amz1.jld2"),"r")
N2w_amz = famz["N2w"]; zfw_amz = famz["zfw"]
lonsel_amz = famz["lonsel"]; latsel_amz = famz["latsel"]
close(famz)

lon_amz = lonsel_amz > 180 ? lonsel_amz - 360 : lonsel_amz
amz_label = string(round(latsel_amz,digits=1),"°N, ",round(abs(lon_amz),digits=1),"°W (WOCE amz1, original)")

lines!(axN2, N2w_amz,               zfw_amz, color = :black, linestyle = :dash, linewidth = 2, label = amz_label)
lines!(axN,  sqrt.(max.(N2w_amz,0)), zfw_amz, color = :black, linestyle = :dash, linewidth = 2, label = amz_label)

Legend(fig[2,1:2], axN2, framevisible = false, orientation = :horizontal, nbanks = 2)

Label(fig[0,1:2], "Merged N² / N (0, 2.5, 5°N, 40°W) vs. original N2_amz1", fontsize = 18)

save(string(dirfig,"N2_N_merged_vs_amz1.png"), fig)
println("saved: ", string(dirfig,"N2_N_merged_vs_amz1.png"))

# ---------------------------------------------------------------------------
# zoom on upper 400 m

fig2 = Figure(size = (900, 700))

axN2z = Axis(fig2[1,1], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = "N² (upper 400 m)")
axNz  = Axis(fig2[1,2], xlabel = "N (s⁻¹)",  ylabel = "Depth (m)", title = "N (upper 400 m)")

for (i,lt) in enumerate(lats)
    fname = @sprintf("N2_40W_lat%04.1f.jld2", lt)
    f = jldopen(string(dirforce,fname),"r")
    N2w = f["N2w"]; zfw = f["zfw"]
    close(f)

    lines!(axN2z, N2w,               zfw, color = cols[i], label = string(lt,"°N, 40°W (merged)"))
    lines!(axNz,  sqrt.(max.(N2w,0)), zfw, color = cols[i], label = string(lt,"°N, 40°W (merged)"))
end

lines!(axN2z, N2w_amz,               zfw_amz, color = :black, linestyle = :dash, linewidth = 2, label = amz_label)
lines!(axNz,  sqrt.(max.(N2w_amz,0)), zfw_amz, color = :black, linestyle = :dash, linewidth = 2, label = amz_label)

ylims!(axN2z, -400, 0)
ylims!(axNz,  -400, 0)

Legend(fig2[2,1:2], axN2z, framevisible = false, orientation = :horizontal, nbanks = 2)

Label(fig2[0,1:2], "Merged N² / N (0, 2.5, 5°N, 40°W) vs. original N2_amz1, upper 400 m", fontsize = 18)

save(string(dirfig,"N2_N_merged_vs_amz1_upper400m.png"), fig2)
println("saved: ", string(dirfig,"N2_N_merged_vs_amz1_upper400m.png"))

# ---------------------------------------------------------------------------
# zoom on upper 50 m

fig3 = Figure(size = (900, 700))

axN2z3 = Axis(fig3[1,1], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = "N² (upper 50 m)")
axNz3  = Axis(fig3[1,2], xlabel = "N (s⁻¹)",  ylabel = "Depth (m)", title = "N (upper 50 m)")

for (i,lt) in enumerate(lats)
    fname = @sprintf("N2_40W_lat%04.1f.jld2", lt)
    f = jldopen(string(dirforce,fname),"r")
    N2w = f["N2w"]; zfw = f["zfw"]
    close(f)

    scatterlines!(axN2z3, N2w,               zfw, color = cols[i], label = string(lt,"°N, 40°W (merged)"))
    scatterlines!(axNz3,  sqrt.(max.(N2w,0)), zfw, color = cols[i], label = string(lt,"°N, 40°W (merged)"))
end

scatterlines!(axN2z3, N2w_amz,               zfw_amz, color = :black, linestyle = :dash, linewidth = 2, label = amz_label)
scatterlines!(axNz3,  sqrt.(max.(N2w_amz,0)), zfw_amz, color = :black, linestyle = :dash, linewidth = 2, label = amz_label)

ylims!(axN2z3, -50, 0)
ylims!(axNz3,  -50, 0)

Legend(fig3[2,1:2], axN2z3, framevisible = false, orientation = :horizontal, nbanks = 2)

Label(fig3[0,1:2], "Merged N² / N (0, 2.5, 5°N, 40°W) vs. original N2_amz1, upper 50 m", fontsize = 18)

save(string(dirfig,"N2_N_merged_vs_amz1_upper50m.png"), fig3)
println("saved: ", string(dirfig,"N2_N_merged_vs_amz1_upper50m.png"))
