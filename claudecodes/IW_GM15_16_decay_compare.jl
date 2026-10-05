#= IW_GM15_16_decay_compare.jl
Maarten Buijsman, USM DMS, 2026-10-3

Reference figure: decay of the GM-only (no tide) runs of series 15
(15.1-12, ~3x GM81 at t = 0 from the earlier amplitude formula) and series 16
(16.1-12, GM81 initial condition rescaled so the day 10-20 mean is 1x GM81),
lat 0-45 N, in one figure.
  (a) E(t) = depth-integrated, zonal-mean (x = 100-1800 km) KE + APE [kJ/m2]
  (b) E(t) / E_GM81,  E_GM81 = rho0 b^2 N0 E0 int N dz  (Munk 1981)
Series 16 solid, series 15 dashed; colour = latitude; days 10-20 (the analysis
window) shaded. Reads the caches written by IW_GM15_decay_vs_GM76.jl and
IW_GM16_decay_vs_GM81.jl (same diagnostic, same reference; the series-15
script called it E_GM76).
=#

using Printf, CairoMakie, Statistics, JLD2

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0     = "/home/mbui/ModelOutput/"
dirout   = string(pth0, "diagout/")
dirfig   = string(pth0, "figs/")
include(string(pathname, "include_functions.jl"))   # savefig300

figflag = 1

c15 = load(string(dirout, "GM15_decay_vs_GM76.jld2"))
c16 = load(string(dirout, "GM16_decay_vs_GM81.jld2"))
LATS = c16["LATS"]
@assert LATS == c15["LATS"]
NL = length(LATS)

E(c, i)  = (c["KEs"][i] .+ c["APEs"][i]) ./ 1e3          # kJ/m2
Er(c, i) = (c["KEs"][i] .+ c["APEs"][i]) ./ c["EGMs"][i]  # / E_GM81

cm_to_pt = 72/2.54
fig = Figure(size = (18cm_to_pt, 9.5cm_to_pt), fontsize = 10)
cmap = cgrad(:viridis, NL, categorical = true)
axA = Axis(fig[1, 1], xlabel = "time [days]", ylabel = "E [kJ/m²]", title = "(a) GM only: KE + APE",
           titlesize = 10, xticks = 0:5:20, xtickalign = 1, ytickalign = 1)
axB = Axis(fig[1, 2], xlabel = "time [days]", ylabel = "E / E_GM81", title = "(b) relative to GM81",
           titlesize = 10, xticks = 0:5:20, xtickalign = 1, ytickalign = 1)
for ax in (axA, axB)
    vspan!(ax, 10, 20, color = (:gray, 0.15))
end
hlines!(axB, [1.0], color = :black, linestyle = :dot, linewidth = 1)
for i in 1:NL
    lines!(axA, c15["tt"][i], E(c15, i),  color = cmap[i], linewidth = 1.2, linestyle = :dash)
    lines!(axA, c16["tt"][i], E(c16, i),  color = cmap[i], linewidth = 1.5)
    lines!(axB, c15["tt"][i], Er(c15, i), color = cmap[i], linewidth = 1.2, linestyle = :dash)
    lines!(axB, c16["tt"][i], Er(c16, i), color = cmap[i], linewidth = 1.5)
end
ylims!(axA, 0, nothing); ylims!(axB, 0, nothing)

# latitude as a categorical colour bar, series as a line-style legend below:
# both outside the axes, so neither can cover a curve
Colorbar(fig[1, 3], colormap = cmap, limits = (0.5, NL + 0.5), label = "latitude [°N]",
         ticks = (1:NL, [@sprintf("%g", l) for l in LATS]), ticklabelsize = 8, width = 8)
Legend(fig[2, 1:2], [LineElement(color = :black, linewidth = 1.5),
                     LineElement(color = :black, linewidth = 1.2, linestyle = :dash),
                     PolyElement(color = (:gray, 0.15))],
       ["series 16 (GM81 IC, 1× GM81 over days 10-20)", "series 15 (earlier IC, ≈3× GM81 at t = 0)",
        "analysis window"],
       orientation = :horizontal, framevisible = false, labelsize = 8, patchsize = (18, 6), colgap = 12)
rowgap!(fig.layout, 4)
display(fig)
if figflag == 1
    fout = string(dirfig, "GM_decay_15vs16.png")
    savefig300(fout, fig)
    println("saved ", fout)
end

## numbers ------------------------------------------------------------------
@printf("%5s | %8s %8s %8s | %8s %8s %8s\n", "lat", "15: t=0", "10-20", "t=20", "16: t=0", "10-20", "t=20")
for i in 1:NL
    m(c) = mean(Er(c, i)[10 .<= c["tt"][i] .<= 20])
    @printf("%5.1f | %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f\n", LATS[i],
            Er(c15, i)[1], m(c15), Er(c15, i)[end], Er(c16, i)[1], m(c16), Er(c16, i)[end])
end
