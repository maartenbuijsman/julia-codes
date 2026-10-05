#= IW_Pomega_2x2_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: segment-averaged P(ω) at four latitudes as a 2x2 panel,
18 x 18 cm, fontsize 10, one legend.

  (a)  2.5°N     (b) 15°N
  (c) 28.8°N     (d) 40°N

Row-major with latitude increasing, so the reader crosses the M2 critical
latitude (28.9°N, where ω/2 = f) between (c) and (d): PSI is permitted in
(a)-(c) and forbidden in (d).

Three curves per panel, all at the same latitude and the same 25 kW/m forcing:
  black  11.x   tide, no GM
  red    GMSER.x   GM + tide       (GMSER = 16: GM81 IC; 15: redistribution IC)
  blue   GMSER.i   GM only, Flux = 0  (no tide)

Only the ω⁻² (GM continuum) reference is drawn -- the ω⁻³ harmonic-envelope
reference that IW_Pomega_segments_compare_all13.jl also plots is dropped here.
It is anchored on the GM-only curve at 1 cpd in every panel, so its LEVEL is
per-panel but its SLOPE is the thing being compared.

Spectra are read from the cache that IW_Pomega_segments_compare_all13.jl
writes, so this script runs in seconds and never re-reads a NetCDF file. Run
that script first (with the same GMSER/TUKEYCF/CELL_STEP/LINFIT) if the cache
is missing entries.
=#

using Printf, CairoMakie, Statistics, JLD2

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1

# must match the batch that wrote the cache
const GMSER     = 16
const TUKEYCF   = 0.0
const CELL_STEP = 20
const LINFIT    = true
const CACHE = string(pth0, "diagout/",
    @sprintf("Pomega_segcache_GM%d_tukey%g_step%d_lin%d.jld2",
             GMSER, TUKEYCF, CELL_STEP, LINFIT))

# 1: 100-600, 2: 600-1100, 3: 1100-1600 km. Overridable: `julia this.jl 2`.
# All three are already in the cache, so switching is a free re-plot.
const SEG = isempty(ARGS) ? 3 : parse(Int, ARGS[1])
const SEGLAB = Dict(1 => "100-600 km", 2 => "600-1100 km", 3 => "1100-1600 km")

# LAT13 indices of the four latitudes: 2.5, 15, 28.8, 40
const IDX = [2, 5, 8, 11]
const PANEL = ["(a)", "(b)", "(c)", "(d)"]

# axes
const XLO, XHI = 0.3, 48.0
const YLO, YHI = 1e-9, 1e2
# anchor the ω⁻² reference in the CONTINUUM, well above the inertial peak.
# Anchoring at 1 cpd (as the per-latitude script does) lands on the near-inertial
# peak itself at 28.8 and 40°N, which lifts the whole reference line a decade
# above the continuum it is supposed to be compared against.
const ANCHOR = 4.0                # cpd
const XT = ([0.5,1,2,4,8,12,24,48], ["0.5","1","2","4","8","12","24","48"])
const YT = ([1e-8,1e-6,1e-4,1e-2,1e0], ["10⁻⁸","10⁻⁶","10⁻⁴","10⁻²","10⁰"])

# white backing for the in-panel label, as a fraction of the LOG axis ranges
const BOXX0, BOXX1 = 0.015, 0.20
const BOXY0, BOXY1 = 0.875, 0.985

col_nogm, col_gmt, col_gm, col_ref = :black, :red, :dodgerblue, :orange

BYRUN = load(CACHE, "BYRUN")
println("cache ", CACHE, ": ", length(BYRUN), " runs")

"""(freq, PKE) for one run in segment SEG, or an error naming what is missing."""
function grab(mainnm, runnm)
    key = (mainnm, runnm)
    haskey(BYRUN, key) || error("run $mainnm.$runnm is not in the cache -- run ",
        "IW_Pomega_segments_compare_all13.jl with GMSER=$GMSER first")
    return BYRUN[key][SEG]
end

## --- figure: 18 x 18 cm, fontsize 10 ------------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size=(18*cm_to_pt, 18*cm_to_pt), fontsize=10)

for (n, i) in enumerate(IDX)
    r, c = fldmod1(n, 2)
    LAT  = LAT13[i]
    ax = Axis(fig[r, c], xscale=log10, yscale=log10, xticks=XT, yticks=YT,
        xlabel = r == 2 ? "frequency [cpd]"      : "",
        ylabel = c == 1 ? "power [m² s⁻² day]"   : "",
        xticklabelsvisible = r == 2, yticklabelsvisible = c == 1)

    curves = [(11,    26+i, col_nogm, "tide, no GM"),
              (GMSER, 26+i, col_gmt,  "GM + tide"),
              (GMSER, i,    col_gm,   "GM only, no tide")]
    for (mainnm, runnm, col, lbl) in curves
        freq, PKE = grab(mainnm, runnm)
        ip = findall((freq .>= XLO) .& (freq .<= XHI))
        lines!(ax, freq[ip], PKE[ip], color=col, linewidth=1.4,
            label = n == 4 ? lbl : nothing)
    end

    # ω⁻² reference, anchored on the GM-only curve in the continuum
    fg, Pg = grab(GMSER, i)
    ipg = findall((fg .>= XLO) .& (fg .<= XHI))
    ia  = argmin(abs.(fg[ipg] .- ANCHOR))
    lines!(ax, fg[ipg], Pg[ipg][ia] .* (fg[ipg] ./ fg[ipg][ia]).^(-2),
        color=col_ref, linestyle=:dash, linewidth=1.5,
        label = n == 4 ? "ω⁻² reference" : nothing)

    # f, and the lower tide x GM sideband at M2 - f. The secondary peak below
    # M2 in panel (b) is that sideband: measured 1.424 cpd against M2 - f =
    # 1.413 at 15°N, inside one frequency bin (df = 0.102 cpd), and it carries
    # 10-23x the GM-only power there, so it exists only when the tide does.
    # At 28.8°N M2 - f collapses onto f itself, which is the critical latitude.
    fcor = coriolis(LAT)/(2π)*86400
    vlines!(ax, [fcor], color=:gray45, linestyle=:dash, linewidth=1.2,
        label = n == 4 ? "f" : nothing)
    # purple dash-dot, deliberately unlike the grey dashed f: at 28.8°N the two
    # lines coincide exactly and would otherwise be indistinguishable
    vlines!(ax, [24/(12+25.2/60) - fcor], color=:purple, linestyle=:dashdot,
        linewidth=1.2, label = n == 4 ? "M2 − f" : nothing)

    xlims!(ax, XLO, XHI); ylims!(ax, YLO, YHI)

    # label block: white backing drawn as a poly, since `text!` has no
    # background attribute. Coordinates are in LOG space because both axes are
    # log-scaled, then exponentiated back.
    lx(t) = 10^(log10(XLO) + t*(log10(XHI)-log10(XLO)))
    ly(t) = 10^(log10(YLO) + t*(log10(YHI)-log10(YLO)))
    poly!(ax, Point2f[(lx(BOXX0),ly(BOXY0)), (lx(BOXX1),ly(BOXY0)),
                      (lx(BOXX1),ly(BOXY1)), (lx(BOXX0),ly(BOXY1))],
        color=:white, strokewidth=0)
    text!(ax, lx(BOXX0+0.012), ly(BOXY1-0.015), align=(:left,:top),
        fontsize=9, font=:bold, color=:black,
        text=@sprintf("%s %.1f°N", PANEL[n], LAT))

    # one legend for the whole figure, in (d) where there is room for it
    n == 4 && axislegend(ax, position=:rt, framevisible=false, labelsize=8,
        padding=(2,2,2,2), rowgap=0, patchsize=(16,8), patchlabelgap=4)
end

Label(fig[0, :], @sprintf("segment-averaged P(ω), %s, F = 25 kW m⁻¹", SEGLAB[SEG]),
    fontsize=10, font=:bold)

colgap!(fig.layout, 0)
rowgap!(fig.layout, 1, 0)      # between the two panel rows
rowgap!(fig.layout, 4)         # title to panels

display(fig)
if figflag == 1
    fout = string(dirfig, @sprintf("Pomega_2x2_ppr_seg%d_GM%d.png", SEG, GMSER))
    savefig300(fout, fig)
    println("saved ", fout)
end

## --- numbers --------------------------------------------------------------------
# how much energy the tide ADDS to the GM background, band by band, as the
# ratio of (GM+tide) to (GM only) -- 1.0 means the tide changed nothing there
bands = [("sub-inertial  0.3-0.9", 0.3, 0.9), ("near-f  0.9-1.5", 0.9, 1.5),
         ("M2  1.8-2.1", 1.8, 2.1), ("2-6 cpd", 2.0, 6.0), ("6-48 cpd", 6.0, 48.0)]
println("\n", "="^86)
println("(GM+tide)/(GM only) band-mean power ratio, segment ", SEGLAB[SEG])
println("="^86)
println(rpad("lat",8), join([rpad(b[1],18) for b in bands]))
for i in IDX
    f1, P1 = grab(GMSER, 26+i)          # GM + tide
    f0, P0 = grab(GMSER, i)             # GM only
    row = String[]
    for (_, lo, hi) in bands
        j = findall((f0 .>= lo) .& (f0 .<= hi))
        push!(row, rpad(@sprintf("%.2f", mean(P1[j])/mean(P0[j])), 18))
    end
    println(rpad(@sprintf("%.1f",LAT13[i]),8), join(row))
end
