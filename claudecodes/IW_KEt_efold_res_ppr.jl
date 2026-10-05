#= IW_KEt_efold_res_ppr.jl
Maarten Buijsman, USM DMS, 2026-9-20

Companion to IW_KEt_efold_ppr.jl: the same three panels, but each series is
drawn at BOTH resolutions, to show that the decay time and its flux scaling are
not artifacts of the 200 m nonhydrostatic configuration.

  solid line, filled marker   200 m NONHYDROSTATIC   mainnm 11 (15 for GM+tide)
  dashed line, open marker    4 km HYDROSTATIC       mainnm 10 (12 for GM+tide)

  (a) stratification   F = 25 kW/m:  N²(lat) 27-38, N²(2.5°N) 40-51, N²(50°N) 53-64
  (b) forcing          N²(lat):      12.5 (1-12), 25 (27-38), 50 kW/m (66-77)
  (c) GM background    F = 25 kW/m:  tide only 11/10.27-38, GM+tide 15/12.27-38

Method is identical to IW_KEt_efold_ppr.jl (bg=0: peak in 50-150 km, first
crossing of KEt_max/e, search capped at min(1800 km, XSRC + Cg1*t1)). The one
configuration-dependent step is the x-smoothing, which as in
IW_analysis_energy_2000km.jl:258 is applied only when dx < 500 m -- the 4 km
transects need none and get none.

The 4 km runs decay 5-10% SLOWER at every latitude and forcing, and their
measurable band ends one latitude equatorward of the 200 m band, but the
F^(-1/2) scaling is reproduced (q = -0.54/-0.52/-0.51 at 0/2.5/5°N against
-0.51/-0.51/-0.50 at 200 m). Two points run against that trend and are called
out in the text: 10°N of N²(50°N) at 4 km returns 8.23 d -- off the top of the
panel, an interference-lobe artifact of that stratification -- and 28.8°N of the
4 km GM run completes a genuine e-fold at 2.45 d where the 200 m GM run falls
3% short, so the PSI-band decay at the critical latitude is present at both
resolutions.
=#

using Printf, JLD2, Statistics, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirEIG    = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1

const NLAT     = 12
const LsmoothE = 1600.0
const XBOUND   = 1800e3
const XSRC     = 80e3
const T2       = 12 + 25.2/60
const TAVG1    = 10.0 + 2*T2/24
const XTICKS   = 0:10:30
# wider than the 22° of IW_KEt_efold_ppr.jl: the 4 km GM run has a genuine
# crossing at 28.8°N that would otherwise fall outside the axis
const XMAX     = 32.0
const YLO      = 1.5
const YHI      = 5.0

col_ref = :black;  col_2 = :crimson;  col_3 = :steelblue

# (panel letter, title, [(rn0, label, colour, marker, hi mainnm, lo mainnm)])
PANELS = [
 ("(a)", "stratification",
  [(27, "N²(lat)",     col_ref, :circle,    11, 10),
   (40, "N²(2.5°N)",   col_2,   :rect,      11, 10),
   (53, "N²(50°N)",    col_3,   :utriangle, 11, 10)]),
 ("(b)", "forcing",
  [( 1, "12.5 kW m⁻¹", col_2,   :rect,      11, 10),
   (27, "25 kW m⁻¹",   col_ref, :circle,    11, 10),
   (66, "50 kW m⁻¹",   col_3,   :utriangle, 11, 10)]),
 ("(c)", "GM background",
  [(27, "tide only",   col_ref, :circle,    11, 10),
   (27, "GM + tide",   col_2,   :rect,      15, 12)]),
]

function efold_bg0(xc, y, imax, xbound)
    Iend = findlast(xc .<= xbound)
    xs = xc[imax:Iend];  ys = y[imax:Iend]
    tgt = ys[1]/ℯ
    ic  = findfirst(ys .< tgt)
    (ic === nothing || ic == 1) && return xs[end], true
    return xs[ic-1] + (tgt - ys[ic-1])/(ys[ic] - ys[ic-1])*(xs[ic] - xs[ic-1]), false
end

function collect_series(mainnm, rn0)
    runnms = collect(rn0:rn0+NLAT-1)
    runs   = get_runs(mainnm, runnms)
    @load string(dirout, @sprintf("energetics_AMZexpt%02i.%02i.jld2", mainnm, runnms[1])) xc
    dxE  = xc[2] - xc[1]
    IxPk = findall(50e3 .<= xc .<= 150e3)
    lat  = [r.lat for r in runs]
    Te   = fill(NaN, NLAT);  fell = falses(NLAT)
    for (i, row) in enumerate(runs)
        @load string(dirout, @sprintf("energetics_AMZexpt%02i.%02i.jld2", mainnm, row.runnm)) KEt
        @load string(dirEIG, @sprintf("EIG_AMZexpt%02i.%02i_LAT_%04.1f.jld2",
                                      mainnm, row.runnm, row.lat)) Cgn
        cg   = Cgn[1]
        xval = min(XBOUND, XSRC + cg*TAVG1*86400)
        y    = dxE < 500 ? gaussfilt(xc, KEt, LsmoothE) : KEt
        i1   = IxPk[argmax(y[IxPk])]
        x2, fb = efold_bg0(xc, y, i1, xval)
        Te[i] = (x2 - xc[i1])/cg/86400;  fell[i] = fb
    end
    return (; lat, Te, fell, dxE, mainnm, runnms)
end

RES = [[(collect_series(s[5], s[1]), collect_series(s[6], s[1])) for s in p[3]] for p in PANELS]

## --- figure: 18 x 8 cm, fontsize 10 -----------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size=(18*cm_to_pt, 8*cm_to_pt), fontsize=10)

axs = Axis[];  offscale = String[]
for (j, (lb, ttl, series)) in enumerate(PANELS)
    ax = Axis(fig[1,j], xlabel="latitude [°]", xticks=XTICKS, yticks=YLO:0.5:YHI,
        title=string(lb, " ", ttl), titlesize=10,
        ylabel = j == 1 ? "Tₑ = Lₑ / Cg₁  [days]" : "",
        yticklabelsvisible = j == 1)
    push!(axs, ax)
    for ((hi, lo), s) in zip(RES[j], series)
        col, mk = s[3], s[4]
        # a point above the shared y-window is dropped from the curve entirely
        # (not clipped, which would draw a line to nowhere) and named below
        okh = .!hi.fell .& (hi.Te .<= YHI)
        okl = .!lo.fell .& (lo.Te .<= YHI)
        for (r, ok) in ((hi, .!hi.fell), (lo, .!lo.fell)), i in findall(ok)
            r.Te[i] > YHI && push!(offscale,
                @sprintf("%s %d.%d at %.1f°N = %.2f d", s[2], r.mainnm, r.runnms[i],
                         r.lat[i], r.Te[i]))
        end
        # NaN at the unplotted latitudes so the line BREAKS there instead of
        # bridging a gap that has no measurement in it
        gap(r, ok) = (v = copy(r.Te);  v[.!ok] .= NaN;  v)
        # 4 km first so the 200 m curve sits on top of it
        lines!(ax, lo.lat, gap(lo, okl), color=col, linewidth=1.4, linestyle=:dash)
        scatter!(ax, lo.lat[okl], lo.Te[okl], color=:white, marker=mk, markersize=7,
            strokecolor=col, strokewidth=1.2)
        lines!(ax, hi.lat, gap(hi, okh), color=col, linewidth=2)
        scatter!(ax, hi.lat[okh], hi.Te[okh], color=col, marker=mk, markersize=7)
        # legend entry: a solid line with the series marker
        scatterlines!(ax, [NaN], [NaN], color=col, marker=mk, markersize=7, linewidth=2,
            label=string(s[2], ", ", hi.mainnm, "/", lo.mainnm, ".", s[1], "-", s[1]+NLAT-1))
    end
    xlims!(ax, -1, XMAX);  ylims!(ax, YLO, YHI)
    axislegend(ax, position=:lt, framevisible=false, labelsize=8,
        backgroundcolor=:white, padding=(2,2,0,2), rowgap=0,
        patchsize=(14,8), patchlabelgap=4)
end
text!(axs[3], 0.97, 0.03, text="solid, filled: 200 m nonhydrostatic\ndashed, open: 4 km hydrostatic",
    align=(:right,:bottom), space=:relative, fontsize=8, color=:gray30)

colgap!(fig.layout, 8)
display(fig)
if figflag == 1
    fout = string(dirfig, "KEt_efold_Te_res_ppr.png")
    savefig300(fout, fig)
    println("saved ", fout)
end
isempty(offscale) || println("\nabove the plotted y-range (not drawn): ", join(offscale, "; "))

## --- numbers ----------------------------------------------------------------
for (j, (lb, ttl, series)) in enumerate(PANELS)
    println("\n", "="^86);  println(lb, " ", ttl);  println("="^86)
    for ((hi, lo), s) in zip(RES[j], series)
        @printf("\n  %s   200 m: %d.%d-%d   |   4 km: %d.%d-%d\n", s[2],
            hi.mainnm, hi.runnms[1], hi.runnms[end], lo.mainnm, lo.runnms[1], lo.runnms[end])
        println("  ", rpad("lat",7), rpad("200 m NH",13), rpad("4 km HYD",13), "ratio")
        for i in 1:NLAT
            s1 = hi.fell[i] ? "no 1/e" : @sprintf("%.2f d", hi.Te[i])
            s2 = lo.fell[i] ? "no 1/e" : @sprintf("%.2f d", lo.Te[i])
            rt = (!hi.fell[i] && !lo.fell[i]) ? @sprintf("%.2f", lo.Te[i]/hi.Te[i]) : "-"
            println("  ", rpad(@sprintf("%.1f",hi.lat[i]),7), rpad(s1,13), rpad(s2,13), rt)
        end
        ok = .!hi.fell .& .!lo.fell
        any(ok) && @printf("  -> both measurable at %d lats: %.2f d vs %.2f d, mean ratio %.2f\n",
            count(ok), mean(hi.Te[ok]), mean(lo.Te[ok]), mean(lo.Te[ok]./hi.Te[ok]))
    end
end
