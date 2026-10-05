#= IW_KEt_efold_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: e-folding decay time scale of the tidal-band internal-tide KE
against latitude, Te = Le/Cg1, for three families of simulations:

  (a) stratification   F = 25 kW/m, DX = 200 m, three N2 profiles
        11.27-38   varying N2(lat)      ("zonalmean")
        11.40-51   constant N2(2.5°N)   ("zonalmeanfixed", latfix 2.5)
        11.53-64   constant N2(50°N)    ("zonalmeanfixed", latfix 50)
  (b) forcing          varying N2, DX = 200 m, three flux amplitudes
        11.1-12    F = 12.5 kW/m
        11.27-38   F = 25   kW/m
        11.66-77   F = 50   kW/m
  (c) GM background    F = 25 kW/m, varying N2
        11.27-38   tide only
        15.27-38   GM + tide (redistribution-fix GM IC)

11.27-38 is common to all three panels and is drawn in black everywhere, so the
panels can be read against a single reference curve. lat=50 is dropped (NLAT=12)
as in IW_analysis_energy_2000km_ppr.jl and IW_coarsegr_GM_diff_ppr.jl.

METHOD (bg = 0 convention only; see IW_analysis_energy_2000km.jl:197-246, from
which first_true_min / efold_crossing_bg are copied verbatim):
Le is the distance from the KEt peak near x = 100 km to where KEt first falls to
KEt_max/e -- the literal e-fold of the energy, measured against an absolute
background of zero. The alternative convention in that file (bg = min, 1/e of
the excess over the first trough) is NOT used here: it only coincides with the
e-fold when the trough is near zero, and where the tide barely decays it
measures 1/e of a few-percent wiggle and reads as spuriously FAST decay -- at
35°N in 11.27-38 it returns 145 km on a trace that is flat to 3%.

Two limits bound where Te can be measured at all, and both are enforced:
  1. the right sponge occupies 1800-2000 km (Sp_Region_right = 200 km), and
  2. the mode-1 front has only reached XSRC + Cg1*t1 by the start of the KEt
     averaging window (t1 ~ 11.04 d: fine output starts at mid_time = 10 d and
     IW_total_energetics.jl:564 drops EXCL = 2 tidal cycles), so beyond that the
     apparent decline is the arrival transient filling in, not decay. At 50°N,
     Cg1 = 1.096 m/s puts the front at only 1125 km, and ignoring this produced
     a spurious FINITE Te = 16 d there.
The search is therefore capped at x_valid = min(1800 km, XSRC + Cg1*t1).
Latitudes where KEt never reaches KEt_max/e inside that window have NO e-folding
scale -- only a lower bound set by the window itself. Those are not plotted as
data: each series ends at its last genuine point and an arrow marks where it
leaves the measurable range. The poleward limit moves with forcing (10°N at
12.5 kW/m, 15°N at 25, 20°N at 50), which is itself part of the result.
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

const GMSER = 16   # GM series: 16 (GM81 IC, 1x GM81 over days 10-20) or 15 (earlier ~3x GM IC; untagged output names)
const NLAT     = 12          # drop the lat=50 transect
const LsmoothE = 1600.0      # Gaussian sigma [m], as in IW_analysis_energy_2000km.jl:190
const XBOUND   = 1800e3      # right sponge starts at L - Sp_Region_right
const XSRC     = 80e3        # gausW_center: where the mode is nudged in
const T2       = 12 + 25.2/60
const TAVG1    = 10.0 + 2*T2/24    # first day of the KEt averaging window [d]
const XTICKS   = 0:5:20
const XMAX     = 22.0        # no series has a measurable e-fold poleward of 20°N,
                             # so the axis stops here; the caption says why

col_ref  = :black            # 11.27-38, common to all three panels
col_2    = :crimson
col_3    = :steelblue

# (mainnm, runnms, label, colour, marker)
PANELS = [
 ("(a)", "stratification",
  [(11, collect(27:26+NLAT), "N²(lat)",        col_ref, :circle),
   (11, collect(40:39+NLAT), "N²(2.5°N)",      col_2,   :rect),
   (11, collect(53:52+NLAT), "N²(50°N)",       col_3,   :utriangle)]),
 # listed in order of increasing flux so the legend reads 12.5, 25, 50; the
 # 25 kW/m run keeps the black of the reference series used in every panel
 ("(b)", "forcing",
  [(11, collect(1:NLAT),     "12.5 kW m⁻¹",    col_2,   :rect),
   (11, collect(27:26+NLAT), "25 kW m⁻¹",      col_ref, :circle),
   (11, collect(66:65+NLAT), "50 kW m⁻¹",      col_3,   :utriangle)]),
 ("(c)", "GM background",
  [(11, collect(27:26+NLAT), "tide only",      col_ref, :circle),
   (GMSER, collect(27:26+NLAT), "GM + tide",   col_2,   :rect)]),
]

## --- measurement -------------------------------------------------------------
function first_true_min(xc, y, istart; xbound = XBOUND, relthresh = 0.02)
    runmin = y[istart]; runidx = istart
    for j in istart+1:length(xc)
        xc[j] > xbound && break
        if y[j] < runmin
            runmin = y[j]; runidx = j
        elseif y[j] > runmin * (1 + relthresh)
            break
        end
    end
    return runidx
end

# 1/e crossing of (y - ybg), searched to xbound. fellback = true means y never
# got there, so the returned x is only a lower bound pinned at the search edge.
function efold_crossing_bg(xc, y, imax, ybg; xbound = XBOUND)
    Iend = findlast(xc .<= xbound)
    xs = xc[imax:Iend]; ys = y[imax:Iend]
    dy = ys .- ybg
    target = dy[1] / ℯ
    Icross = findfirst(dy .< target)
    (Icross === nothing || Icross == 1) && return xs[end], true
    return xs[Icross-1] + (target - dy[Icross-1]) / (dy[Icross] - dy[Icross-1]) *
           (xs[Icross] - xs[Icross-1]), false
end

function collect_series(mainnm, runnms)
    runs = get_runs(mainnm, runnms)
    @load string(dirout, @sprintf("energetics_AMZexpt%02i.%02i.jld2", mainnm, runnms[1])) xc
    dxE  = xc[2] - xc[1]
    IxPk = findall(50e3 .<= xc .<= 150e3)     # search window for the near-100-km peak
    n    = length(runnms)
    lat  = [r.lat for r in runs]
    Le = zeros(n); Cg1 = zeros(n); fell = falses(n); ratio = zeros(n); xval = zeros(n)
    for (i, row) in enumerate(runs)
        @load string(dirout, @sprintf("energetics_AMZexpt%02i.%02i.jld2", mainnm, row.runnm)) KEt
        @load string(dirEIG, @sprintf("EIG_AMZexpt%02i.%02i_LAT_%04.1f.jld2",
                                      mainnm, row.runnm, row.lat)) Cgn
        Cg1[i]  = Cgn[1]
        xval[i] = min(XBOUND, XSRC + Cg1[i]*TAVG1*86400)
        KEti = dxE < 500 ? gaussfilt(xc, KEt, LsmoothE) : KEt
        imax = IxPk[argmax(KEti[IxPk])]
        imin = first_true_min(xc, KEti, imax; xbound = xval[i])
        ratio[i] = KEti[imin]/KEti[imax]
        xc0, fb  = efold_crossing_bg(xc, KEti, imax, 0.0; xbound = xval[i])
        Le[i] = xc0 - xc[imax];  fell[i] = fb
    end
    return (; lat, Le, Te = Le ./ Cg1 ./ 86400, Cg1, fell, ratio, xval, mainnm, runnms)
end

RES = [[collect_series(s[1], s[2]) for s in p[3]] for p in PANELS]

## --- figure: 18 x 6.5 cm, fontsize 10 ---------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size=(18*cm_to_pt, 8*cm_to_pt), fontsize=10)

# y-range: only the genuine points are plotted, so it is set by hand to a
# common window that holds all of them (1.66-4.49 d) with room for the legend
const YLO = 1.5
# the 12.5 kW/m curve in (b) peaks at 4.49 d and so runs behind that panel's
# legend; the legend is given an opaque backing so it stays readable
const YHI = 5.0

axs = Axis[]
for (j, (lb, ttl, series)) in enumerate(PANELS)
    ax = Axis(fig[1,j], xlabel="latitude [°]", xticks=XTICKS, yticks=YLO:0.5:YHI,
        title=string(lb, " ", ttl), titlesize=10,
        ylabel = j == 1 ? "Tₑ = Lₑ / Cg₁  [days]" : "",
        yticklabelsvisible = j == 1)
    push!(axs, ax)
    for (r, s) in zip(RES[j], series)
        ok = .!r.fell          # latitudes where KEt genuinely reaches KEt_max/e
        scatterlines!(ax, r.lat[ok], r.Te[ok], color=s[4], marker=s[5], markersize=7,
            linewidth=2,
            label=string(s[3], ", ", r.mainnm, ".", r.runnms[1], "-", r.runnms[end]))
    end
    xlims!(ax, -1, XMAX);  ylims!(ax, YLO, YHI)
    axislegend(ax, position=:lt, framevisible=false, labelsize=8,
        backgroundcolor=:white, padding=(2,2,0,2), rowgap=0,
        patchsize=(14,8), patchlabelgap=4)
end

colgap!(fig.layout, 8)
display(fig)
if figflag == 1
    fout = string(dirfig, "KEt_efold_Te_ppr", GMSER == 15 ? "" : "_GM$(GMSER)", ".png")
    savefig300(fout, fig)
    println("saved ", fout, "  (", size(fig.scene)[1], " x ", size(fig.scene)[2],
            " pt -> ", round(Int, 18*300/2.54), " px, tagged 300 dpi)")
end

## --- numbers behind the figure ----------------------------------------------
for (j, (lb, ttl, series)) in enumerate(PANELS)
    println("\n", "="^96)
    println(lb, " ", ttl)
    println("="^96)
    for (r, s) in zip(RES[j], series)
        @printf("\n  %s   (%d.%d-%d)\n", s[3], s[1], s[2][1], s[2][end])
        println("  ", rpad("lat",7), rpad("Cg1",8), rpad("x_valid",10),
                rpad("KEmin/KEmax",13), rpad("Le",12), rpad("Te",11), "status")
        for i in eachindex(r.lat)
            println("  ", rpad(@sprintf("%.1f",r.lat[i]),7), rpad(@sprintf("%.3f",r.Cg1[i]),8),
                rpad(@sprintf("%.0f km",r.xval[i]/1e3),10), rpad(@sprintf("%.3f",r.ratio[i]),13),
                rpad(@sprintf("%.0f km",r.Le[i]/1e3),12), rpad(@sprintf("%.2f d",r.Te[i]),11),
                r.fell[i] ? "lower bound only" : "measured")
        end
        ok = .!r.fell
        if any(ok)
            @printf("  -> measurable to %.1f°N;  mean Te = %.2f d over %d lats\n",
                    r.lat[ok][end], mean(r.Te[ok]), count(ok))
        end
    end
end
allTe = vcat([r.Te[.!r.fell] for pr in RES for r in pr]...)
@printf("\nplotted y-range %.2f to %.2f d;  measured points span %.2f to %.2f d\n",
        YLO, YHI, minimum(allTe), maximum(allTe))
