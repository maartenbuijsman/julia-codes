#= IW_energy_flux_CGE_2col_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: 4 rows x 2 columns of x-transects, one column per latitude, each
panel overlaying the tide-only and the GM+tide run at that latitude.

  rows                                          (all depth-integrated, 1-day-mean
                                                 fields written by the *_tile.jl codes)
    (a),(b)  KE   : total, tidal band, supertidal band          [kJ/m2]
    (c),(d)  APE  : total, tidal band, supertidal band          [kJ/m2]
    (e),(f)  F    : pressure + advective flux, same three bands  [kW/m]
                    = (Fx+FKx+FAx), Fx = <u'p'>, FKx/FAx = KE/APE advective flux
                      -- the "Fp+adv tot" combination of
                      IW_total_energetics_tile.jl:482
    (g),(h)  rho0*Pi : total cross-scale transfer                [mW/m2]
                    = rho0*(Pinhxa + Pixxa + Pizxa); Pi as saved is a
                      depth-integrated MASS-SPECIFIC power (W/kg*m), so the
                      rho0 factor makes it a power density comparable to dF/dx
                      (same convention as IW_analysis_energy_2000km_ppr.jl)

  columns                left = 2.5 N (runnm 28)     right = 28.8 N (runnm 34)
  runs per panel         11.RR  D2 tide only, no GM  -- thin, opaque
                         15.RR  GM + D2 tide         -- THICK, TRANSPARENT
  both blocks: 200 m NH, F = 25 kW/m, "zonalmean" Mercator N2(lat), and 15 is
  the redistribution-fix GM IC (see run_master.jl:89-95). So each pair differs
  ONLY by the presence of the GM background.

Band encoding is by COLOUR (black = total, red = tidal/D2, green =
supertidal/HH, following the colour convention of
IW_total_energetics_tile.jl:478-481) and run by WEIGHT (thin/opaque = no GM,
thick/transparent = GM), so six curves per panel need only a five-entry legend,
drawn once in the top margin above row 1.

Row 3 is the point of the figure: at 2.5 N the tidal-band flux collapses from
~24 kW/m near the source to ~0.2 kW/m by x = 1000 km while the supertidal band
picks up almost exactly that amount and the total stays ~25 kW/m, i.e. the
nonlinear steepening cascade read directly off a conserved flux budget. At
28.8 N the flux stays in the tidal band all the way down the transect
(~23 kW/m) with only ~1 kW/m ever reaching the supertidal band.

LAYOUT -- zero inter-panel whitespace (DSH = DSV = 0, as requested). The panels
then share edges exactly, which only works because the tick marks are turned
INWARD (x/ytickalign = 1); with Makie's default outward ticks the bottom ticks
of each panel would poke into the panel underneath. If outward ticks are wanted
instead, DSV must be at least one tick length, DSV = TICKSIZE/fig_h (~0.01 here,
and correspondingly DSH = TICKSIZE/fig_w) -- both are single constants below.
Interior tick LABELS are hidden (x on rows 1-3, y on column 2), and the y-range
is shared per row so the two latitudes are directly comparable.
Panels are placed at explicit bbox positions via functions/subplot_hor_vertpos.jl
(note its documented gotcha: vers/vere are BOTTOM/TOP, i.e. swapped from their
names, and an Axis parented to fig.scene does NOT autolimit itself).
=#

using Printf, JLD2, Statistics, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))   # RUN_TABLE, get_runs()

figflag = 1

const rho0 = 1020.0

# run selection -------------------------------------------------------------
const MAIN_NOGM = 11          # D2 tide only, no GM
const MAIN_GM   = 16          # GM + D2 tide: 16 = GM81 IC (1x GM81 over days 10-20); 15 = earlier ~3x GM IC
const RUNNMS    = (28, 34)    # column 1, column 2 -> lat 2.5, 28.8 (run_master.jl:39)

# x window: the right sponge occupies 1800-2000 km (Sp_Region_right = 200 km),
# so the transect is cut there -- same XBOUND convention as IW_KEt_efold_ppr.jl.
# The left cut at 100 km is the "clean domain" convention used elsewhere in this
# project: the flux-forcing/nudging region sits at x < ~100 km and the raw
# tidal-band flux swings to about -35 kW/m there, which is a property of the
# source formulation, not of the wave, and which single-handedly sets the row-3
# y-range if it is left in.
const XLO, XHI = 100e3, 1800e3

# smoothing -----------------------------------------------------------------
# Pi is ALWAYS Gaussian-smoothed at sigma = 1600 m, matching the established
# convention of IW_analysis_coarsegr_2000km.jl (applied whenever dx < 500 m;
# both blocks here are on the 200 m grid). The KE/APE/flux transects are left at
# FULL RESOLUTION by default -- turn SMOOTH_E on only if the 200 m wiggle
# actually obscures something.
const LSM      = 1600.0
const SMOOTH_E = false

# unit scalings -------------------------------------------------------------
const fcKE = 1e-3     # J/m2   -> kJ/m2
const fcF  = 1e-3     # W/m    -> kW/m
const fcPI = 1e3      # W/m2   -> mW/m2  (applied AFTER the rho0 factor)

## load ---------------------------------------------------------------------
#= energetics_AMZexptMM.RR.jld2 (IW_total_energetics_tile.jl:540) holds
   xc, freq, KEoma, KEommax, Fx, Fxt, Fxh, Fxs, FAx, FAxt, FAxh, FAxs,
   FKx, FKxt, FKxh, FKxs, KE, KEt, KEh, KEs, APE, APEt, APEh, APEs
   Etran_AMZexptMM.RR.jld2   (IW_coarsegraining_tile.jl:369) holds
   LAT, xc, zc, Pinhxa, Pixxa, Pizxa, Pinhza, Pixza, Pizza, Pixztot          =#

"""
    load_run(mainnm, runnm) -> NamedTuple

Everything one panel column needs for one simulation, already in plot units and
already cut to XLO..XHI. `Pi` comes from the Etran file on its own xc (same
200 m grid, but read independently rather than assumed identical).
"""
function load_run(mainnm, runnm)
    row = get_runs(mainnm, [runnm])[1]        # errors if the pair isn't in RUN_TABLE
    fn  = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)

    d  = load(string(dirout, "energetics_", fn, ".jld2"))
    xe = d["xc"]
    sm(y) = SMOOTH_E ? gaussfilt(xe, y, LSM) : y

    # pressure + advective flux, per band: Fx is <u'p'>, FKx/FAx the KE/APE
    # advective parts -- summed exactly as in IW_total_energetics_tile.jl:482
    Fb(p) = sm(d["Fx"*p] .+ d["FKx"*p] .+ d["FAx"*p]) .* fcF

    e  = load(string(dirout, "Etran_", fn, ".jld2"))
    xp = e["xc"]
    Pi = e["Πnhxa"] .+ e["Πxxa"] .+ e["Πzxa"]
    (xp[2]-xp[1]) < 500 && (Pi = gaussfilt(xp, Pi, LSM))   # 200 m grid -> smooth
    Pi = Pi .* (rho0*fcPI)

    Ie = findall(x -> XLO <= x <= XHI, xe)
    Ip = findall(x -> XLO <= x <= XHI, xp)

    println(fn, "; lat=", row.lat, "; F=", row.Flux/1e3, " kW/m -------------------")

    return (lat  = row.lat, tag = fn,
            xe   = xe[Ie]/1e3,                                   # km
            KE   = (sm(d["KE" ])[Ie].*fcKE, sm(d["KEt" ])[Ie].*fcKE, sm(d["KEh" ])[Ie].*fcKE),
            APE  = (sm(d["APE"])[Ie].*fcKE, sm(d["APEt"])[Ie].*fcKE, sm(d["APEh"])[Ie].*fcKE),
            F    = (Fb("")[Ie], Fb("t")[Ie], Fb("h")[Ie]),
            xp   = xp[Ip]/1e3,                                   # km
            Pi   = Pi[Ip])
end

# [column][1] = no GM, [column][2] = GM
RUNS = [(load_run(MAIN_NOGM, r), load_run(MAIN_GM, r)) for r in RUNNMS]

## figure geometry ----------------------------------------------------------
cm_to_pt = 72/2.54
fig_w = 18*cm_to_pt          # 18 cm
fig_h = 22*cm_to_pt          # 22 cm

const TICKSIZE = 4.0         # tick length [pt]; also the DSH/DSV fallback below

# DSH/DSV = 0 gives panels that share edges exactly -- only legitimate with
# inward ticks (TICKALIGN = 1). For outward ticks set both to one tick length:
#   DSH = TICKSIZE/fig_w ; DSV = TICKSIZE/fig_h
const TICKALIGN = 1
const DSH = 0.0
const DSV = 0.0

# hors/hore = LEFT/RIGHT margin, vers/vere = BOTTOM/TOP margin (the names are
# swapped relative to their meaning -- see subplot_hor_vertpos.jl). The top
# margin holds the shared legend, the bottom one the x tick labels + xlabel.
const HORS, HORE = 0.105, 0.032   # HORE leaves room for the last x tick label ("1800")
const VERS, VERE = 0.058, 0.034   # VERE holds the column titles only

# Panel labels sit in the top-RIGHT corner of each panel, in a white box that
# occludes whatever passes behind it. LABLIFT = true would instead stretch each
# row's upper limit until no curve enters that corner (everything drawn within
# LABX of the label's side kept under LABTOP of the panel height) -- left off
# here so the rows keep their natural data range.
const LABLIFT = false
const LABTOP  = 0.80
const LABX    = 0.16

# y ticks a row must show even though insideticks() would drop them (too close
# to a panel edge) or the padded data range would not reach them. Where a forced
# tick lies outside that range the row's limits are stretched to make room --
# the only place in this file where a limit is set by anything but the data.
const YFORCE = Dict(4 => [-10.0, 30.0])
# row 1's cap (20) IS the panel's hard top edge now, so its forced tick needs
# no stretch-for-clearance (there's nothing beyond the edge to clear) -- just
# added straight to the label list below, unlike YFORCE's entries.
const EDGETICKS = Dict(1 => [20.0])

pos  = subplot_hor_vertpos(2, 4, HORS, HORE, VERS, VERE, DSH, DSV)
bb(i) = BBox(subplot_bbox(pos[i], fig_w, fig_h)...)
pidx(row, col) = (row-1)*2 + col      # subplot_hor_vertpos fills L->R, top->bottom

## style --------------------------------------------------------------------
const CBAND  = (:black, :red, :green)                  # total, tidal (D2), supertidal (HH)
const LBAND  = ("total", "tidal (D2)", "supertidal (HH)")
const LW_NOGM, LW_GM = 1.6, 3.6
const AL_GM  = 0.45                                    # GM lines thick AND transparent

rowlab = ("KE [kJ/m²]", "APE [kJ/m²]", "F [kW/m]", "ρ₀Π [mW/m²]")

fig = Figure(size=(fig_w, fig_h), fontsize=10)   # 10 pt: paper-figure standard

axs = Matrix{Any}(undef, 4, 2)
for row in 1:4, col in 1:2
    ax = Axis(fig.scene, bbox = bb(pidx(row, col)),
        xtickalign = TICKALIGN, ytickalign = TICKALIGN,
        xticksize = TICKSIZE, yticksize = TICKSIZE,
        xminortickalign = TICKALIGN, yminortickalign = TICKALIGN,
        xgridvisible = false, ygridvisible = false,
        # "1800" is labelled in BOTH columns: col 1's sits on the shared seam,
        # but col 2's first label is 300 (its axis starts at 100 km), so there
        # is nothing on the other side of the seam for it to collide with
        xticks = (collect(300.0:300.0:1800.0), ["300","600","900","1200","1500","1800"]),
        xlabel = row == 4 ? "x [km]" : "",
        ylabel = col == 1 ? rowlab[row] : "",
        xticklabelsvisible = row == 4,      # interior labels off: panels touch
        yticklabelsvisible = col == 1,
        xlabelvisible = row == 4, ylabelvisible = col == 1)
    axs[row, col] = ax
end

# draw ----------------------------------------------------------------------
for col in 1:2
    for (ir, key) in enumerate((:KE, :APE, :F))
        ax = axs[ir, col]
        hlines!(ax, [0.0], color = :gray70, linewidth = 0.6)   # zero reference in every row
        for (r, (lw, al)) in zip(RUNS[col], ((LW_NOGM, 1.0), (LW_GM, AL_GM)))
            y3 = getfield(r, key)
            for ib in 1:3
                lines!(ax, r.xe, y3[ib], color = (CBAND[ib], al), linewidth = lw)
            end
        end
    end
    ax = axs[4, col]
    hlines!(ax, [0.0], color = :gray70, linewidth = 0.6)
    for (r, (lw, al)) in zip(RUNS[col], ((LW_NOGM, 1.0), (LW_GM, AL_GM)))
        lines!(ax, r.xp, r.Pi, color = (:black, al), linewidth = lw)
    end
end

## limits -------------------------------------------------------------------
# an Axis parented to fig.scene never autolimits itself (subplot_hor_vertpos.jl),
# so every limit is set explicitly here. y is SHARED per row across both
# columns so the two latitudes can be read against each other; rows 3-4 change
# sign and always keep 0 inside the range.
"""
    padlims(arrays; signed=false, pad=0.05)

Common y-limits over all `arrays`, padded by `pad` of the range. `signed=true`
forces 0 into the range (used for the flux and Π rows).
"""
function padlims(arrays; signed::Bool=false, pad::Real=0.05)
    lo = minimum(minimum.(arrays)); hi = maximum(maximum.(arrays))
    if signed
        lo = min(lo, 0.0); hi = max(hi, 0.0)
    end
    d = hi - lo
    d == 0 && (d = max(abs(hi), 1.0))
    return (lo - pad*d, hi + pad*d)
end

"""
    liftfor_label(lo, hi, ys, xs)

Raise `hi` until every curve in `ys` (on abscissae `xs`) that falls in the
rightmost `LABX` of the panel sits below `LABTOP` of the panel height, i.e.
clear of the "(a)"-style corner label. `lo` is never touched, and a row whose
curves already clear the corner keeps its padded limits unchanged. A no-op
unless `LABLIFT` is set.
"""
function liftfor_label(lo, hi, ys, xs)
    LABLIFT || return (lo, hi)
    xstart = (XHI - LABX*(XHI - XLO))/1e3        # label is at the RIGHT edge now
    m = -Inf
    for (y, x) in zip(ys, xs)
        I = findall(>=(xstart), x)
        isempty(I) || (m = max(m, maximum(y[I])))
    end
    m == -Inf && return (lo, hi)
    return (lo, max(hi, lo + (m - lo)/LABTOP))
end

# y-limits are shared per row across both columns, so the label clearance is
# evaluated over both columns at once and the row keeps a single scale
ylims_row = Vector{Tuple{Float64,Float64}}(undef, 4)
for (ir, key) in enumerate((:KE, :APE, :F))
    ys = [getfield(r, key)[ib] for col in 1:2 for r in RUNS[col] for ib in 1:3]
    xs = [r.xe                 for col in 1:2 for r in RUNS[col] for ib in 1:3]
    lo, hi = padlims(ys; signed = (ir == 3))
    ylims_row[ir] = liftfor_label(lo, hi, ys, xs)
end
let ys = [r.Pi for col in 1:2 for r in RUNS[col]],
    xs = [r.xp for col in 1:2 for r in RUNS[col]]
    lo, hi = padlims(ys; signed = true)
    ylims_row[4] = liftfor_label(lo, hi, ys, xs)
end

# Rows 1-2 capped by hand instead of the data (which reaches ~29 kJ/m2 KE and
# ~10 kJ/m2 APE at 28.8N, both driven by the GM background in panel (b)): KE
# capped at 20, APE at 8. This clips panel (b)'s grey GM total line at the top
# in both rows -- deliberate, called out in the caption, not a bug to fix here.
ylims_row[1] = (ylims_row[1][1], 20.0)
ylims_row[2] = (ylims_row[2][1], 8.0)

"""
    insideticks(lo, hi; target=4, edge=0.08)

Nice-number ticks that all fall STRICTLY INSIDE `lo..hi`, keeping clear of the
panel edges by `edge` of the range. With DSV = 0 the panels share edges, so a
tick label at the very top of one panel would otherwise sit on top of the label
at the very bottom of the panel above it.
"""
function insideticks(lo, hi; target::Int=4, edge::Real=0.08)
    rng = hi - lo
    lo2, hi2 = lo + edge*rng, hi - edge*rng
    raw = rng/target
    mag  = 10.0^floor(log10(raw))
    step = (raw/mag <= 1.5 ? 1.0 : raw/mag <= 3.0 ? 2.0 : raw/mag <= 7.0 ? 5.0 : 10.0)*mag
    tk = collect(ceil(lo2/step)*step : step : hi2)
    # 0 is always shown, even when it falls in the edge-clearance band -- the
    # zero crossing is the reference every panel is read against. Added to BOTH
    # columns so the tick marks stay aligned across the shared column edge; only
    # column 1 draws the labels.
    lo < 0 < hi && (tk = sort(unique(vcat(tk, 0.0))))
    return tk
end

# stretch a row's limits, if needed, so its forced ticks fall inside with a
# little clearance from the panel edge
for row in 1:4
    ft = get(YFORCE, row, Float64[])
    isempty(ft) && continue
    lo, hi = ylims_row[row]; rng = hi - lo
    ylims_row[row] = (min(lo, minimum(ft) - 0.06*rng),
                      max(hi, maximum(ft) + 0.06*rng))
end

for row in 1:4, col in 1:2
    xlims!(axs[row, col], XLO/1e3, XHI/1e3)
    ylims!(axs[row, col], ylims_row[row]...)
    tk = insideticks(ylims_row[row]...)
    ft = get(YFORCE, row, Float64[])
    isempty(ft) || (tk = sort(unique(vcat(tk, ft))))
    et = get(EDGETICKS, row, Float64[])
    isempty(et) || (tk = sort(unique(vcat(tk, et))))
    axs[row, col].yticks = tk
end

## panel labels and column titles ------------------------------------------
# text_fignum! reads the axis's finallimits, so it must come AFTER the limits
labs = ("(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)")
for row in 1:4, col in 1:2
    text_fignum!(axs[row, col], labs[pidx(row, col)];
        horloc = :right, vertloc = :top, offhorz = -0.025, offvert = -0.04, fs = 10,
        bckclr = :white, boxw = 0.056, boxh = 0.091)  # boxw 0.075 -> -25% to the left; boxh 0.13 -> -30%, top edge unchanged (bottom raised)
end

# Column titles live in the TOP MARGIN, just above the row-1 panels, not inside
# them: with DSV = 0 an in-panel title is one more thing the curves have to stay
# clear of, and row 1 would have to be stretched a long way to keep the GM trace
# out of a centred title. The band legend, by contrast, sits INSIDE panel (a).
for col in 1:2
    r  = RUNS[col][1]
    xc_px = (pos[pidx(1, col)][1] + pos[pidx(1, col)][3]/2) * fig_w
    text!(fig.scene, xc_px, (1 - VERE + 0.008)*fig_h, align = (:center, :bottom),
        text = @sprintf("%.1f°N   (%i.%02i vs %i.%02i)", r.lat,
                        MAIN_NOGM, RUNNMS[col], MAIN_GM, RUNNMS[col]),
        fontsize = 10, font = :bold)
end

## two legends, one per run series ------------------------------------------
# Split into a "no GM (11)" legend (thin/opaque line style) and a "GM+tide
# (15)" legend (thick/transparent line style) -- each shows the same 3 bands
# in that series' own linewidth/alpha, so the legend glyphs themselves carry
# the run distinction that used to be stated only in the column titles.
elems_nogm = [LineElement(color = CBAND[ib], linewidth = LW_NOGM) for ib in 1:3]
elems_gm   = [LineElement(color = (CBAND[ib], AL_GM), linewidth = LW_GM) for ib in 1:3]
labels3    = ["Total", "Tidal (D2)", "Supertidal (HH)"]

"""
    mklegend!(elems, labels, title, bbox; f=1.0, wf=1.0)

One legend BLOCK: the run title sits directly on top of its three entries,
left-aligned with them and with almost no gap, so title + entries read as a
single unit rather than a floating heading above a separate list. Opaque
white background over the whole block.
"""
function mklegend!(elems, labels, title, bbox; f=1.0, wf=1.0, valign=:top)
    Legend(fig.scene, elems, labels, title;
        bbox = bbox, orientation = :vertical, framevisible = false, backgroundcolor = :white,
        titlehalign = :left, titlegap = 1, titlesize = 8*f, titlefont = :bold,
        labelsize = 8*f, patchsize = (16*wf, 7*f), rowgap = 0, padding = (3, 3, 2, 2),
        halign = :left, valign = valign)
end

const LEGW_PT, LEGH_PT = 90.0, 40.0   # legend box's DEFAULT size in points -- place_legend! shrinks
                                       # it automatically (see SHRINKS) if no clear window exists at this size

"""
    safe_legend_left(xs, ys_list, xlo, xhi, width, edge; anchor, step)

Leftmost x (data units) such that a window of horizontal extent `width`, anchored
to the TOP (curves must stay below `edge`) or BOTTOM (curves must stay above
`edge`) of the panel, contains no point from any curve in `ys_list` (each
sampled at the matching `xs`). Scans left to right in steps of `step` and
returns the first x that clears every curve over the full window, or
`nothing` if no such window exists at this size (the caller decides what to
do next -- shrink, relocate, or accept an overlap -- rather than this
function silently picking one).
"""
function safe_legend_left(xs, ys_list, xlo, xhi, width, edge; anchor::Symbol=:top, step=20e3)
    x = xlo
    while x + width <= xhi
        clear = true
        for ys in ys_list
            I = findall(t -> x <= t <= x + width, xs)
            if !isempty(I)
                v = anchor == :top ? maximum(@view ys[I]) : minimum(@view ys[I])
                bad = anchor == :top ? (v > edge) : (v < edge)
                bad && (clear = false; break)
            end
        end
        clear && return x
        x += step
    end
    return nothing
end

# Width is tried first, independently of height/font -- a narrow-but-legible
# box (font/patch unchanged) is always preferable to a smaller-font one, so
# only fall through to shrinking height/font (WHSHRINKS) if NO width alone,
# at full height, ever clears.
const WSHRINKS  = (1.0, 0.85, 0.7, 0.55, 0.4, 0.3)
const WHSHRINKS = (0.8, 0.65, 0.5)

"""
    place_legend!(elems, labels, title, panel_pos, col, key, lo, hi; anchor=:top)

Places a legend flush against the TOP or BOTTOM (`anchor`) of the panel at
`panel_pos` (a `pos[...]` tuple), at whichever x clears every curve in that
panel for the given data `key` (:KE or :APE) across BOTH runs shown there (so
a legend inside panel (a), which overlays 11.28 and 15.28, is kept clear of
both series even though it only labels one of them). `lo`/`hi` must be the
row's ACTUAL shared y-limits (`ylims_row[row]`), not recomputed from this
column alone -- the panel's pixel height corresponds to that full range, so
the edge has to be measured against it, not a column-local min/max.

If the default size (`LEGW_PT`x`LEGH_PT`) has no clear window anywhere, WIDTH
alone is shrunk first (font/patch untouched -- a narrower but fully legible
box) through `WSHRINKS`; only if that entirely fails is height/font also
shrunk (`WHSHRINKS`, both dimensions together) before giving up and placing
the legend at the panel edge anyway -- always printed, never silent, either
way.
"""
function place_legend!(elems, labels, title, panel_pos, col, key, lo, hi;
                        anchor::Symbol=:top, w0::Real=LEGW_PT, h0::Real=LEGH_PT)
    xs     = RUNS[col][1].xe .* 1e3   # r.xe is in km -- back to m to match XLO/XHI
    ys_all = vcat([collect(getfield(r, key)) for r in RUNS[col]]...)   # 3 bands x 2 runs

    x0 = nothing; f = 1.0; wf = 1.0; w = w0; h = h0; edge = NaN
    for hfac in (1.0, WHSHRINKS...)
        f = hfac; h = h0*f
        frac = h / (panel_pos[4]*fig_h)
        edge = anchor == :top ? hi - frac*(hi - lo) : lo + frac*(hi - lo)
        for wfac in WSHRINKS
            wf = wfac; w = w0*wfac
            width = w / (panel_pos[3]*fig_w) * (XHI - XLO)
            x0 = safe_legend_left(xs, ys_all, XLO, XHI, width, edge; anchor)
            x0 !== nothing && break
        end
        x0 !== nothing && break
    end
    width = w / (panel_pos[3]*fig_w) * (XHI - XLO)
    fellback = x0 === nothing
    fellback && (x0 = anchor == :top ? XLO : XHI - width)

    # verify -- print the actual extreme curve value inside the chosen window
    # so a silent overlap can never slip through unreported
    I = findall(t -> x0 <= t <= x0+width, xs)
    wext = isempty(I) ? NaN : (anchor == :top ? maximum(maximum(@view ys[I]) for ys in ys_all)
                                               : minimum(minimum(@view ys[I]) for ys in ys_all))
    clear = anchor == :top ? wext <= edge : wext >= edge
    println(title, " legend (", anchor, ", w=", round(wf, digits=2), "x h=", round(f, digits=2), "x): x0=",
            round(x0/1e3, digits=0), "-", round((x0+width)/1e3, digits=0),
            " km, extreme curve in window=", round(wext, digits=2), ", edge=", round(edge, digits=2),
            clear ? "  [CLEAR]" : "  [OVERLAP]")

    lg_l = panel_pos[1]*fig_w + (x0 - XLO)/(XHI - XLO) * panel_pos[3]*fig_w
    if anchor == :top
        lg_t = (panel_pos[2] + panel_pos[4])*fig_h - 0.015*panel_pos[4]*fig_h
        bbox = BBox(lg_l, lg_l + w, lg_t - h, lg_t)
    else
        lg_b = panel_pos[2]*fig_h + 0.015*panel_pos[4]*fig_h
        bbox = BBox(lg_l, lg_l + w, lg_b, lg_b + h)
    end
    return mklegend!(elems, labels, title, bbox; f, wf, valign = anchor)
end

"""
    place_legend_gap!(elems, labels, title, panel_pos, col, key, lo, hi; w, h, margin)

Places a `w`x`h` (points) legend block FLOATING inside the panel -- not tied
to its top or bottom edge -- in the widest vertical gap between curves. For
every candidate x-window it takes each curve's min..max over that window (a
curve anywhere in that band would cross the box), merges those bands, and
finds the free y-intervals left over; the box goes, centred vertically, in
the largest free interval over all windows. In panel (c) that is the wedge
between the rising supertidal (green) lines above and the decayed tidal (red)
lines below, east of their crossover. `margin` (points) is extra clearance
demanded above and below the box.
"""
function place_legend_gap!(elems, labels, title, panel_pos, col, key, lo, hi;
                            w::Real=LEGW_PT, h::Real=LEGH_PT, margin::Real=3.0, step=20e3)
    xs     = RUNS[col][1].xe .* 1e3
    ys_all = vcat([collect(getfield(r, key)) for r in RUNS[col]]...)   # 3 bands x 2 runs
    ppu    = panel_pos[4]*fig_h / (hi - lo)                             # points per data unit (y)
    hd     = (h + 2*margin) / ppu                                       # needed gap, data units
    width  = w / (panel_pos[3]*fig_w) * (XHI - XLO)

    best = (gap = -Inf, x0 = XLO, ymid = (lo+hi)/2)
    x = XLO
    while x + width <= XHI
        I = findall(t -> x <= t <= x + width, xs)
        bands = sort([(minimum(@view ys[I]), maximum(@view ys[I])) for ys in ys_all])
        # free intervals between merged bands, inside the panel's own y-range
        cur = lo
        for (b0, b1) in bands
            if b0 > cur
                g = b0 - cur
                g > best.gap && (best = (gap = g, x0 = x, ymid = (cur + b0)/2))
            end
            cur = max(cur, b1)
        end
        (hi - cur) > best.gap && (best = (gap = hi - cur, x0 = x, ymid = (cur + hi)/2))
        x += step
    end

    clear = best.gap >= hd
    println(title, " legend (gap): x0=", round(best.x0/1e3, digits=0), "-",
            round((best.x0 + width)/1e3, digits=0), " km, free gap=", round(best.gap, digits=2),
            " (needs ", round(hd, digits=2), "), centred at y=", round(best.ymid, digits=2),
            clear ? "  [CLEAR]" : "  [OVERLAP]")

    lg_l = panel_pos[1]*fig_w + (best.x0 - XLO)/(XHI - XLO) * panel_pos[3]*fig_w
    yc   = panel_pos[2]*fig_h + (best.ymid - lo) * ppu
    return mklegend!(elems, labels, title, BBox(lg_l, lg_l + w, yc - h/2, yc + h/2); valign = :center)
end

# Both legends grouped in COLUMN 1 (2.5N), one directly above the other --
# "no GM (11)" in (a), string("GM + tide (", MAIN_GM, ")") in (c) right below it -- rather than
# split diagonally across (a) and (d), which read as unrelated call-outs
# instead of a paired key.
#
# Each is one identical block: run title, then Total / Tidal (D2) /
# Supertidal (HH) in that run's own line style. (a)'s sits at the top of the
# panel; (c)'s floats in the wedge between the supertidal (green, above) and
# tidal (red, below) lines -- (c)'s own GM-total APE line sits at 6.3-9.8
# kJ/m2 almost everywhere, so under the row-2 cap of 8 there is no room for a
# 4-row block at the top of (c), but plenty between green and red.
LEGBOX = (w = 105.0, h = 53.0)   # both blocks the same size, so they read as a pair
lgA = place_legend!(elems_nogm, labels3, "no GM (11)", pos[pidx(1, 1)], 1, :KE, ylims_row[1]...;
                    w0 = LEGBOX.w, h0 = LEGBOX.h)
lgC = place_legend_gap!(elems_gm, labels3, string("GM + tide (", MAIN_GM, ")"), pos[pidx(2, 1)], 1, :APE, ylims_row[2]...;
                        w = LEGBOX.w, h = LEGBOX.h)

# the drawn block is sized to its CONTENT inside the checked box -- confirm
# it really does fit inside the box that was checked for clearance
for (nm, lg, bb) in (("(a)", lgA, nothing), ("(c)", lgC, nothing))
    cb = lg.layoutobservables.computedbbox[]
    println(nm, " legend drawn size: ", round(cb.widths[1], digits=1), " x ",
            round(cb.widths[2], digits=1), " pt (checked box ", LEGBOX.w, " x ", LEGBOX.h, ")")
end

display(fig)

if figflag == 1
    fout = string(dirfig, @sprintf("energy_flux_CGE_2col_%i-%i.%02i-%02i.png",
                                   MAIN_NOGM, MAIN_GM, RUNNMS[1], RUNNMS[2]))
    savefig300(fout, fig)
    println("saved ", fout)
end

## quick numbers for the text ----------------------------------------------
# the band-to-band flux handover, quoted at a few x positions
for col in 1:2, r in RUNS[col]
    println("--- ", r.tag, "  lat=", r.lat, "  F [kW/m]: tot | tidal | supertidal")
    for xt in (200.0, 500.0, 1000.0, 1750.0)
        i = argmin(abs.(r.xe .- xt))
        @printf("    x=%6.0f km  %7.2f %7.2f %7.2f   KEt=%7.1f KEh=%7.1f  ρ₀Π=%8.2f\n",
                r.xe[i], r.F[1][i], r.F[2][i], r.F[3][i], r.KE[2][i], r.KE[3][i],
                r.Pi[argmin(abs.(r.xp .- xt))])
    end
end

##
return
