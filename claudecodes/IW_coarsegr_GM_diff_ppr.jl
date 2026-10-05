#= IW_coarsegr_GM_diff_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: sign-conditioned cross-scale energy transfer Π for the mainnm=15
GM series (w,b-consistent polarization-relation IC + redistribution fix), and
the nonlinear GM-tide residual as a fraction of the tide's own transfer.
Paper version of the exploratory 2x2 in IW_coarsegr_GM_diff.jl, with the
mainnm=13 series dropped.

Blocks used:
  - 15.1-12   GM only, no tide   (Flux=0)
  - 15.27-38  GM + D2 tide       (F=25 kW/m)
  - 11.27-38  D2 tide only, no GM (F=25 kW/m, same lat/flux)
(lat=50 transect excluded, 12 sims, matching IW_analysis_energy_2000km_ppr.jl)

Π oscillates in x -- alternating down-scale (Π>0) and up-scale (Π<0) lobes --
so a plain transect mean lets the two partially cancel and the lobe positions
then control the apparent latitude dependence. The transect is therefore split
into positive- and negative-Π regions defined by the TIDE-ONLY run, and each
region is averaged separately. The mask is a property of the tide alone and is
applied identically to all three runs, so it introduces no bias between them.
(A separate phase-shift test -- cross-correlation of Π_tide against the de-GM'd
Π_gm+tide -- gives a median |lag| of ~10 km and a median correlation gain from
shifting of only +0.002, so the masked comparison is not shift-contaminated.)

Layout: top row = forward (Π>0) and inverse (Π<0) transfer, shared y-range,
bottom row = the corresponding relative change
    (⟨Π⟩gm+tide − ⟨Π⟩gm − ⟨Π⟩tide) / ⟨Π⟩tide
In the inverse regions ⟨Π⟩tide < 0, so a POSITIVE ratio there still means the
interaction reinforces the tide's own transfer -- same reading as forward.
=#

using Printf, JLD2, Statistics, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1

const LATALL  = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
const GMSER = 16   # GM series: 16 (GM81 IC, 1x GM81 over days 10-20) or 15 (earlier ~3x GM IC; untagged output names)
const NLAT    = 12        # drop the lat=50 transect (last runnm of each block), as in IW_analysis_energy_2000km_ppr.jl
const LATS    = LATALL[1:NLAT]
const Lsmooth = 1600.0    # Gaussian sigma [m], as in IW_analysis_coarsegr_2000km.jl
const fcPi    = 1e6       # W/kg m -> 1e-6 W/kg m, folded into the y-axis label
const XTICKS  = 0:10:40
# Analysis window: the Π transect is written over the full 0-2000 km, which
# includes the left sponge (to 40 km) and the right sponge (from 1800 km) --
# 12% of the domain. Both are trimmed here so that "fraction of the domain"
# and the domain average refer to the freely propagating part only. Same
# window as the M4 beat analysis.
const XLO     = 100e3
const XHI     = 1800e3

function load_CGE(mainnm, runnms)
    fnames0 = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "Etran_", fnames0, ".jld2") xc
    dx = xc[2] - xc[1]
    CGE = zeros(length(runnms), length(xc))
    for (i, rn) in enumerate(runnms)
        @load string(dirout, @sprintf("Etran_AMZexpt%02i.%02i.jld2", mainnm, rn)) Πnhxa Πxxa Πzxa
        p = Πnhxa .+ Πxxa .+ Πzxa
        dx < 500 && (p = gaussfilt(xc, p, Lsmooth))
        CGE[i, :] = p
    end
    return xc, CGE
end

xc,  C11 = load_CGE(11, collect(27:26+NLAT))   # tide only
_,   Cgm = load_CGE(GMSER, collect(1:NLAT))       # GM only
_,   Cbo = load_CGE(GMSER, collect(27:26+NLAT))   # GM + tide

# trim the sponges: everything below is over 100-1800 km only
Iw  = findall(v -> XLO <= v <= XHI, xc)
xc  = xc[Iw];  C11 = C11[:, Iw];  Cgm = Cgm[:, Iw];  Cbo = Cbo[:, Iw]
Ld  = xc[end] - xc[1]
@printf("analysis window: %.0f-%.0f km, Ld = %.0f km, %d points\n",
        xc[1]/1e3, xc[end]/1e3, Ld/1e3, length(xc))

nlat  = NLAT
maskP = [C11[i, :] .> 0 for i in 1:nlat]
maskN = [C11[i, :] .< 0 for i in 1:nlat]
fracP = [count(maskP[i]) / length(maskP[i]) for i in 1:nlat]
fracN = 1 .- fracP

condmean(A, mask) = [mean(A[i, mask[i]]) for i in 1:size(A, 1)]

mP_tide = condmean(C11, maskP);  mN_tide = condmean(C11, maskN)
mP_gm   = condmean(Cgm, maskP);  mN_gm   = condmean(Cgm, maskN)
mP_both = condmean(Cbo, maskP);  mN_both = condmean(Cbo, maskN)

# The top row plots the DOMAIN-normalised partial mean
#     ⟨Π⟩± = (1/Ld) ∫_{Π≷0} Π dx  =  f± × (mean of Π over that region)
# i.e. the same sum divided by the WHOLE window rather than by the region it
# came from. Dividing by the region instead says how intense the transfer is
# where it occurs but not how much of the domain it occupies, so a large
# inverse mean over a sliver looks as important as a moderate one over half the
# domain -- at the equator the region mean is -4.2e-8 over 5.9% of the
# transect, which contributes essentially nothing. Normalising by Ld also makes
# the two panels ADD UP: ⟨Π⟩₊ + ⟨Π⟩₋ is exactly the net domain mean (verified
# to round-off), so (a) and (b) are the down- and up-scale halves of one budget.
# The masks come from the tide-only run and are identical for all three runs,
# so f± cancels exactly in the ratios below: the bottom row is unchanged by
# this normalisation.
wP_tide = fracP .* mP_tide;  wN_tide = fracN .* mN_tide
wP_gm   = fracP .* mP_gm;    wN_gm   = fracN .* mN_gm
wP_both = fracP .* mP_both;  wN_both = fracN .* mN_both

ratP = (mP_both .- mP_gm .- mP_tide) ./ mP_tide
ratN = (mN_both .- mN_gm .- mN_tide) ./ mN_tide

# lat=0 has only ~6% of the transect with Π<0, so ⟨Π⟩₋ there is a small-sample
# mean of a near-zero quantity and ratN(0°) is correspondingly huge. It IS
# plotted in (d), but it is excluded from the y-range calculation below so it
# does not compress the rest of the panel -- it simply runs off the top.

## --- figure: 18 x 18 cm, fontsize 10 ----------------------------------------
cm_to_pt = 72/2.54
fig_w = 18*cm_to_pt
fig_h = 18*cm_to_pt

col_tide = :black
col_both = :crimson
col_gm   = :steelblue

fig = Figure(size=(fig_w, fig_h), fontsize=10)

# shared y-range for the two top panels, so forward and backward magnitudes
# can be compared directly by eye
alltop = vcat(wP_tide, wP_both, wP_gm, wN_tide, wN_both, wN_gm) .* fcPi
pad    = 0.05*(maximum(alltop) - minimum(alltop))
ytop   = (minimum(alltop) - pad, maximum(alltop) + pad)

ax_a = Axis(fig[1,1], title="forward,  Π > 0 regions", xticks=XTICKS,
    ylabel="⟨Π⟩ [10⁻⁶ W kg⁻¹ m]", xticklabelsvisible=false)
ax_b = Axis(fig[1,2], title="inverse,  Π < 0 regions", xticks=XTICKS,
    xticklabelsvisible=false, yticklabelsvisible=false)
ax_c = Axis(fig[2,1], xlabel="latitude [°]", ylabel="Δ⟨Π⟩ / ⟨Π⟩tide", xticks=XTICKS)
ax_d = Axis(fig[2,2], xlabel="latitude [°]", yticklabelsvisible=false, xticks=XTICKS)

# Area fraction of each sign, drawn FIRST so it sits behind every other line.
# Stretched onto the left axis with 0 -> 0 and a fraction of 1 -> FRAC1 (in the
# plotted 10⁻⁶ units), so the curve uses the full height of the panel instead
# of being squeezed into one unit of it. Both panels draw the fraction UPWARD
# on the same mapping, so the two green curves are directly comparable; only
# panel (b) carries the tick labels for it.
const FRAC1 = 7.5
col_frac = :seagreen
lines!(ax_a, LATS, FRAC1 .* fracP, color=col_frac, linewidth=2.2, linestyle=:dash)
lines!(ax_b, LATS, FRAC1 .* fracN, color=col_frac, linewidth=2.2, linestyle=:dash)

for (ax, yv) in ((ax_a, (wP_tide, wP_both, wP_gm)), (ax_b, (wN_tide, wN_both, wN_gm)))
    hlines!(ax, [0.0], color=:gray70, linestyle=:dash)
    lines!(ax, LATS, yv[1].*fcPi, color=col_tide, linewidth=2, label="tide only")
    scatterlines!(ax, LATS, yv[2].*fcPi, color=col_both, marker=:rect, markersize=7,
        linewidth=1.5, label="GM + tide")
    scatterlines!(ax, LATS, yv[3].*fcPi, color=col_gm, marker=:circle, markersize=7,
        linewidth=1.5, label="GM only")
    ylims!(ax, ytop...)
    xlims!(ax, -2, 47)
end
axislegend(ax_a, position=:rb, framevisible=false, labelsize=10)

# Label-only right-hand axis for the green area-fraction curve, on panel (b)
# only. Ticks sit at FRAC1*f, the same mapping both panels use; only fractions
# inside the shared y-range are labelled.
ftk   = [f for f in 0:0.25:1.0 if FRAC1*f <= ytop[2]]
ax_bf = Axis(fig[1,2], yaxisposition=:right, backgroundcolor=:transparent,
    ylabel="area fraction", ylabelcolor=col_frac, yticklabelcolor=col_frac,
    yticks=(collect(FRAC1 .* ftk), [@sprintf("%.2f",v) for v in ftk]))
hidexdecorations!(ax_bf);  hidespines!(ax_bf)
ax_bf.ygridvisible = false
ylims!(ax_bf, ytop...);  xlims!(ax_bf, -2, 47)

# shared y-range for the bottom pair too: the forward and backward relative
# changes are then directly comparable, which is the point of the figure
allbot = vcat(ratP, ratN[2:end])     # ratN(0°) excluded: see note above
padb   = 0.08*(maximum(allbot) - minimum(allbot))
ybot   = (minimum(allbot) - padb, maximum(allbot) + 2.2*padb)   # headroom for the panel letter

for (ax, rv) in ((ax_c, ratP), (ax_d, ratN))
    hlines!(ax, [0.0], color=:gray70, linestyle=:dash)
    scatterlines!(ax, LATS, rv, color=col_both, marker=:rect, markersize=7, linewidth=1.5)
    xlims!(ax, -2, 47)
    ylims!(ax, ybot...)
end

# limits must be frozen on EVERY axis before this: text_fignum! draws a
# background box, and on an axis still using autolimits that box expands the
# y-range (it silently blew the bottom panels up to 0-10 before ylims! was set)
for (ax, lb) in ((ax_a,"(a)"), (ax_b,"(b)"), (ax_c,"(c)"), (ax_d,"(d)"))
    text_fignum!(ax, lb; fs=10, bckclr=:white)
end

colgap!(fig.layout, 8)
rowgap!(fig.layout, 8)

display(fig)
if figflag==1
    fout = string(dirfig, "CGE_signconditioned_$(GMSER)_ppr.png")
    savefig300(fout, fig)
    println("saved ", fout, "  (", size(fig.scene)[1], " x ", size(fig.scene)[2],
            " pt -> ", round(Int, 18*300/2.54), " px, tagged 300 dpi)")
end

## --- numbers behind the figure ----------------------------------------------
println("\n", "="^88)
println("sign-conditioned transect means, mainnm=$(GMSER) (regions set by the TIDE-ONLY run)")
println("="^88)
println(rpad("lat",7), rpad("frac(Pi>0)",12), rpad("<Pi>+tide",13), rpad("<Pi>+both",13),
        rpad("ratP",10), rpad("<Pi>-tide",13), rpad("<Pi>-both",13), "ratN")
for i in 1:nlat
    println(rpad(@sprintf("%.1f",LATS[i]),7), rpad(@sprintf("%.2f",fracP[i]),12),
        rpad(@sprintf("%+.3e",mP_tide[i]),13), rpad(@sprintf("%+.3e",mP_both[i]),13),
        rpad(@sprintf("%+.3f",ratP[i]),10),
        rpad(@sprintf("%+.3e",mN_tide[i]),13), rpad(@sprintf("%+.3e",mN_both[i]),13),
        @sprintf("%+.3f",ratN[i]))
end

println("\n", "="^88)
println("area-weighted contributions actually plotted in the top row (f x <Pi>)")
println("="^88)
println(rpad("lat",7), rpad("f+",7), rpad("f-",7),
        rpad("f+<Pi>+tide",14), rpad("f+<Pi>+both",14),
        rpad("f-<Pi>-tide",14), rpad("f-<Pi>-both",14), rpad("net tide",12), "net both")
for i in 1:nlat
    println(rpad(@sprintf("%.1f",LATS[i]),7), rpad(@sprintf("%.3f",fracP[i]),7),
        rpad(@sprintf("%.3f",fracN[i]),7),
        rpad(@sprintf("%+.3e",wP_tide[i]),14), rpad(@sprintf("%+.3e",wP_both[i]),14),
        rpad(@sprintf("%+.3e",wN_tide[i]),14), rpad(@sprintf("%+.3e",wN_both[i]),14),
        rpad(@sprintf("%+.3e",wP_tide[i]+wN_tide[i]),12),
        @sprintf("%+.3e",wP_both[i]+wN_both[i]))
end
@printf("\n(a)+(b) reproduces the net transect mean to %.1e of it\n",
    maximum(abs.((wP_tide.+wN_tide) .- [mean(C11[i,:]) for i in 1:nlat])))
println(@sprintf("\ntop-panel shared y-range: %.2f to %.2f  [10^-6 W/kg m]", ytop[1], ytop[2]))
println(@sprintf("bottom-panel shared y-range: %.2f to %.2f  [-]", ybot[1], ybot[2]))
println(@sprintf("lat=0 in panel (d) is plotted but runs off the top: only %.0f%% of that",
    100*(1-fracP[1])), @sprintf(" transect has Pi<0, <Pi>- = %.2e, ratN = %+.2f",
    mN_tide[1], ratN[1]))
