#= IW_coarsegr_GM_diff.jl
Maarten Buijsman, USM DMS, 2026-9-6 (revised 2026-9-15)

Difference heatmaps (x vs latitude) testing whether GM's contribution to the
cross-scale energy transfer (CGE, from coarse-graining) is simply ADDITIVE on
top of the tide's own transfer, for BOTH GM series side-by-side with a SHARED
colour axis per row so the two columns are directly comparable:
  - column 1: mainnm=13 GM blocks (original u,v-only GM initialization)
  - column 2: mainnm=15 GM blocks (w,b-consistent polarization-relation IC +
              redistribution fix for the domain-length cutoff)

Per GM series (mainnm = 13 or 15):
  - <m>.i     (i=1:13):  GM only, no tide   (Flux=0)
  - <m>.26+i (i=1:13):  GM + D2 tide       (F=25kW/m)
and the common tide-only baseline, identical for both:
  - 11.26+i  (i=1:13):  D2 tide only, no GM (F=25kW/m, same lat/flux)
(runnm-to-latitude mapping verified identical across all blocks.)

Step 1: "de-GM" the GM+tide run by subtracting the GM-only run's own transfer:
    diffA = CGE(GM+tide) - CGE(GM only)
(and the same subtraction on the cumulative CGEsum).

Step 2: compare that de-GM'd result against the pure-tide (no GM) run -- if
GM's effect were purely linear/additive, this should be ~zero:
    diffB = diffA - CGE(11, tide only)

Any structure remaining in diffB indicates a genuine NONLINEAR interaction
between the GM background field and the tide, not just independent
superposition.

REVISED 2026-9-15: both series are now plotted together with a shared colour
range per row (previously each figure auto-scaled to its own range, which made
the two series look similar even when their amplitudes differed).

Same smoothing convention as IW_analysis_coarsegr_2000km.jl: Gaussian smoothing
(Lsmooth=1600m) applied to CGE on the 200m grid before any further processing,
since all blocks here are on the 200m grid.
=#

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, ColorSchemes

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1

const LAT13   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
const Lsmooth = 1600.0   # same Gaussian sigma [m] as IW_analysis_coarsegr_2000km.jl
const LdomH   = 2000e3
const fcH     = 1e5      # scale Π to match the panel (c)-style units used before
const fc5H    = 1

# loads CGE (smoothed) and its cumulative integral CGEsum for one (mainnm,runnms) block
function load_CGE(mainnm, runnms)
    fnames0 = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "Etran_", fnames0, ".jld2") xc
    dx = xc[2] - xc[1]
    CGE = zeros(length(runnms), length(xc))
    CGEsum = copy(CGE)
    for (i, runnm) in enumerate(runnms)
        fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
        @load string(dirout, "Etran_", fnames, ".jld2") Πnhxa Πxxa Πzxa
        pietot = Πnhxa .+ Πxxa .+ Πzxa
        if dx < 500
            pietot = gaussfilt(xc, pietot, Lsmooth)
        end
        CGE[i, :] = pietot
        CGEsum[i, :] = cumtrapz(xc, CGE[i, :])
    end
    return xc, CGE, CGEsum
end

SERIES = [13, 15]
LABEL_SERIES = ["mainnm=13 (original GM IC)", "mainnm=15 (w,b + redistribution GM IC)"]

# common tide-only baseline
xc, CGE_11tide, CGEsum_11tide = load_CGE(11, collect(27:39))

diffA_CGE    = Vector{Any}(undef, length(SERIES))
diffA_CGEsum = Vector{Any}(undef, length(SERIES))
diffB_CGE    = Vector{Any}(undef, length(SERIES))
diffB_CGEsum = Vector{Any}(undef, length(SERIES))
CGE_gm_s     = Vector{Any}(undef, length(SERIES))   # kept for the transect-mean summary
CGE_tide_s   = Vector{Any}(undef, length(SERIES))

for (s, m) in enumerate(SERIES)
    _, CGE_gm,   CGEsum_gm   = load_CGE(m, collect(1:13))     # GM only, no tide
    _, CGE_tide, CGEsum_tide = load_CGE(m, collect(27:39))    # GM + tide
    CGE_gm_s[s]   = CGE_gm
    CGE_tide_s[s] = CGE_tide

    # ---- step 1: de-GM the GM+tide run -------------------------------------
    diffA_CGE[s]    = CGE_tide    .- CGE_gm
    diffA_CGEsum[s] = CGEsum_tide .- CGEsum_gm

    # ---- step 2: compare de-GM'd result against the pure-tide run ----------
    diffB_CGE[s]    = diffA_CGE[s]    .- CGE_11tide
    diffB_CGEsum[s] = diffA_CGEsum[s] .- CGEsum_11tide
end

# plots both series side-by-side, shared colour range per row across columns
function plot_diff_compare(CGEd, CGEsumd, titletxt, fname_out)
    cmaxH  = maximum(maximum(abs.(m)) for m in CGEd)    * fcH
    cmaxsH = maximum(maximum(abs.(m)) for m in CGEsumd) * fc5H

    fig = Figure(size=(1050, 760), fontsize=11)
    axs = Matrix{Axis}(undef, 2, length(SERIES))
    local hm1, hm2
    for s in 1:length(SERIES)
        axs[1,s] = Axis(fig[1,s], title=LABEL_SERIES[s], titlesize=11,
            ylabel = s==1 ? "latitude [°]" : "", yticklabelsvisible = s==1,
            xticklabelsvisible = false)
        axs[2,s] = Axis(fig[2,s], xlabel="x [km]",
            ylabel = s==1 ? "latitude [°]" : "", yticklabelsvisible = s==1)

        hm1 = heatmap!(axs[1,s], xc/1e3, LAT13, CGEd[s]'*fcH,
            colormap=Reverse(:RdBu_5), colorrange=(-cmaxH, cmaxH))
        hm2 = heatmap!(axs[2,s], xc/1e3, LAT13, CGEsumd[s]'*fc5H,
            colormap=Reverse(:RdBu_5), colorrange=(-cmaxsH, cmaxsH))

        for r in 1:2
            xlims!(axs[r,s], 0, LdomH/1e3)
        end
    end
    Label(fig[0,1:length(SERIES)], titletxt, fontsize=12)
    Label(fig[1,0], "(a) ΔΠ", rotation=π/2, tellheight=false, fontsize=12)
    Label(fig[2,0], "(b) cumulative ΔΠ", rotation=π/2, tellheight=false, fontsize=12)
    Colorbar(fig[1,length(SERIES)+1], hm1, label=@sprintf("ΔΠ [%.0e W/kg m]", 1/fcH))
    Colorbar(fig[2,length(SERIES)+1], hm2, label="Σ ΔΠ dx [W/kg m2]")
    display(fig)
    if figflag==1; savefig300(string(dirfig,fname_out), fig); end

    for (s, m) in enumerate(SERIES)
        println(titletxt, " [mainnm=", m, "]: ΔΠ min/max = ",
            @sprintf("%.2e / %.2e", minimum(CGEd[s]), maximum(CGEd[s])),
            "; ΣΔΠ min/max = ",
            @sprintf("%.2e / %.2e", minimum(CGEsumd[s]), maximum(CGEsumd[s])))
    end
end

plot_diff_compare(diffA_CGE, diffA_CGEsum,
    "(GM+tide) minus (GM only)  =  de-GM'd cross-scale transfer",
    "CGE_diff_tide_minus_gm_13vs15.png")

plot_diff_compare(diffB_CGE, diffB_CGEsum,
    "[de-GM'd] minus 11.27-39 (tide only)  =  nonlinear GM-tide interaction",
    "CGE_diff_deGM_minus_11tide_13vs15.png")


## ============================================================================
## Transect-mean summary: the nonlinear GM-tide residual as a fraction of the
## tide's own cross-scale transfer, vs latitude
##     (<Π>_gm+tide - <Π>_gm - <Π>_tide) / <Π>_tide
## where <·> is the transect (x) mean. Taken as a ratio of x-MEANS rather than
## the x-mean of a pointwise ratio: Π is a signed quantity with zero crossings,
## so a pointwise Π_residual/Π_tide diverges wherever the denominator crosses
## zero (same pathology that made the old KEt-percentage panel unreadable).
## The transect mean is well posed -- <Π> = (1/L)∫Π dx = ΣΠdx(L)/L -- so this
## reads as "what fraction of the tide's net downscale transfer is added (or
## removed) by the nonlinear interaction with the GM background".
mPi_tide = vec(mean(CGE_11tide, dims=2))                         # <Π> tide only
mPi_gm   = [vec(mean(CGE_gm_s[s],   dims=2)) for s in 1:length(SERIES)]
mPi_both = [vec(mean(CGE_tide_s[s], dims=2)) for s in 1:length(SERIES)]
mPi_res  = [mPi_both[s] .- mPi_gm[s] .- mPi_tide for s in 1:length(SERIES)]
ratio_res = [mPi_res[s] ./ mPi_tide for s in 1:length(SERIES)]

figT = Figure(size=(1000, 420), fontsize=11)
colsS = [:steelblue, :crimson]
mrkS  = [:circle, :rect]

axT1 = Axis(figT[1,1], title="(a) transect-mean Π components",
    xlabel="latitude [°]", ylabel="⟨Π⟩ [W/kg m]")
hlines!(axT1, [0.0], color=:gray, linestyle=:dash)
lines!(axT1, LAT13, mPi_tide, color=:black, linewidth=2.5, label="tide only (11.27-39)")
for s in 1:length(SERIES)
    scatterlines!(axT1, LAT13, mPi_both[s], color=colsS[s], marker=mrkS[s], markersize=9,
        linewidth=1.5, label=string("GM+tide (",SERIES[s],".27-39)"))
    scatterlines!(axT1, LAT13, mPi_gm[s], color=colsS[s], marker=mrkS[s], markersize=9,
        linewidth=1.5, linestyle=:dot, label=string("GM only (",SERIES[s],".1-13)"))
end
axislegend(axT1, position=:rt, framevisible=false, labelsize=8)

# log y-axis: the ratio goes negative (45°N, series 15) and dips well below the
# plotted range at several latitudes where the interaction is weak, so mask
# non-positive values to NaN -- scatterlines simply breaks the line there
yticks_b = [0.1, 1, 2, 4, 10, 20, 30, 40]
ratio_pos = [[v > 0 ? v : NaN for v in ratio_res[s]] for s in 1:length(SERIES)]

axT2 = Axis(figT[1,2], title="(b) (⟨Π⟩gm+tide − ⟨Π⟩gm − ⟨Π⟩tide) / ⟨Π⟩tide",
    xlabel="latitude [°]", ylabel="nonlinear residual / tide [-]",
    yscale=log10, yticks=(yticks_b, ["0.1","1","2","4","10","20","30","40"]))
vlines!(axT2, [28.8], color=:gray, linestyle=:dot)
for s in 1:length(SERIES)
    scatterlines!(axT2, LAT13, ratio_pos[s], color=colsS[s], marker=mrkS[s], markersize=10,
        linewidth=2, label=string("mainnm=",SERIES[s]))
end
ylims!(axT2, 0.006, 55)   # low enough that every positive point is visible; only negatives are masked
axislegend(axT2, position=:lt, framevisible=false)

for s in 1:length(SERIES)
    hidden = [(LAT13[i], ratio_res[s][i]) for i in 1:length(LAT13) if !(ratio_res[s][i] > 0.006)]
    println("mainnm=", SERIES[s], ": points below the log axis (lat, ratio): ",
        join([@sprintf("(%.1f, %+.3f)", h[1], h[2]) for h in hidden], " "))
end


## ============================================================================
## Sign-conditioned transect means. Π oscillates in x -- it has alternating
## down-scale (Π>0) and up-scale (Π<0) lobes -- so a plain transect mean lets
## the two partially cancel, and where the lobes happen to fall in x then
## controls the latitude dependence (the 15°N/20°N dip-peak pair above is
## suspect for exactly this reason). Here the transect is split into
## positive- and negative-Π regions defined by the TIDE-ONLY run, and each
## region is averaged separately. The mask is a property of the tide alone and
## is applied identically to all three runs, so it introduces no bias between
## them; conditioning on each run's own sign would.
##
## Ratios keep the same form as the unconditioned version:
##     (⟨Π⟩gm+tide − ⟨Π⟩gm − ⟨Π⟩tide) / ⟨Π⟩tide
## evaluated within each region. In the negative region ⟨Π⟩tide < 0, so a
## POSITIVE ratio there still means "the interaction reinforces the tide's own
## transfer" (here, strengthens up-scale transfer) -- same reading as in the
## positive region.
nlat  = length(LAT13)
maskP = [CGE_11tide[i, :] .> 0 for i in 1:nlat]
maskN = [CGE_11tide[i, :] .< 0 for i in 1:nlat]

condmean(A, mask) = [mean(A[i, mask[i]]) for i in 1:size(A, 1)]

mP_tide = condmean(CGE_11tide, maskP);  mN_tide = condmean(CGE_11tide, maskN)
mP_gm   = [condmean(CGE_gm_s[s],   maskP) for s in 1:length(SERIES)]
mN_gm   = [condmean(CGE_gm_s[s],   maskN) for s in 1:length(SERIES)]
mP_both = [condmean(CGE_tide_s[s], maskP) for s in 1:length(SERIES)]
mN_both = [condmean(CGE_tide_s[s], maskN) for s in 1:length(SERIES)]

resP  = [mP_both[s] .- mP_gm[s] .- mP_tide for s in 1:length(SERIES)]
resN  = [mN_both[s] .- mN_gm[s] .- mN_tide for s in 1:length(SERIES)]
ratP  = [resP[s] ./ mP_tide for s in 1:length(SERIES)]
ratN  = [resN[s] ./ mN_tide for s in 1:length(SERIES)]

fracP = [count(maskP[i]) / length(maskP[i]) for i in 1:nlat]   # transect fraction with Π>0

figS = Figure(size=(1100, 800), fontsize=11)

# --- row 1: positive-Π (down-scale) regions ---------------------------------
axP1 = Axis(figS[1,1], title="(a) ⟨Π⟩ over tide-only Π>0 regions (down-scale)",
    ylabel="⟨Π⟩₊ [W/kg m]", xticklabelsvisible=false)
lines!(axP1, LAT13, mP_tide, color=:black, linewidth=2.5, label="tide only")
for s in 1:length(SERIES)
    scatterlines!(axP1, LAT13, mP_both[s], color=colsS[s], marker=mrkS[s], markersize=9,
        linewidth=1.5, label=string("GM+tide (",SERIES[s],")"))
    scatterlines!(axP1, LAT13, mP_gm[s], color=colsS[s], marker=mrkS[s], markersize=9,
        linewidth=1.5, linestyle=:dot, label=string("GM only (",SERIES[s],")"))
end
axislegend(axP1, position=:rt, framevisible=false, labelsize=8)

# linear (not log) y-axis on the ratio panels: once the cancellation between
# the Π>0 and Π<0 lobes is removed, the residual ratios are O(0.1-1) and
# predominantly NEGATIVE, so a log axis would mask almost every point
axP2 = Axis(figS[1,2], title="(b) (⟨Π⟩gm+tide − ⟨Π⟩gm − ⟨Π⟩tide) / ⟨Π⟩tide, Π>0 regions",
    ylabel="residual / tide [-]", xticklabelsvisible=false)
hlines!(axP2, [0.0], color=:gray, linestyle=:dash)
vlines!(axP2, [28.8], color=:gray, linestyle=:dot)
for s in 1:length(SERIES)
    scatterlines!(axP2, LAT13, ratP[s], color=colsS[s],
        marker=mrkS[s], markersize=10, linewidth=2, label=string("mainnm=",SERIES[s]))
end
axislegend(axP2, position=:lt, framevisible=false)

# --- row 2: negative-Π (up-scale) regions -----------------------------------
axN1 = Axis(figS[2,1], title="(c) ⟨Π⟩ over tide-only Π<0 regions (up-scale)",
    xlabel="latitude [°]", ylabel="⟨Π⟩₋ [W/kg m]")
lines!(axN1, LAT13, mN_tide, color=:black, linewidth=2.5, label="tide only")
for s in 1:length(SERIES)
    scatterlines!(axN1, LAT13, mN_both[s], color=colsS[s], marker=mrkS[s], markersize=9,
        linewidth=1.5, label=string("GM+tide (",SERIES[s],")"))
    scatterlines!(axN1, LAT13, mN_gm[s], color=colsS[s], marker=mrkS[s], markersize=9,
        linewidth=1.5, linestyle=:dot, label=string("GM only (",SERIES[s],")"))
end
axislegend(axN1, position=:rb, framevisible=false, labelsize=8)

axN2 = Axis(figS[2,2], title="(d) same ratio, Π<0 regions",
    xlabel="latitude [°]", ylabel="residual / tide [-]")
hlines!(axN2, [0.0], color=:gray, linestyle=:dash)
vlines!(axN2, [28.8], color=:gray, linestyle=:dot)
for s in 1:length(SERIES)
    scatterlines!(axN2, LAT13, ratN[s], color=colsS[s],
        marker=mrkS[s], markersize=10, linewidth=2, label=string("mainnm=",SERIES[s]))
end
# lat=0 is off-scale (+3.4/+5.9) and unreliable: only ~6% of the transect there
# has Π<0, so ⟨Π⟩₋ is a small-sample mean of a near-zero quantity
ylims!(axN2, -1.6, 1.0)
axislegend(axN2, position=:rt, framevisible=false)

display(figS)
if figflag==1; savefig300(string(dirfig,"CGE_signconditioned_residual_13vs15.png"), figS); end

println("\n", "="^104)
println("sign-conditioned transect means (regions set by the TIDE-ONLY run) and residual/tide ratios")
println("="^104)
println(rpad("lat",7), rpad("frac(Pi>0)",12),
        rpad("<Pi>+tide",12), rpad("ratP(13)",11), rpad("ratP(15)",11),
        rpad("<Pi>-tide",12), rpad("ratN(13)",11), "ratN(15)")
for i in 1:nlat
    println(rpad(@sprintf("%.1f",LAT13[i]),7), rpad(@sprintf("%.2f",fracP[i]),12),
        rpad(@sprintf("%+.3e",mP_tide[i]),12), rpad(@sprintf("%+.3f",ratP[1][i]),11), rpad(@sprintf("%+.3f",ratP[2][i]),11),
        rpad(@sprintf("%+.3e",mN_tide[i]),12), rpad(@sprintf("%+.3f",ratN[1][i]),11), @sprintf("%+.3f",ratN[2][i]))
end
println("\nnote: at lat=0 only ", @sprintf("%.0f%%", 100*(1-fracP[1])),
    " of the transect has Π<0, so ⟨Π⟩₋ there is a small-sample mean of a",
    " near-zero quantity -- ratN(0°) = ",
    @sprintf("%+.2f / %+.2f", ratN[1][1], ratN[2][1]),
    " is unreliable and is plotted off-scale in panel (d).")

display(figT)
if figflag==1; savefig300(string(dirfig,"CGE_transectmean_residual_13vs15.png"), figT); end

println("\n", "="^92)
println("transect-mean Π and the nonlinear residual as a fraction of the tide's own transfer")
println("="^92)
println(rpad("lat",7), rpad("<Pi>tide",13),
        rpad("<Pi>gm(13)",13), rpad("<Pi>both(13)",14), rpad("resid/tide(13)",16),
        rpad("<Pi>gm(15)",13), rpad("<Pi>both(15)",14), "resid/tide(15)")
for i in 1:length(LAT13)
    println(rpad(@sprintf("%.1f",LAT13[i]),7), rpad(@sprintf("%+.3e",mPi_tide[i]),13),
        rpad(@sprintf("%+.3e",mPi_gm[1][i]),13), rpad(@sprintf("%+.3e",mPi_both[1][i]),14),
        rpad(@sprintf("%+.3f",ratio_res[1][i]),16),
        rpad(@sprintf("%+.3e",mPi_gm[2][i]),13), rpad(@sprintf("%+.3e",mPi_both[2][i]),14),
        @sprintf("%+.3f",ratio_res[2][i]))
end
