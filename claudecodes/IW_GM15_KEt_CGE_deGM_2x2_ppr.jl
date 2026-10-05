#= IW_GM15_KEt_CGE_deGM_2x2_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: 2 x 2 x-vs-latitude heatmaps for the GM + D2 tide series with
the redistribution-fix GM IC (mainnm = 15, F = 25 kW/m, 200 m NH, zonalmean N2),
lat = 0-45 N only -- runnm 27-38; 15.39 (lat 50) is excluded, and so are its
partners 15.13 and 11.39.

            left column                       right column
            15.27-38 itself (GM + tide)       nonlinear GM-tide residual
  row 1     (a) KEt          [kJ/m2]          (b) dKEt          [kJ/m2]
  row 2     (c) rho0*Pi      [mW/m2]          (d) rho0*dPi      [mW/m2]

The right column is the top row of the older KEt_diff_deGM_minus_11tide_15series.png
and CGE_diff_deGM_minus_11tide_15series.png (IW_KEt_GM_diff.jl, IW_coarsegr_GM_diff.jl),
with the same two-step "de-GM" difference:
    d(.) = [ (.)(15.27-38, GM + tide) - (.)(15.1-12, GM only) ] - (.)(11.27-38, tide only)
i.e. what is left of the GM + tide run after removing the GM background's own
contribution and the tide's own contribution; nonzero only where GM and tide
interact nonlinearly.

Pi is the depth-integrated cross-scale transfer Pinhxa + Pixxa + Pizxa from the
coarse-graining files, Gaussian-smoothed at sigma = 1600 m on the 200 m grid as
in IW_analysis_coarsegr_2000km.jl, and scaled by rho0 so it is a power density
[W/m2], shown in mW/m2 (same convention as IW_energy_flux_CGE_2col_ppr.jl).

Colour ranges: row 2 shares ONE symmetric range across both columns, so the
residual in (d) is read directly against the transfer itself in (c). Row 1
cannot share -- KEt is positive (sequential map) while dKEt is signed
(diverging map) -- so each panel has its own.

Layout: 18 x 14 cm, 10 pt. Every panel has its own vertical colour bar on its
right, so the columns are separated by a gap (Dsh > 0) wide enough for the
left column's bars; rows still share edges (Dsv = 0, inward ticks). Each bar
is inset a little top and bottom so the two bars in a column do not touch at
the row seam.
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

const rho0    = 1020.0
const Lsmooth = 1600.0          # Gaussian sigma [m], as IW_analysis_coarsegr_2000km.jl
const LdomH   = 2000e3
const fcKE    = 1e-3            # J/m2 -> kJ/m2
const fcPI    = rho0*1e3        # W/(kg m) -> rho0*Pi in mW/m2

# ---- run selection: lat 0-45 N only (15.39/15.13/11.39 = lat 50 excluded) ---
const GMSER = 16   # GM series: 16 (GM81 IC, 1x GM81 over days 10-20) or 15 (earlier ~3x GM IC; untagged output names)
const RN_TIDE = collect(27:38)  # GM + tide (GMSER) and tide only (11)
const RN_GM   = collect(1:12)   # GM only, no tide (GMSER), same latitudes
LATS = [r.lat for r in get_runs(GMSER, RN_TIDE)]
@assert LATS == [r.lat for r in get_runs(GMSER, RN_GM)] == [r.lat for r in get_runs(11, RN_TIDE)]

## load ---------------------------------------------------------------------
function load_KEt(mainnm, runnms)
    fn0 = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "energetics_", fn0, ".jld2") xc
    A = zeros(length(runnms), length(xc))
    for (i, rn) in enumerate(runnms)
        @load string(dirout, "energetics_", @sprintf("AMZexpt%02i.%02i", mainnm, rn), ".jld2") KEt
        A[i, :] = KEt
    end
    return xc, A
end

function load_Pi(mainnm, runnms)
    fn0 = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "Etran_", fn0, ".jld2") xc
    A = zeros(length(runnms), length(xc))
    for (i, rn) in enumerate(runnms)
        @load string(dirout, "Etran_", @sprintf("AMZexpt%02i.%02i", mainnm, rn), ".jld2") Πnhxa Πxxa Πzxa
        p = Πnhxa .+ Πxxa .+ Πzxa
        (xc[2] - xc[1]) < 500 && (p = gaussfilt(xc, p, Lsmooth))   # 200 m grid -> smooth
        A[i, :] = p
    end
    return xc, A
end

xe, KEt15  = load_KEt(GMSER, RN_TIDE)
_,  KEt15g = load_KEt(GMSER, RN_GM)
_,  KEt11  = load_KEt(11, RN_TIDE)
xp, Pi15   = load_Pi(GMSER, RN_TIDE)
_,  Pi15g  = load_Pi(GMSER, RN_GM)
_,  Pi11   = load_Pi(11, RN_TIDE)

KEtA = KEt15 .* fcKE                                   # (a)
dKEt = ((KEt15 .- KEt15g) .- KEt11) .* fcKE            # (b)
PiC  = Pi15 .* fcPI                                    # (c)
dPi  = ((Pi15 .- Pi15g) .- Pi11) .* fcPI               # (d)

## colour ranges --------------------------------------------------------------
cmaxA = maximum(KEtA)
cmaxB = maximum(abs.(dKEt))
cmaxP = max(maximum(abs.(PiC)), maximum(abs.(dPi)))    # shared across row 2

cmapKE  = :thermal
cmapDIV = Reverse(:RdBu_5)

## layout ---------------------------------------------------------------------
cm_to_pt = 72/2.54
fig_w = 18*cm_to_pt
fig_h = 14*cm_to_pt
# every colour-bar zone (gap + bar + tick labels + rotated label) is CBZONE pt
# wide: once between the columns (Dsh) and once in the right margin
const CBGAP, CBW, CBZONE = 6.0, 7.0, 48.0   # bars carry no label -- units are in the panel titles
const MARGL, MARGR = 50.0, CBZONE  # pt: left = lat tick labels + ylabel
const MARGT, MARGB = 18.0, 36.0    # pt: top = row-1 titles; bottom = x tick labels + xlabel
const DSVPT = 18.0                 # pt between rows, for the row-2 titles
pos = subplot_hor_vertpos(2, 2, MARGL/fig_w, MARGR/fig_w, MARGB/fig_h, MARGT/fig_h, CBZONE/fig_w, DSVPT/fig_h)
bb(i) = BBox(subplot_bbox(pos[i], fig_w, fig_h)...)
pidx(r, c) = (r - 1)*2 + c
const TICKSIZE = 4.0

fig = Figure(size = (fig_w, fig_h), fontsize = 10)

# columns no longer touch (Dsh > 0), so both columns label 2000
xtk(c) = (collect(0.0:500.0:2000.0), ["0","500","1000","1500","2000"])

# panel letter + quantity + units in the title; run details go in the caption
TITLES = ["(a) KEt [kJ/m²]"  "(b) ΔKEt [kJ/m²]";
          "(c) ρ₀Π [mW/m²]"  "(d) ρ₀ΔΠ [mW/m²]"]

axs = Matrix{Axis}(undef, 2, 2)
for r in 1:2, c in 1:2
    axs[r, c] = Axis(fig.scene, bbox = bb(pidx(r, c)),
        xtickalign = 1, ytickalign = 1, xticksize = TICKSIZE, yticksize = TICKSIZE,
        xticks = xtk(c), yticks = collect(0.0:10.0:40.0),
        xlabel = r == 2 ? "x [km]" : "", ylabel = c == 1 ? "latitude [°]" : "",
        xticklabelsvisible = r == 2, yticklabelsvisible = c == 1,
        xlabelvisible = r == 2, ylabelvisible = c == 1,
        title = TITLES[r, c], titlesize = 10, titlegap = 3)
end

hmA = heatmap!(axs[1,1], xe/1e3, LATS, KEtA', colormap = cmapKE,  colorrange = (0, cmaxA))
hmB = heatmap!(axs[1,2], xe/1e3, LATS, dKEt', colormap = cmapDIV, colorrange = (-cmaxB, cmaxB))
hmC = heatmap!(axs[2,1], xp/1e3, LATS, PiC',  colormap = cmapDIV, colorrange = (-cmaxP, cmaxP))
hmD = heatmap!(axs[2,2], xp/1e3, LATS, dPi',  colormap = cmapDIV, colorrange = (-cmaxP, cmaxP))

# an Axis parented to fig.scene does not autolimit itself (subplot_hor_vertpos.jl)
for r in 1:2, c in 1:2
    autolimits!(axs[r, c])
    xlims!(axs[r, c], 0, LdomH/1e3)
end



## vertical colour bars, right of each panel ----------------------------------
# colorbar_bbox (functions/subplot_hor_vertpos.jl): bar starts CBGAP pt right of
# the panel, is CBW pt wide, spans 90% of the panel height starting 5% up --
# the 5% inset top and bottom keeps a column's two bars (and their end tick
# labels) apart at the shared row seam
function cbar!(hm, c, row, label)
    cb = colorbar_bbox(pos[pidx(row, c)], fig_w, fig_h, CBGAP/fig_w, CBW/fig_w, 1.0, 0.0)
    Colorbar(fig.scene, hm, bbox = BBox(cb...), vertical = true,
             ticksize = 3, labelpadding = 2)
end
cbar!(hmA, 1, 1, "KEt [kJ/m²]")
cbar!(hmB, 2, 1, "ΔKEt [kJ/m²]")
cbar!(hmC, 1, 2, "ρ₀Π [mW/m²]")
cbar!(hmD, 2, 2, "ρ₀ΔΠ [mW/m²]")

display(fig)
if figflag == 1
    fout = string(dirfig, "KEt_CGE_GM$(GMSER)_deGM_2x2.png")
    savefig300(fout, fig)
    println("saved ", fout)
end

## numbers for the text -------------------------------------------------------
@printf("KEt(GM)   max = %.2f kJ/m2\n", cmaxA)
@printf("dKEt      min/max = %.2f / %.2f kJ/m2\n", minimum(dKEt), maximum(dKEt))
@printf("rho0*Pi   min/max = %.2f / %.2f mW/m2\n", minimum(PiC), maximum(PiC))
@printf("rho0*dPi  min/max = %.2f / %.2f mW/m2\n", minimum(dPi), maximum(dPi))
