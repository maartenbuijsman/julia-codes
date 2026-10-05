#= IW_nondim_vs_lat.jl
Maarten Buijsman, USM DMS, 2026-9-4

Non-dimensional mode-1 internal-tide parameters (Sutherland & Dhaliwal 2022
style) vs latitude, for the 5 standard run blocks (mainnm=10 or 11,
excluding the lat=50 transect -- 12 sims per series), as a 3x2 grid:
  left column:  (a) kH  (b) dnl/H  (c) f/N0
  right column: (d) alpha  (e) epsilon (omega-based) & epsilon_k together  (f) alpha/epsilon
Nonhydrostatic only throughout. epsilon = omega-based (epsnh); epsilon_k is
shown alongside it in the SAME panel (e) for comparison. alpha uses the
THEORETICAL/analytic A0 (A0nlana), not the measured A0nl. N0 = max(N(z)).

Formulas ported directly from oceananigans_IW/IW_nondim_params.jl (same
run_analysis logic) rather than re-derived, to stay consistent with that
file's validated methods -- in particular the e-folding dnl calculation
(peak-N2 depth, background-subtracted N2, 1/e crossing) and the Roots.jl
root-finds for the omega- and k-based epsilon.

Layout: panels are placed at EXPLICIT bbox positions via the new
functions/subplot_hor_vertpos.jl (ported from Maarten's MATLAB
subplot_hor_vertpos.m) instead of GridLayout row/col + rowsize!/rowgap! --
this sidesteps the Relative/Auto sizing fights and axis-protrusion/gap
surprises worked through by direct computedbbox inspection in the previous
version of this file (see git history), by controlling every panel's exact
position/size directly, same as the MATLAB workflow.
=#

using Printf, CairoMakie, Statistics, JLD2, Trapz, Roots

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirforce  = string(pth0, "IW/forcingfiles/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

const T2   = 12 + 25.2/60
const ω    = 2π / (T2*3600)
const rho0 = 1020.0

# ---- USER SETTING: which grid series -----------------------------------
mainnm = 11   # 10 = 4km hydrostatic (fast); 11 = 200m nonhydrostatic

# the 5 standard run blocks (see run_master.jl) -- excluding lat=50
blocks = [
    (collect(1:12),  "F=12.5 kW/m, varying N²"),
    (collect(27:38), "F=25 kW/m, varying N²"),
    (collect(40:51), "F=25 kW/m, fixed N², at 2.5°N"),
    (collect(53:64), "F=25 kW/m, fixed N², at 50°N"),
    (collect(66:77), "F=50 kW/m, varying N²"),
]

# ω-based epsilon: solve for the frequency at which mode-1 k doubles
# relative to k(ω) (Roots.jl find_zero, same as IW_nondim_params.jl)
function getomres(zfw, N2w, ω, fcor, nonhyd, Nm)
    kn1, = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)
    k1 = kn1[Nm]
    g(w) = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, w, nonhyd)[1][Nm] - 2*k1
    return find_zero(g, 2*ω)
end

# k-based epsilon: k(ω) and k(2ω) directly
function getkres(zfw, N2w, ω, fcor, nonhyd, Nm)
    kn, = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)
    k_k = kn[Nm]
    kn2, = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, 2ω, nonhyd)
    k2_k = kn2[Nm]
    return k_k, k2_k
end

function nondim_theory(row)
    LAT = row.lat
    Fx  = row.Flux
    fcor = coriolis(LAT)
    nonhyd = 1   # nonhydrostatic only, per Maarten

    fnamegrid = n2_filename(row)
    @load string(dirforce, fnamegrid) N2w zfw
    N2c = N2w[1:end-1]/2 + N2w[2:end]/2
    zc  = (zfw[1:end-1] .+ zfw[2:end]) ./ 2

    H  = maximum(zfw) - minimum(zfw)
    N0 = sqrt(maximum(N2w))
    fN0 = fcor / N0

    Im = 1
    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
        sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)
    k = kn[Im]
    kH = k*H

    W1   = Weig[:, Im]
    imax = argmax(abs.(W1))
    W1n  = W1 ./ W1[imax]
    U1   = diff(W1n) ./ diff(zfw)
    U2int = trapz(zc, U1 .^ 2)
    A0nlana = sqrt(2*Fx / (rho0 * Cn[Im]^2 * Cgn[Im] * U2int))

    # dnl: e-folding depth of the (background-subtracted) N2 profile below
    # the stratification peak -- identical method to IW_nondim_params.jl
    iord = sortperm(zc)
    zcs  = zc[iord]; N2cs = N2c[iord]
    I1s  = argmax(N2cs)
    z1   = zcs[I1s]
    deepfrac = 0.2
    zdeep_cutoff = zcs[1] + deepfrac*(zcs[end]-zcs[1])
    N2_deep = mean(N2cs[zcs .<= zdeep_cutoff])
    dN2  = N2cs .- N2_deep
    target = dN2[I1s] / ℯ
    Isub = 1:I1s
    zsub = zcs[Isub]; dN2sub = dN2[Isub]
    Icross = findlast(dN2sub .< target)
    if Icross === nothing
        z2 = zsub[1]
    else
        z2 = zsub[Icross] + (target-dN2sub[Icross])/(dN2sub[Icross+1]-dN2sub[Icross]) * (zsub[Icross+1]-zsub[Icross])
    end
    dnl = z1 - z2
    dnlH = dnl/H

    alpha = A0nlana / dnl

    omr = getomres(zfw, N2w, ω, fcor, nonhyd, Im)
    epsnh = ((2ω)^2 - omr^2) / (2ω)^2
    # (IW_nondim_params.jl's LAT==0 guard zeroes epshy there because the
    # HYDROSTATIC dispersion relation is exactly linear/non-dispersive at
    # f=0 -- that doesn't apply to epsnh, which is nonhydrostatic and has
    # genuine curvature regardless of rotation, so no guard needed here)

    k_k, k2_k = getkres(zfw, N2w, ω, fcor, nonhyd, Im)
    epsnh_k = (k2_k^2 - (2*k_k)^2) / (2*k_k)^2

    alpepsnh = alpha / epsnh
    isinf(alpepsnh) && (alpepsnh = NaN)

    return kH, dnlH, fN0, alpha, epsnh, epsnh_k, alpepsnh
end

blockcolors = [:black, :red, :dodgerblue, :darkorange, :seagreen]

# paper size: 18cm wide x 21cm tall, fontsize 10pt, ~300dpi raster at save
cm_to_pt = 72/2.54
fig_w = 18*cm_to_pt
fig_h = 21*cm_to_pt
figN = Figure(size=(fig_w, fig_h), fontsize=10, figure_padding=(18,18,10,4))

# explicit bbox layout: 2 cols x 3 rows of panels, bottom margin (vers --
# NOT vere, see functions/subplot_hor_vertpos.jl gotcha note) reserved for
# the shared legend below the grid. pos[] fills left-to-right then
# top-to-bottom: pos[1]=(row1,col1) pos[2]=(row1,col2) pos[3]=(row2,col1) ...
pos = subplot_hor_vertpos(2, 3, 0.09, 0.02, 0.1936, 0.045, 0.099, 0.066)
bb(i) = BBox(subplot_bbox(pos[i], fig_w, fig_h)...)

ax_kH    = Axis(figN.scene, bbox=bb(1), title="(a) kH")
ax_alpha = Axis(figN.scene, bbox=bb(2), title="(d) α")
ax_dnlH  = Axis(figN.scene, bbox=bb(3), title=rich("(b) d", subscript("N"), "/H"))
ax_eps   = Axis(figN.scene, bbox=bb(4), title="(e) ε (ω-based, k-based)")
ax_fN0   = Axis(figN.scene, bbox=bb(5), title="(c) f/N₀", xlabel="latitude [°]")
ax_alpeps= Axis(figN.scene, bbox=bb(6), title="(f) α/ε", xlabel="latitude [°]")

# uniform line thickness -- the earlier 3x/2x black/red thickening looked
# ugly; the fact that black/red/green overlap in several panels (kH, dnl/H,
# f/N0, epsilon only depend on N2/latitude, not on forcing flux) will just
# be noted in the figure caption instead
blocklw = [2, 2, 2, 2, 2]

lineplots = Any[]   # one representative line handle per block, for the shared legend
for (bi, (runnms, blabel)) in enumerate(blocks)
    rows = get_runs(mainnm, runnms)
    LATS = [r.lat for r in rows]
    n = length(rows)
    kHv, dnlHv, fN0v, alphav, epsv, epskv, alpepsv = (zeros(n) for _ in 1:7)
    for (i, row) in enumerate(rows)
        kHv[i], dnlHv[i], fN0v[i], alphav[i], epsv[i], epskv[i], alpepsv[i] = nondim_theory(row)
    end

    c  = blockcolors[bi]
    lw = blocklw[bi]
    lp = lines!(ax_kH,     LATS, kHv,     color=c, linewidth=lw)
    lines!(ax_dnlH,   LATS, dnlHv,   color=c, linewidth=lw)
    lines!(ax_fN0,    LATS, fN0v,    color=c, linewidth=lw)
    lines!(ax_alpha,  LATS, alphav,  color=c, linewidth=lw)
    lines!(ax_eps,    LATS, epsv,    color=c, linewidth=lw)
    lines!(ax_eps,    LATS, epskv,   color=c, linewidth=lw, linestyle=:dash)
    lines!(ax_alpeps, LATS, alpepsv, color=c, linewidth=lw)
    push!(lineplots, lp)
end

# NOTE: axes built via `Axis(fig.scene, bbox=...)` (i.e. NOT parented to the
# Figure's GridLayout) do not auto-update their limits from plotted data --
# confirmed directly (ax.finallimits stayed the Makie default (0,10)x(0,10)
# even after lines!() calls) -- so autolimits! must be called explicitly.
for a in (ax_kH, ax_dnlH, ax_fN0, ax_alpha, ax_eps, ax_alpeps)
    autolimits!(a)
end

# simple black solid/dashed key for panel (e), meaning applies regardless of
# color: solid = epsilon (omega-based), dashed = epsilon_k
lines!(ax_eps, [NaN], [NaN], color=:black, linewidth=2, label="ε")
lines!(ax_eps, [NaN], [NaN], color=:black, linewidth=2, linestyle=:dash, label="εₖ")
axislegend(ax_eps, position=:lt, labelsize=8, framevisible=false)

# shared legend for all 5 blocks, placed BELOW the whole grid (in the
# vere=0.14 bottom margin reserved above), 2 columns, also an explicit bbox
blocklabels = [@sprintf("%i.%i-%i: %s", mainnm, b[1][1], b[1][end], b[2]) for b in blocks]
leg_bbox = BBox(0.09*fig_w, 0.98*fig_w, 0.0*fig_h, 0.15*fig_h)
Legend(figN.scene, lineplots, blocklabels, bbox=leg_bbox, nbanks=2, labelsize=8, framevisible=false)

display(figN)
savefig300(string(dirfig, "nondim_vs_lat_mainnm", mainnm, ".png"), figN)
println("saved nondim_vs_lat_mainnm", mainnm, ".png")
