#= IW_mode1_theory_vs_lat.jl
Maarten Buijsman, USM DMS, 2026-9-4 (split legend: first 2 blocks on KE, last 3 on F, positioned to avoid overlapping any line)

Linear mode-1 internal-tide theory (A0, KE, APE, F) vs latitude, for the 5
standard run blocks (mainnm=10 or 11), as a 2x2 grid (A0, KE, APE, F), each
panel holding all 5 run blocks as separate colored lines. Simulated
(measured) KEt/APEt/Fxt at x~100km are overlaid as circles of the same
color, from the energetics_AMZexptXX.YY.jld2 files IW_total_energetics_tile.jl
already saves per run (no measured A0 overlay yet).

Formulas, combined from two existing files rather than re-derived from
scratch:

  A0 (peak mode-1 vertical displacement) -- IDENTICAL formula/normalization
  as A0nlana in oceananigans_IW/IW_nondim_params.jl:
      A0 = sqrt(2F / (rho0 * Cn^2 * Cg * U2int))
      U2int = integral of (dW1n/dz)^2 dz, W1n = mode-1 W-eigenfunction
      normalized so its max |value| is 1 (Sutherland & Dhaliwal 2022 conv.)

  KE/APE split -- rather than re-deriving KE/APE directly from A0 using
  THIS file's own eigenfunction normalization (which would need reconciling
  against oceananigans_IW/IW_Energy_scenarios.jl's different Ueig2-based
  normalization -- risk of a subtle, hard-to-catch amplitude mismatch
  between the two conventions), use the convention-INDEPENDENT route
  instead: total depth-integrated energy E = F/Cg (the same basic flux
  relation A0 is already solved from), then split E into KE and APE using
  the polarization ratio from IW_Energy_scenarios.jl (a standard internal
  wave result that depends only on omega and f, not on how the
  eigenfunction happens to be normalized):
      KE/APE = (omega^2+f^2) / (omega^2-f^2)
      APE = E / (1+rat),  KE = E*rat / (1+rat),  rat = APE/KE = (1-fw2)/(1+fw2)

  F: just the run's own prescribed target flux (row.Flux from run_master.jl)
  -- constant across each block by construction, shown as a flat theory line
  for visual completeness even though it's not latitude-dependent physics.
=#

using Printf, CairoMakie, Statistics, JLD2, Trapz

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirforce  = string(pth0, "IW/forcingfiles/")
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

const T2   = 12 + 25.2/60
const rho0 = 1020.0
const ω    = 2π / (T2*3600)

# same Gaussian-source geometry as IW_A0nl_extract.jl / IW_nondim_params.jl,
# to sample the measured energy terms at the SAME cell as measured A0
const gausW_center    = 80_000   # m
const gausW_width     = 16_000   # m
const A0_offset_sigma = 2
const xA0 = gausW_center + A0_offset_sigma*gausW_width   # 112 km target (-> nearest cell x=110km on the 4km grid)

# ---- USER SETTING: which grid series -----------------------------------
mainnm = 10   # 10 = 4km hydrostatic (fast); 11 = 200m nonhydrostatic

# the 5 standard run blocks (see run_master.jl) -- excluding the lat=50
# transect (the last runnm in each block), so 12 sims per series instead of 13
blocks = [
    (collect(1:12),  "F=12.5 kW/m, varying N²"),
    (collect(27:38), "F=25 kW/m, varying N²"),
    (collect(40:51), "F=25 kW/m, fixed N², at 2.5°N"),
    (collect(53:64), "F=25 kW/m, fixed N², at 50°N"),
    (collect(66:77), "F=50 kW/m, varying N²"),
]

function mode1_theory(row)
    LAT    = row.lat
    Fx     = row.Flux
    fcor   = coriolis(LAT)
    nonhyd = row.DX < 500 ? 1 : 0

    fnamegrid = n2_filename(row)
    @load string(dirforce, fnamegrid) N2w zfw
    zc = (zfw[1:end-1] .+ zfw[2:end]) ./ 2

    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
        sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)
    Im = 1
    W1   = Weig[:, Im]
    imax = argmax(abs.(W1))
    W1n  = W1 ./ W1[imax]
    U1   = diff(W1n) ./ diff(zfw)
    U2int = trapz(zc, U1 .^ 2)

    A0 = sqrt(2*Fx / (rho0 * Cn[Im]^2 * Cgn[Im] * U2int))

    Etot = Fx / Cgn[Im]                 # F = E*Cg
    fw2  = (fcor/ω)^2
    rat  = (1-fw2) / (1+fw2)            # APE/KE, from IW_Energy_scenarios.jl
    KE   = Etot / (1+rat)
    APE  = Etot * rat / (1+rat)

    return A0, KE, APE, Fx
end

# simulated (measured) tidal-band KEt/APEt/Fxt, at the SAME single x-cell
# used for the measured A0 (x=110km -- the cell nearest xA0=gausW_center+
# 2*gausW_width=112km on this 4km grid, see IW_A0nl_extract.jl), rather than
# a different point/average -- from the energetics_AMZexptXX.YY.jld2 files
# IW_total_energetics_tile.jl already saves per run (xc in meters, KEt/APEt
# in J/m^2, Fxt in W/m)
xmeas = xA0
function measured_KEAPEF(mainnm, runnm)
    fname = string(dirout, @sprintf("energetics_AMZexpt%02i.%02i.jld2", mainnm, runnm))
    isfile(fname) || return NaN, NaN, NaN
    d = load(fname)
    i = argmin(abs.(d["xc"] .- xmeas))
    return d["KEt"][i], d["APEt"][i], d["Fxt"][i]
end

# measured A0 -- from a0nl_AMZexptXX.YY.jld2 (IW_A0nl_extract.jl), the
# measured companion to A0nlana, extracted at x=xA0=gausW_center+2*gausW_width
# ~112km (close to the 100km reference used for KEt/APEt/Fxt above)
function measured_A0(mainnm, runnm)
    fname = string(dirout, @sprintf("a0nl_AMZexpt%02i.%02i.jld2", mainnm, runnm))
    isfile(fname) || return NaN
    return load(fname, "A0nl")
end

nblocks = length(blocks)
# 2x2 grid: A0 | KE  //  APE | F -- each panel holds all 5 blocks as
# separate colored lines, instead of 5 rows x 3 cols
blockcolors = [:black, :red, :dodgerblue, :darkorange, :seagreen]

# paper size: 18x18 cm at fontsize 10pt. Makie's `size` is in points (72
# pt/inch); px_per_unit at save-time rasterizes to ~300 dpi for print while
# keeping the 18cm/10pt physical proportions (same convention as the
# 11cm/9pt used for the k-omega spectrum paper figures earlier)
cm_to_pt = 72/2.54
figE = Figure(size=(18*cm_to_pt, 18*cm_to_pt), fontsize=10)
ax_F   = Axis(figE[1,1], title="(a) F [kW/m]")
ax_A0  = Axis(figE[1,2], title="(b) A0 [m]")
ax_KE  = Axis(figE[2,1], title="(c) KE [kJ/m²]", xlabel="latitude [°]")
ax_APE = Axis(figE[2,2], title="(d) APE [kJ/m²]",xlabel="latitude [°]")

for (bi, (runnms, blabel)) in enumerate(blocks)
    rows = get_runs(mainnm, runnms)
    LATS = [r.lat for r in rows]
    n = length(rows)
    A0v, KEv, APEv, Fv = zeros(n), zeros(n), zeros(n), zeros(n)
    for (i, row) in enumerate(rows)
        A0v[i], KEv[i], APEv[i], Fv[i] = mode1_theory(row)
    end

    c = blockcolors[bi]
    label1 = @sprintf("%i.%i-%i: %s", mainnm, runnms[1], runnms[end], blabel)
    # split legend: first 2 blocks labeled on (c) KE, last 3 on (a) F --
    # avoids the legend box overlapping the green KE line
    lines!(ax_A0,  LATS, A0v,       color=c, linewidth=2)
    lines!(ax_KE,  LATS, KEv./1e3,  color=c, linewidth=2, label = bi<=2 ? label1 : nothing)
    lines!(ax_APE, LATS, APEv./1e3, color=c, linewidth=2)
    lines!(ax_F,   LATS, Fv./1e3,   color=c, linewidth=2, label = bi>=3 ? label1 : nothing)

    # simulated values, x~100km, same color circles
    KEtm, APEtm, Fxtm, A0m = zeros(n), zeros(n), zeros(n), zeros(n)
    for (i, runnm) in enumerate(runnms)
        KEtm[i], APEtm[i], Fxtm[i] = measured_KEAPEF(mainnm, runnm)
        A0m[i] = measured_A0(mainnm, runnm)
    end
    scatter!(ax_A0,  LATS, A0m,        color=c, markersize=10, strokecolor=:black, strokewidth=0.5)
    scatter!(ax_KE,  LATS, KEtm./1e3,  color=c, markersize=10, strokecolor=:black, strokewidth=0.5)
    scatter!(ax_APE, LATS, APEtm./1e3, color=c, markersize=10, strokecolor=:black, strokewidth=0.5)
    scatter!(ax_F,   LATS, Fxtm./1e3,  color=c, markersize=10, strokecolor=:black, strokewidth=0.5)
end

axislegend(ax_F,  position=(0.02, 0.62), labelsize=9, framevisible=false)   # in the gap between the 25 and 50 flat lines
axislegend(ax_KE, position=:lt, labelsize=9, framevisible=false)

display(figE)
save(string(dirfig, "mode1_theory_vs_lat_mainnm", mainnm, ".png"), figE; px_per_unit=300/72)
println("saved mode1_theory_vs_lat_mainnm", mainnm, ".png")
