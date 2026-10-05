#= IW_KEt_GM_diff.jl
Maarten Buijsman, USM DMS, 2026-9-6 (revised 2026-9-15)

Difference heatmaps (x vs latitude) of tidal-band KE (KEt) testing whether the
GM background's effect on the tide is purely additive, for BOTH GM series
side-by-side with a SHARED colour axis per row so the two columns are directly
comparable:
  - column 1: mainnm=13 GM blocks (original u,v-only GM initialization)
  - column 2: mainnm=15 GM blocks (w,b-consistent polarization-relation IC +
              redistribution fix for the domain-length cutoff)
Tide-only baseline (11.27-39) is the same for both -- it has no GM component.

Two-step "de-GM" comparison, same logic as IW_coarsegr_GM_diff.jl's CGE version:
  Step 1: de-GM the GM+tide run:
      KEt_deGM = KEt(GM+tide) - KEt(GM only)
  Step 2: compare the de-GM'd tidal-band energy against the pure-tide run:
      ΔKEt = KEt_deGM - KEt(11, tide only)

REVISED 2026-9-15: the second panel used to show ΔKEt as a percentage of
KEt(11, tide only). That normalization is ill-behaved -- KEt(11,·) is itself a
strong function of x (it decays along the transect and has interference nodes),
so dividing by it blows the percentage up wherever the denominator is small
(unclipped range reached +1200%), which says more about the denominator than
about the GM-tide interaction. It is now normalized instead by KEtmax, the
THEORETICAL linear depth-integrated mode-1 KE of the parent forced wave
(F=E*Cg with the KE/APE polarization split) -- one constant per latitude, not a
function of x -- so the panel reads directly as "what fraction of the parent
wave's linear energy does the GM-tide interaction move around". Same
mode1_KEtmax() formula as IW_analysis_energy_2000km_ppr.jl / the KE half of
mode1_theory() in IW_mode1_theory_vs_lat.jl.

Caveat worth flagging: KEt(GM only) is NOT a real tidal signal -- per the
normalized-spectrum finding, it's the GM background's own near-inertial peak
drifting into the fixed tidal frequency BAND as f approaches the M2 frequency
near/above the critical latitude (an incidental frequency-band overlap, not a
real tide-GM interaction). So step-1 subtraction removes that band-overlap
artifact where it's large (mid/high latitude) but is a near-zero (harmless)
correction elsewhere.
=#

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, ColorSchemes

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1
const LAT13 = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]
const fcKE  = 1e-3               # J/m^2 -> kJ/m^2
const T2    = 12 + 25.2/60
const rho0  = 1020
const ω     = 2π/(T2*3600)
const LdomH = 2000e3

function load_KEt(mainnm, runnms)
    fnames0 = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "energetics_", fnames0, ".jld2") xc
    KEtr = zeros(length(runnms), length(xc))
    for (i, runnm) in enumerate(runnms)
        fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
        @load string(dirout, "energetics_", fnames, ".jld2") KEt
        KEtr[i, :] = KEt
    end
    return xc, KEtr
end

# theoretical linear depth-integrated mode-1 KE of the parent forced wave
# [J/m^2] -- KE half of mode1_theory() (IW_mode1_theory_vs_lat.jl), identical
# to mode1_KEtmax() in IW_analysis_energy_2000km_ppr.jl
function mode1_KEtmax(row)
    Fx     = row.Flux
    fcor   = coriolis(row.lat)
    nonhyd = row.DX < 500 ? 1 : 0

    fnamegrid = n2_filename(row)
    @load string(dirforce, fnamegrid) N2w zfw

    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
        sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)

    Etot = Fx / Cgn[1]                  # F = E*Cg
    fw2  = (fcor/ω)^2
    rat  = (1-fw2) / (1+fw2)            # APE/KE, from IW_Energy_scenarios.jl
    return Etot / (1+rat)
end

# de-GM'd tidal-band KE difference for one GM series, against the common
# 11.27-39 tide-only baseline
function deGM_diff_KEt(gmmain)
    xc, KEt_tide = load_KEt(gmmain, collect(27:39))   # GM + D2 tide
    _,  KEt_gm   = load_KEt(gmmain, collect(1:13))    # GM only, no tide
    _,  KEt_11   = load_KEt(11,     collect(27:39))   # D2 tide only, no GM
    return xc, (KEt_tide .- KEt_gm) .- KEt_11, KEt_gm, KEt_11
end

SERIES = [13, 15]
LABELS = ["13.27-39 de-GM'd − 11.27-39\n(original GM IC)",
          "15.27-39 de-GM'd − 11.27-39\n(w,b + redistribution GM IC)"]

xcs   = Vector{Any}(undef, length(SERIES))
dKEt  = Vector{Any}(undef, length(SERIES))   # [J/m^2]
KEtgm = Vector{Any}(undef, length(SERIES))
for (s, m) in enumerate(SERIES)
    xcs[s], dKEt[s], KEtgm[s], _ = deGM_diff_KEt(m)
end

# KEtmax is set by the forced tide (F=25 kW/m, DX=200m, zonalmean N2), which is
# identical for the 13.27-39, 15.27-39 and 11.27-39 blocks -> one vector for all
KEtmax = [mode1_KEtmax(row) for row in get_runs(11, collect(27:39))]   # [J/m^2]

dKEtn = [d ./ KEtmax for d in dKEt]      # ΔKEt / KEtmax [-]

# ---- shared colour axes across BOTH series (per row) -----------------------
cmaxD = maximum(maximum(abs.(d)) for d in dKEt)  * fcKE   # [kJ/m^2]
cmaxN = maximum(maximum(abs.(d)) for d in dKEtn)          # [-]

function plot_KEt_compare(fname_out)
    fig = Figure(size=(1050, 760), fontsize=11)
    axs = Matrix{Axis}(undef, 2, length(SERIES))
    local hm1, hm2
    for s in 1:length(SERIES)
        axs[1,s] = Axis(fig[1,s], title=LABELS[s], titlesize=11,
            ylabel = s==1 ? "latitude [°]" : "", yticklabelsvisible = s==1,
            xticklabelsvisible = false)
        axs[2,s] = Axis(fig[2,s], xlabel="x [km]",
            ylabel = s==1 ? "latitude [°]" : "", yticklabelsvisible = s==1)

        hm1 = heatmap!(axs[1,s], xcs[s]/1e3, LAT13, (dKEt[s]*fcKE)',
            colormap=Reverse(:RdBu_5), colorrange=(-cmaxD, cmaxD))
        hm2 = heatmap!(axs[2,s], xcs[s]/1e3, LAT13, dKEtn[s]',
            colormap=Reverse(:RdBu_5), colorrange=(-cmaxN, cmaxN))

        for r in 1:2
            xlims!(axs[r,s], 0, LdomH/1e3)
        end
    end
    Label(fig[1,0], "(a) ΔKEt", rotation=π/2, tellheight=false, fontsize=12)
    Label(fig[2,0], "(b) ΔKEt / KEtmax", rotation=π/2, tellheight=false, fontsize=12)
    Colorbar(fig[1,length(SERIES)+1], hm1, label="ΔKEt [kJ/m²]")
    Colorbar(fig[2,length(SERIES)+1], hm2, label="ΔKEt / KEtmax [-]")
    display(fig)
    if figflag==1; savefig300(string(dirfig,fname_out), fig); end
    return fig
end

plot_KEt_compare("KEt_diff_deGM_minus_11tide_13vs15.png")

# ---- printout ---------------------------------------------------------------
println("\nKEtmax (theoretical linear mode-1 KE of parent wave) [kJ/m²]:")
for (i,lat) in enumerate(LAT13)
    println(@sprintf("  lat=%5.1f: %8.3f", lat, KEtmax[i]*fcKE))
end
for (s, m) in enumerate(SERIES)
    println("\n--- series ", m, " ---")
    println("  ΔKEt          min/max = ", @sprintf("%.2e / %.2e", minimum(dKEt[s])*fcKE, maximum(dKEt[s])*fcKE), " kJ/m²")
    println("  ΔKEt/KEtmax   min/max = ", @sprintf("%.3f / %.3f", minimum(dKEtn[s]), maximum(dKEtn[s])))
    for (i,lat) in enumerate(LAT13)
        println(@sprintf("  lat=%5.1f: mean ΔKEt/KEtmax = %7.3f   (GM-only KEt mean = %.2e J/m²)",
            lat, mean(dKEtn[s][i,:]), mean(KEtgm[s][i,:])))
    end
end
