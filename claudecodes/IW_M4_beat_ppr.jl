#= IW_M4_beat_ppr.jl
Maarten Buijsman, USM DMS, 2026-9-17

Paper figure: simulated vs theoretical M4 beat wavelength, for two series that
differ only in their stratification (both D2 tide only, F=25 kW/m, DX=200 m):

  11.27-38  N2 varies with latitude (the Atlantic zonal-mean profile at each
            run's own latitude)
  11.53-64  N2 FIXED at the 50 deg N profile for every latitude, so the beat
            varies with f alone

The second series is the stronger test of the theory: the fixed 50 deg N
profile is much weaker, which shortens the predicted beat by a factor 1.2-2.5
(e.g. 751 -> 454 km at 25 deg N, 1084 -> 659 km at 20 deg N). The prediction
has to move by that factor and the simulations have to move with it. It also
buys one extra latitude -- the shorter beat puts 20 deg N inside the domain
(2.6 cycles) where the varying-N2 series only managed 1.6.

Beat wavelength against latitude -- the predicted bound-free n=1 beat
2*pi/|2*k1(M2) - k1(M4)| from the Sturm-Liouville eigenvalue problem, with the
spectral-peak measurements from the simulated surface M4 KE. (A one-to-one
panel was tried and dropped as redundant.)

Measurements are shown only where the beat is regular in space over the
domain, i.e. where at least two cycles fit the record. Equatorward of that it
is not regular: for the varying-N2 series the predicted beat grows to 8473 km
at the equator, where the bound wave 2*k1(M2) and the free mode-1 wave k1(M4)
are nearly degenerate (108.0 vs 53.7 km, so 2*k1(M2) and k1(M4) differ by a
few tenths of a percent), less than one cycle fits the domain, the wavenumber
bin width (lambda^2/L) exceeds the wavelength itself, and the periodogram has
no peak to find -- it returns ~1100-1200 km at every latitude from 0 to 15
deg regardless of what theory predicts, i.e. the record length rather than the
flow. Those latitudes are therefore not plotted.

A third series, 15.27-38 (the same varying-N2 stratification plus a
Garrett-Munk background), was checked and is NOT plotted: its predicted beat
is identical to 11.27-38 by construction and its measured beat agrees with
11.27-38 to 1-3% at 30-45 deg N, so it lands on top of the red squares. The
GM background does not shift the beat; that belongs in the text, not in a
third symbol set. (Its one outlier is 28.8 deg N, +7%, which is also where the
M4 spectral peak is weakest by three orders of magnitude -- the PSI latitude,
where the daughter waves contaminate the M4 band.)

Inputs are the m4harm_AMZexptXX.YY.jld2 files written by
IW_M4_harmonic_surface.jl; the measurement method (interior window, detrend,
Tukey taper, zero-padded periodogram) is that of IW_M4_beat_spectrum.jl.

Paper format: 18 x 9 cm, fontsize 10, saved at a true 300 dpi via savefig300.
=#

using Printf, JLD2, Statistics, FFTW, DSP, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1

const T2      = 12 + 25.2/60
const w2      = 2π/(T2*3600)
const w4      = 2*w2
# Analysis window. Model geometry: left sponge 0-40 km with the forcing
# Gaussian at 80 km (width 16 km), right sponge 1800-2000 km (Sp_Region_right =
# 200 km, quadratic mask). The left sponge still passes the wave, so the record
# starts just downstream of the forcing at 100 km; it ends where the right
# sponge begins. Of the four windows tested (100-2000, 100-1800, 200-1900,
# 200-1800) this one gives both the smallest mean bias vs theory (1.035) and
# the tightest spread across latitude (0.043, about half the others) -- the
# windows that reach past 1800 km leak the sponge ramp-down into the
# long-wavelength end, which hurts the longest beat (25 deg N) most.
const XLO     = 100e3
const XHI     = 1800e3
const ZPAD    = 16
const TUKEY   = 0.25
const LSEARCH = (60.0, 1200.0)   # search band for the dominant beat [km]
const XTICKS  = 0:10:40
const NCYC    = 2.0              # cycles needed in the record to plot a point

# the two series, in plot order: (mainnm, runnms, colour, marker, label stem)
SERIES = [(11, collect(27:38), :crimson,   :rect,      "N₂(lat)"),
          (11, collect(53:64), :royalblue, :utriangle, "N₂(50°N)")]

# predicted bound-free n=1 beat wavelength [m]
function beat_pred(row)
    f  = coriolis(row.lat)
    nh = row.DX < 500 ? 1 : 0
    @load string(dirforce, n2_filename(row)) N2w zfw
    k2, = sturm_liouville_noneqDZ_norm(zfw, N2w, f, w2, nh)
    k4, = sturm_liouville_noneqDZ_norm(zfw, N2w, f, w4, nh)
    return 2π/abs(2*k2[1] - k4[1])
end

# zero-padded periodogram of the detrended, tapered interior record
function beat_meas(xc, ke)
    I  = findall(v -> XLO <= v <= XHI, xc)
    xs = xc[I];  y = ke[I]
    n  = length(y);  dx = xs[2] - xs[1]
    A  = hcat(ones(n), (xs .- mean(xs)) ./ (xs[end]-xs[1]))
    yw = (y .- A*(A \ y)) .* DSP.tukey(n, TUKEY)
    nf = ZPAD*n
    P  = abs2.(rfft(vcat(yw, zeros(nf-n))))
    L  = [i == 0 ? Inf : 1e-3/(i/(nf*dx)) for i in 0:length(P)-1]
    J  = findall(v -> LSEARCH[1] <= v <= LSEARCH[2], L)
    return L[J[argmax(P[J])]], (xs[end]-xs[1])
end

# record length is the same for every run, so take it once (a `local` declared
# here would not survive the top-level for-loop's own scope)
@load string(dirout, @sprintf("m4harm_AMZexpt%02i.%02i.jld2",
    SERIES[1][1], SERIES[1][2][1])) xc
Lrec = let I = findall(v -> XLO <= v <= XHI, xc); xc[I][end] - xc[I][1] end

function collect_series(mainnm, runnms)
    lat = Float64[];  Lpred = Float64[];  Lmeas = Float64[];  ok = Bool[]
    for row in get_runs(mainnm, runnms)
        @load string(dirout, @sprintf("m4harm_AMZexpt%02i.%02i.jld2",
            mainnm, row.runnm)) xc ke4
        lm, _ = beat_meas(xc, ke4)
        lp = beat_pred(row)/1e3
        push!(lat, row.lat);  push!(Lpred, lp);  push!(Lmeas, lm)
        push!(ok, Lrec/1e3 / lp >= NCYC)
    end
    return (; lat, Lpred, Lmeas, ok)
end

S = [collect_series(s[1], s[2]) for s in SERIES]

for (si, s) in enumerate(SERIES)
    r = S[si]
    println("\n", "="^86)
    @printf("%d.%d-%d   %s   (record %.0f km)\n", s[1], s[2][1], s[2][end], s[5], Lrec/1e3)
    println("="^86)
    println(rpad("lat",7), rpad("predicted",12), rpad("measured",12), rpad("ratio",9),
            rpad("%diff",9), rpad("cycles",9), "used")
    for i in eachindex(r.lat)
        println(rpad(@sprintf("%.1f",r.lat[i]),7), rpad(@sprintf("%.0f km",r.Lpred[i]),12),
            rpad(r.ok[i] ? @sprintf("%.0f km",r.Lmeas[i]) : "--",12),
            rpad(r.ok[i] ? @sprintf("%.3f",r.Lmeas[i]/r.Lpred[i]) : "--",9),
            rpad(r.ok[i] ? @sprintf("%+.1f",100*(r.Lmeas[i]/r.Lpred[i]-1)) : "--",9),
            rpad(@sprintf("%.2f",(Lrec/1e3)/r.Lpred[i]),9), r.ok[i] ? "yes" : "no")
    end
    d = 100 .* (r.Lmeas[r.ok]./r.Lpred[r.ok] .- 1)
    @printf("usable latitudes %.1f-%.1f (%d runs): mean %+.2f%%, mean abs %.2f%%, std %.2f%%, rms %.2f%%\n",
        minimum(r.lat[r.ok]), maximum(r.lat[r.ok]), length(d),
        mean(d), mean(abs.(d)), std(d), sqrt(mean(d.^2)))
end
dall = vcat([100 .* (S[i].Lmeas[S[i].ok]./S[i].Lpred[S[i].ok] .- 1) for i in 1:length(SERIES)]...)
@printf("\nboth series pooled (%d points): mean %+.2f%%, mean abs %.2f%%, std %.2f%%\n",
    length(dall), mean(dall), mean(abs.(dall)), std(dall))

## --- figure: single panel, 18 x 9 cm, fontsize 10 -------------------------
# The one-to-one panel was dropped as redundant: with the theory curves and the
# measurements on the same axes, agreement is already legible here.
cm_to_pt = 72/2.54
fig = Figure(size=(18*cm_to_pt, 9*cm_to_pt), fontsize=10)

ax1 = Axis(fig[1,1], xlabel="latitude [°]", ylabel="beat wavelength [km]",
    title = rich("M", subscript("4"), " beat wavelength"), titlesize = 11,
    yscale=log10, xticks=XTICKS,
    yticks=([100,200,500,1000,2000,5000,10000],
            ["100","200","500","1000","2000","5000","10000"]))

# theory curves first, so the symbols sit on top
for (si, s) in enumerate(SERIES)
    lines!(ax1, S[si].lat, S[si].Lpred, color=si == 1 ? :black : :gray55,
        linewidth = si == 1 ? 1.8 : 1.5, linestyle = si == 1 ? :solid : :dash,
        label = string("theory 2π/Δk, ", s[5]))
end
for (si, s) in enumerate(SERIES)
    r  = S[si]
    # label the measured series by the run numbers actually plotted, so it
    # stays correct if the selection changes
    rn = s[2][r.ok]
    scatter!(ax1, r.lat[r.ok], r.Lmeas[r.ok], color=s[3], marker=s[4], markersize=8,
        label = @sprintf("simulated, %d.%d-%d", s[1], rn[1], rn[end]))
end
xlims!(ax1, -2, 47);  ylims!(ax1, 100, 12000)
axislegend(ax1, position=:lb, framevisible=false, labelsize=8, patchsize=(14,10))

display(fig)

if figflag == 1
    fout = string(dirfig, "M4_beat_theory_vs_sim_ppr.png")
    savefig300(fout, fig)
    println("saved ", fout)
end
