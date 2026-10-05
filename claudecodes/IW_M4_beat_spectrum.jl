#= IW_M4_beat_spectrum.jl
Maarten Buijsman, USM DMS, 2026-9-16

Step 2 of the M4 beat-wavelength analysis: along-transect WAVENUMBER SPECTRUM
of the surface M4 kinetic energy from IW_M4_harmonic_surface.jl (11.27-38),
to measure the beat wavelength and compare it with theory.

Theory. The M4 field is the sum of a BOUND harmonic forced by the M2 mode-1
parent, at wavenumber 2*k1(M2), and FREE M4 modes at k_n(M4). KE_M4 ~ |u_M4|^2
therefore beats in x at every difference wavenumber:
    bound-free :  dk_n  = |2*k1(M2) - k_n(M4)|          -> Lbeat_n = 2pi/dk_n
    free-free  :  dk_nm = |k_n(M4)  - k_m(M4)|
Mode 1 carries most of the energy (k-omega spectra), so the bound-free n=1
beat is expected to dominate; the rest are plotted to see whether any of the
higher beats are detectable.

Method. Per latitude:
  1. restrict to the interior x-window (XLO..XHI) -- the first ~150 km is the
     generation ramp and the last ~100 km is the sponge, and a "peak" in
     either is a boundary artifact rather than a beat crest (same lesson as
     IW_KEt_beat_distance.jl).
  2. remove mean + linear trend, so the slow along-transect decay of the
     envelope does not pile power into the lowest wavenumbers.
  3. Tukey taper (mild, alpha=0.25) -- gentler main-lobe broadening than a
     Hann window, which matters because the longest beats fit only a couple of
     cycles into the record.
  4. zero-pad x ZPAD and take the periodogram. NOTE: zero-padding interpolates
     the spectrum, letting the peak be LOCATED finely; it does not improve the
     true resolution, which is set by the record length (two peaks closer than
     ~1/L_record in wavenumber cannot be separated).
  5. report the dominant peak over the search band as the measured Lbeat.

Reliability. The interior record is ~1700 km, so a beat is only measurable if
several cycles fit inside it. Latitudes where the PREDICTED beat exceeds half
the record are reported as "beat > record" rather than fitted -- at 0-10 deg N
the predicted beat is 2855-8473 km, i.e. longer than the whole domain, so
there is nothing to measure there by any method.
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

const T2    = 12 + 25.2/60
const w2    = 2π/(T2*3600)
const w4    = 2*w2
const NMODE = 5
const XLO   = 200e3      # interior window start [m] (past the generation ramp)
const XHI   = 1900e3     # interior window end   [m] (before the sponge)
const ZPAD  = 16         # zero-padding factor (peak localization only)
const TUKEY = 0.25
const LSEARCH = (60.0, 1200.0)    # search band for the dominant beat [km]

mainnm = 11
runnms = collect(27:38)
runs   = get_runs(mainnm, runnms)
fnum   = string(mainnm, ".", runnms[1], "-", runnms[end])

# ---- theoretical beat wavelengths per run -----------------------------------
function beats_theory(row)
    f  = coriolis(row.lat)
    nh = row.DX < 500 ? 1 : 0
    @load string(dirforce, n2_filename(row)) N2w zfw
    k2, = sturm_liouville_noneqDZ_norm(zfw, N2w, f, w2, nh)
    k4, = sturm_liouville_noneqDZ_norm(zfw, N2w, f, w4, nh)
    kb  = 2*k2[1]
    Lbf = [2π/abs(kb - k4[n]) for n in 1:NMODE]                     # bound-free
    Lff = [2π/abs(k4[n] - k4[m]) for n in 1:3 for m in (n+1):4]     # free-free
    return Lbf, Lff
end

# ---- periodogram of KE_M4(x) -------------------------------------------------
function beat_spectrum(x, ke)
    I  = findall(v -> XLO <= v <= XHI, x)
    xs = x[I];  y = ke[I]
    dx = xs[2] - xs[1];  n = length(y)

    # remove mean + linear trend
    A  = hcat(ones(n), (xs .- mean(xs)) ./ (xs[end]-xs[1]))
    y  = y .- A*(A \ y)

    w  = DSP.tukey(n, TUKEY)
    yw = y .* w

    nfft = ZPAD * n
    Y  = rfft(vcat(yw, zeros(nfft - n)))
    P  = abs2.(Y)
    fk = (0:length(P)-1) ./ (nfft*dx)        # cycles per metre
    return fk, P, (xs[end]-xs[1]), n*dx
end

results = []
for row in runs
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, row.runnm)
    @load string(dirout, "m4harm_", fnames, ".jld2") xc ke4 LAT

    fk, P, Lrec, _ = beat_spectrum(xc, ke4)
    Lkm = [f > 0 ? 1e-3/f : Inf for f in fk]        # wavelength [km]

    Lbf, Lff = beats_theory(row)

    # dominant peak within the search band
    J  = findall(v -> LSEARCH[1] <= v <= LSEARCH[2], Lkm)
    jm = J[argmax(P[J])]
    Lmeas = Lkm[jm]

    ncyc_meas = (Lrec/1e3)/Lmeas
    ncyc_pred = (Lrec/1e3)/(Lbf[1]/1e3)
    reliable  = ncyc_pred >= 2.0

    push!(results, (; lat=LAT, runnm=row.runnm, Lkm, P, Lbf, Lff,
                      Lmeas, ncyc_meas, ncyc_pred, reliable, Lrec))
end

## ---- table -------------------------------------------------------------------
println("\n", "="^96)
println("measured vs predicted M4 beat wavelength   (interior record = ",
        @sprintf("%.0f", results[1].Lrec/1e3), " km)")
println("="^96)
println(rpad("lat",7), rpad("L_pred n=1",13), rpad("L_meas",11), rpad("ratio",9),
        rpad("cycles(pred)",14), "verdict")
for r in results
    lp = r.Lbf[1]/1e3
    println(rpad(@sprintf("%.1f",r.lat),7),
        rpad(@sprintf("%.0f km",lp),13),
        rpad(r.reliable ? @sprintf("%.0f km",r.Lmeas) : "--",11),
        rpad(r.reliable ? @sprintf("%.2f",r.Lmeas/lp) : "--",9),
        rpad(@sprintf("%.2f",r.ncyc_pred),14),
        r.reliable ? "measurable" : "beat > record, not fitted")
end

## ---- figure: spectra with predicted beats overlaid ---------------------------
nrun = length(results)
fig  = Figure(size=(1150, 145*nrun + 70), fontsize=10)
for (ir, r) in enumerate(results)
    # log y: the beat peak dominates by many decades, so on a linear axis the
    # weaker n>=2 / free-free beats are invisible even if present
    ax = Axis(fig[ir,1], xscale=log10, yscale=log10,
        ylabel = @sprintf("%.1f°", r.lat),
        xlabel = ir == nrun ? "wavelength [km]" : "",
        xticks = ([20,50,100,200,500,1000,2000], ["20","50","100","200","500","1000","2000"]),
        yticks = ([1e-8,1e-6,1e-4,1e-2,1], ["10⁻⁸","10⁻⁶","10⁻⁴","10⁻²","1"]),
        xticklabelsvisible = ir == nrun, yticklabelsize = 7)

    J  = findall(v -> 15 <= v <= 2500, r.Lkm)
    Pn = max.(r.P[J]./maximum(r.P[J]), 1e-12)     # floor so log10 is finite
    lines!(ax, r.Lkm[J], Pn, color=:crimson, linewidth=1.0)

    # predicted beats: bound-free n=1 (solid), n>=2 (dashed), free-free (dotted)
    vlines!(ax, [r.Lbf[1]/1e3], color=:black, linewidth=1.8)
    vlines!(ax, [r.Lbf[n]/1e3 for n in 2:NMODE], color=:steelblue, linewidth=1, linestyle=:dash)
    vlines!(ax, [L/1e3 for L in r.Lff], color=:seagreen, linewidth=0.8, linestyle=:dot)
    r.reliable && vlines!(ax, [r.Lmeas], color=:crimson, linewidth=1.2, linestyle=:dashdot)

    xlims!(ax, 15, 2500); ylims!(ax, 1e-9, 5.0)
    ir == 1 && (ax.title = string("wavenumber spectrum of surface M4 KE, ", fnum,
        "   black = predicted bound-free n=1, blue dash = n=2..5, green dot = free-free, red dash-dot = measured"))
end
display(fig)
figflag == 1 && savefig300(string(dirfig, "M4_beat_spectrum_", fnum, ".png"), fig)

## ---- summary: measured vs predicted ------------------------------------------
ok = filter(r -> r.reliable, results)
fig2 = Figure(size=(950, 420), fontsize=10)
ax1 = Axis(fig2[1,1], xlabel="latitude [°]", ylabel="beat wavelength [km]",
    title="(a) measured vs predicted", yscale=log10)
lines!(ax1, [r.lat for r in results], [r.Lbf[1]/1e3 for r in results],
    color=:black, linewidth=2, label="predicted 2π/Δk (n=1)")
scatter!(ax1, [r.lat for r in ok], [r.Lmeas for r in ok],
    color=:crimson, marker=:rect, markersize=11, label="measured (spectral peak)")
hlines!(ax1, [results[1].Lrec/2e3], color=:gray, linestyle=:dash, label="½ record length")
axislegend(ax1, position=:lb, framevisible=false, labelsize=9)

ax2 = Axis(fig2[1,2], xlabel="predicted [km]", ylabel="measured [km]",
    title="(b) one-to-one", xscale=log10, yscale=log10)
lp = [r.Lbf[1]/1e3 for r in ok]; lm = [r.Lmeas for r in ok]
lo, hi = 0.8*minimum(vcat(lp,lm)), 1.25*maximum(vcat(lp,lm))
lines!(ax2, [lo,hi], [lo,hi], color=:gray, linestyle=:dash)
scatter!(ax2, lp, lm, color=:crimson, marker=:rect, markersize=11)
for r in ok
    text!(ax2, r.Lbf[1]/1e3, r.Lmeas; text=@sprintf("%.0f°",r.lat), fontsize=8,
        align=(:left,:bottom), offset=(4,2))
end
xlims!(ax2, lo, hi); ylims!(ax2, lo, hi)
display(fig2)
figflag == 1 && savefig300(string(dirfig, "M4_beat_measured_vs_theory_", fnum, ".png"), fig2)
