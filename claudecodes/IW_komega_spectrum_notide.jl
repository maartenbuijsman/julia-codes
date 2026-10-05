#= IW_komega_spectrum_notide.jl
Maarten Buijsman, USM DMS, 2026-9-5

k-omega and P(omega) spectra for the mainnm=13, runnm=1:13 block
(params_13_noforce.jl): 200m grid, Garrett-Munk-spectrum-INITIALIZED,
Flux=0 -- NO tidal (D2) forcing at all, i.e. a pure free-decay/background
internal-wave-field run. This is a different question from the
tide-forced GM comparison in IW_komega_spectrum.jl (mainnm 12/13, runnm
27:39, which DOES have D2 forcing on top of the GM initial spectrum) --
here there is no forcing frequency/wavenumber to reference at all, so:
  - no (omega,k1)/(2omega,2k1)/(2omega,k2) bound/free markers (meaningless
    without a forcing frequency)
  - no snap-to-whole-M2-periods/no-taper trick (that relied on tidal
    periodicity) -- use a real Hann taper in time instead, since the GM
    field has no periodicity to exploit and a raw window WILL have a
    genuine edge mismatch
  - no ω⁻³ harmonic-envelope reference (there are no discrete tidal
    harmonics to test against) -- only the ω⁻² GM continuum reference
  - the mode-1/mode-2 dispersion curves are still plotted, since they're a
    property of the stratification/rotation, not the (absent) forcing --
    useful to see whether the GM background energy still organizes along
    the internal-wave dispersion relation

REVISED 2026-9-5: no longer lowpass-filters+decimates x before the FFT.
That was found (via IW_Pomega_x1000km_check.jl / a direct full-res-vs-
decimated overlay) to NOT be frequency-neutral -- it silently discarded
~50% of true power by 4 cpd and ~98% by 10 cpd, because higher-frequency
motions genuinely carry energy at wavenumbers the antialiasing filter
removed. Now the full 2D FFT is computed on the FULL 200m-resolution
u,v (no pre-filtering at all), and only the ALREADY-COMPUTED (freq,k)
power matrix is cropped/remapped to a smaller (freq<=50 cpd, k<=2 cyc/km)
subset purely for the heatmap's sake (CairoMakie chokes on the full
~2900x8000 grid) -- a display crop of real output, not a filter of the
input, so every remaining cell is the true full-resolution FFT value.
P(ω) still integrates over the FULL (uncropped) k-range for the correct
total energy per frequency.

REVISED 2026-9-5 (again): P(ω) now uses the segment-averaged 1D
FFT-in-time approach (3 x-regions, 4km-subsampled u/v FFTs averaged as
POWER) instead of integrating the 2D k-omega spectrum over k -- same
method and reasoning as IW_komega_spectrum.jl's P(ω), much cheaper and
shows regional structure (3 lines) instead of one domain-wide curve. The
k-ω heatmap itself is unchanged and still needs the full 2D FFT.
=#

println("number of threads is ",Threads.nthreads())

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, DSP, FFTW

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0 = "/home/mbui/ModelOutput/"
dirsim = string(pth0,"IW/")
dirfig = string(pth0,"figs/")
dirforce = string(pth0,"IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))

figflag = 1

# analysis window -- same clean-domain reasoning as IW_komega_spectrum.jl
# (source-adjacent/sponge regions avoided), though here there is no tidal
# source at all so this just avoids the domain edges/sponge
const xleft_km  = 100.0   # widened from 200 so it also covers the P(omega) segments below (100-1600km)
const xright_km = 1800.0
const tlast_days = 10.0   # last N days of the run (early spin-up/adjustment excluded)

# heatmap display crop (applied to the ALREADY-COMPUTED full-resolution
# power matrix, not a filter of the input -- see header note)
const freq_crop_max = 50.0   # cpd
const k_crop_max    = 2.0    # cyc/km

# run selection: mainnm=13, runnm=11 -> lat=40, Flux=0 (GM-only, no tide)
mainnm = 13
runnm  = 11

row = get_runs(mainnm, [runnm])[1]
LAT = row.lat
@assert row.Flux == 0.0 "expected a Flux=0 (no-tide) run, got Flux=$(row.Flux) for $mainnm.$runnm"
fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
filename = string(dirsim, fnames, ".nc")
println(fnames, "; lat=", LAT, ", Flux=", row.Flux, " (GM-only, no tidal forcing) -------------------")

# mode-1/mode-2 dispersion curves (hydrostatic and nonhydrostatic) -- a
# property of the stratification/rotation only, plotted for reference even
# though there's no forcing frequency to mark on them
fnamegrid = n2_filename(row)
@load string(dirforce,fnamegrid) N2w zfw
fcor = coriolis(LAT)
nonhyd = row.DX < 500 ? 1 : 0

fcor_cpd = fcor/(2π)*86400
Nmax_cpd = sqrt(maximum(N2w))/(2π)*86400
disp_fmax_cpd = min(freq_crop_max*1.05, Nmax_cpd*0.98)
disp_freq_cpd = range(fcor_cpd*1.001, disp_fmax_cpd, length=600)
disp_om = disp_freq_cpd .* (2π/86400)
k_disp_nh_cpkm  = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 1)[1][1]/(2π)*1e3 for om_i in disp_om]
k_disp_hy_cpkm  = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 0)[1][1]/(2π)*1e3 for om_i in disp_om]
k_disp_nh_cpkm2 = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 1)[1][2]/(2π)*1e3 for om_i in disp_om]
k_disp_hy_cpkm2 = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 0)[1][2]/(2π)*1e3 for om_i in disp_om]
println("f (inertial) = ", fcor_cpd, " cpd;  N (buoyancy) = ", Nmax_cpd, " cpd")

# load surface u,v over the x/t window only
ds = NCDataset(filename,"r")
xf   = ds["x_faa"][:]
tday = ds["time"][:] ./ (24*3600)

Ix = findall(xleft_km*1e3 .<= xf .<= xright_km*1e3)
It = findall(tday .>= tday[end]-tlast_days)

isodd(length(Ix)) || (Ix = Ix[1:end-1])
isodd(length(It)) || (It = It[1:end-1])

dx = xf[2]-xf[1]
dt = (tday[It[2]]-tday[It[1]]) * 24*3600
dt_check = diff(tday[It]) .* 24*3600
maximum(abs.(dt_check .- dt)) > 1e-3*dt && @warn "time window is not uniformly sampled -- FFT frequency axis will be wrong"

println("x window: ", xf[Ix[1]]/1e3, "-", xf[Ix[end]]/1e3, " km (", length(Ix), " points, dx=", dx, " m)")
println("t window: ", tday[It[1]], "-", tday[It[end]], " days (", length(It), " points, dt=", dt, " s)")

Nz = size(ds["u"], 2)
u_slice = permutedims(ds["u"][Ix, Nz, It], (2,1))   # (Nt, Nx), surface, FULL 200m resolution
v_slice = permutedims(ds["v"][Ix, Nz, It], (2,1))
close(ds)

# 2D k-omega spectrum at FULL resolution -- no lowpass/decimation (see
# header note). REAL Hann taper in BOTH dimensions: there is no tidal
# periodicity to snap the time window to, so a boxcar in time would leave a
# genuine edge discontinuity and leak energy across the whole spectrum
dt_days, dx_km = dt/86400, dx/1e3
freq_cpd, k_cpkm, power  = komega_spectrum(u_slice, dt_days, dx_km; taper=(:hann,:hann))
_,        _,      powerv = komega_spectrum(v_slice, dt_days, dx_km; taper=(:hann,:hann))
powerKE = power .+ powerv

# crop to non-negative frequencies, then fold (double every row except
# freq=0) -- same one-sided-PSD convention as IW_komega_spectrum.jl
posf = findall(freq_cpd .>= 0)
freq_cpd = freq_cpd[posf]
power, powerv, powerKE = power[posf,:], powerv[posf,:], powerKE[posf,:]
fold = ones(length(freq_cpd)); fold[freq_cpd .> 0] .= 2
power, powerv, powerKE = power.*fold, powerv.*fold, powerKE.*fold

println("full-resolution FFT: Nx=", size(u_slice,2), ", Nyquist k=", maximum(k_cpkm), " cyc/km; Nyquist freq=", maximum(freq_cpd), " cpd")

pmax = log10(maximum(power))
pmaxKE = log10(maximum(powerKE))

# ---- display crop for the heatmaps ONLY: remap the already-computed
# full-resolution (freq,k) power matrix down to a smaller subset (freq<=50
# cpd, k<=2 cyc/km) so CairoMakie doesn't choke on the full ~Nt x 8000
# array -- this is a crop of real output, not a filter of the input, so
# every remaining cell is still the true full-resolution FFT value.
# P(ω) below still integrates over the FULL (uncropped) k-range.
fcrop = findall(freq_cpd .<= freq_crop_max)
kcrop = findall(abs.(k_cpkm) .<= k_crop_max)
freq_hm, k_hm = freq_cpd[fcrop], k_cpkm[kcrop]
power_hm, powerKE_hm = power[fcrop,kcrop], powerKE[fcrop,kcrop]
println("heatmap crop: freq<=", freq_crop_max, " cpd (", length(fcrop), " pts), |k|<=", k_crop_max, " cyc/km (", length(kcrop), " pts)")

# ---- k-omega heatmap (u), with dispersion-curve overlay only (no forcing
# markers -- there is no forcing) --------------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size=(15*cm_to_pt, 11*cm_to_pt), fontsize=9)
ax = Axis(fig[1,1], title=string("u spectrum, ",fnames," (lat=",LAT,", GM only, no tide)"), titlesize=9,
    xlabel="wavenumber [cycles/km]", ylabel="frequency [cpd]")
xmax = k_crop_max
ymax = freq_crop_max
kmax_disp = maximum(vcat(k_disp_nh_cpkm, k_disp_hy_cpkm))

hm = heatmap!(ax, k_hm, freq_hm, log10.(power_hm)', colormap = Reverse(:Spectral), colorrange=(pmax-12.2,pmax-1.7))
Colorbar(fig[1,2], hm, label="log10 power [m²/s²·day·km]")
ylims!(ax, 0, ymax)
xlims!(ax, 0, xmax)

ytick_vals = [1,2,4,8,10,20,30,40,50]
ax.yticks = (ytick_vals, string.(ytick_vals))

lines!(ax, k_disp_nh_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, label="nonhydrostatic (mode 1)")
lines!(ax, k_disp_hy_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 1)")
lines!(ax, k_disp_nh_cpkm2,  disp_freq_cpd, color=:gray50, linewidth=1.5, linestyle=:dash, label="nonhydrostatic (mode 2)")
lines!(ax, k_disp_hy_cpkm2,  disp_freq_cpd, color=:gray50, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 2)")
hlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
axislegend(ax, position=:lt, labelsize=8, patchsize=(14,6), rowgap=2, patchlabelgap=4)

display(fig)
if figflag==1; savefig300(string(dirfig,"komega_",fnames,".png"), fig); end

# ---- KE = Pu+Pv version -----------------------------------------------------
figKE = Figure(size=(800,600))
axKE = Axis(figKE[1,1], title=string("k-ω spectrum of KE (Pu+Pv) — ",fnames," (lat=",LAT,", GM only, no tide)"),
    xlabel="wavenumber [cycles/km] (1/wavelength)", ylabel="frequency [cpd]")
hmKE = heatmap!(axKE, k_hm, freq_hm, log10.(powerKE_hm)', colormap = Reverse(:Spectral), colorrange=(pmaxKE-8,pmaxKE))
Colorbar(figKE[1,2], hmKE, label="log10 power [m²/s²·day·km]")
ylims!(axKE, 0, freq_crop_max)
xlims!(axKE, 0, k_crop_max)
lines!(axKE, k_disp_nh_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, label="nonhydrostatic (mode 1)")
lines!(axKE, k_disp_hy_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 1)")
lines!(axKE, k_disp_nh_cpkm2, disp_freq_cpd, color=:dodgerblue, linewidth=1.5, label="nonhydrostatic (mode 2)")
lines!(axKE, k_disp_hy_cpkm2, disp_freq_cpd, color=:dodgerblue, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 2)")
hlines!(axKE, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
axislegend(axKE, position=:lt, labelsize=11)
display(figKE)
if figflag==1; save(string(dirfig,"komega_KE_",fnames,".png"), figKE); end

# ---- P(ω): segment-averaged 1D FFT-in-time spectra, NOT the 2D k-omega
# integral -- much cheaper (plain 1D FFTs) and shows regional structure
# instead of one domain-wide number; same method as the D2-forced
# IW_komega_spectrum.jl. For each of 3 x-segments, run fft_spectra() on u
# and v at every x-point (subsampled every seg_dx_km=4km -- ~126 points per
# 500km segment instead of ~2500 at native 200m, ~20x fewer FFT calls with
# no meaningful loss of averaging robustness), then average the resulting
# POWER (Pu+Pv) across those x-points -- NOT the raw complex FFT
# coefficients, which would partially cancel by phase for a propagating
# wave. Tukey(0.5) taper (NOT boxcar): unlike the D2-forced script, this
# run has no tidal periodicity to snap the window to, so a real taper is
# needed (same reasoning as the Hann taper used for the 2D FFT above).
xseg_km = xf[Ix] ./ 1e3
segments_km = [(100.0,600.0), (600.0,1100.0), (1100.0,1600.0)]
seg_colors = [:seagreen, :black, :dodgerblue]
seg_labels = ["100-600 km", "600-1100 km", "1100-1600 km"]
seg_dx_km = 4.0

seg_freq = Float64[]
seg_PKE = Vector{Vector{Float64}}(undef, length(segments_km))
for (si, (lo,hi)) in enumerate(segments_km)
    idxseg_all = findall(lo .<= xseg_km .<= hi)
    stride = max(1, round(Int, seg_dx_km/dx_km))
    idxseg = idxseg_all[1:stride:end]
    n = length(idxseg)
    Pu_acc = Float64[]; Pv_acc = Float64[]
    for ix in idxseg
        _, f1d, Pu_x = fft_spectra(tday[It], u_slice[:,ix]; tukeycf=0.5, numwin=1, linfit=true)
        _,   _, Pv_x = fft_spectra(tday[It], v_slice[:,ix]; tukeycf=0.5, numwin=1, linfit=true)
        if isempty(Pu_acc)
            Pu_acc = zeros(length(Pu_x)); Pv_acc = zeros(length(Pv_x))
            global seg_freq = f1d   # cpd (tday is in days) -- top-level script, so `global` is needed here (unlike IW_komega_spectrum.jl, this isn't wrapped in a function)
        end
        Pu_acc .+= Pu_x; Pv_acc .+= Pv_x
    end
    seg_PKE[si] = (Pu_acc .+ Pv_acc) ./ n
    println("segment ", seg_labels[si], ": averaged over ", n, " x-points")
end

xlo, xhi = 0.1, freq_crop_max
ipos = findall((seg_freq .>= xlo) .& (seg_freq .<= xhi))

# omega^-2 GM continuum reference, anchored at 1 cpd from the middle segment
f_anchor = 1.0
i_anchor = argmin(abs.(seg_freq[ipos] .- f_anchor))
gm_ref = seg_PKE[2][ipos][i_anchor] .* (seg_freq[ipos] ./ seg_freq[ipos][i_anchor]).^(-2)

figPom = Figure(size=(700,450))
axPom = Axis(figPom[1,1], title=string("P(ω), segment-averaged (u,v FFT per x, power-averaged) — ",fnames," (lat=",LAT,", GM only, no tide)"),
    xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)
for (si, lbl) in enumerate(seg_labels)
    lines!(axPom, seg_freq[ipos], seg_PKE[si][ipos], color=seg_colors[si], label=lbl)
end
lines!(axPom, seg_freq[ipos], gm_ref, color=:red, linestyle=:dash, label="ω⁻² (GM continuum) reference")
vlines!(axPom, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
xlims!(axPom, xlo, xhi)
ylims!(axPom, 1e-10, 1e0)
pomega_xtick_candidates = [1,2,4,8,10,20,30,40,60,80,100,120,140]
pomega_xticks = pomega_xtick_candidates[findall(xlo .<= pomega_xtick_candidates .<= xhi)]
axPom.xticks = (pomega_xticks, string.(pomega_xticks))
axislegend(axPom, position=:lb, labelsize=9)
display(figPom)
if figflag==1; save(string(dirfig,"komega_Pomega_",fnames,".png"), figPom); end
