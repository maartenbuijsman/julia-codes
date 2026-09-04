#= IW_komega_spectrum.jl
Maarten Buijsman, USM DMS, 2026-9-3 (fixed P(omega) ylims across runs for GM comparison; switched to 10.37)
2D (space-time) FFT of surface u-velocity: a wavenumber-frequency (k-omega)
spectrum, meant to directly SEE the 2k/2omega compound-tide wave that the
k-based epsilon (epsnh_k/epshy_k in IW_nondim_params.jl) is a mismatch
measure of. Overlays three reference points on top of the empirical spectrum:
  - (ω, k1)        the D2 forcing frequency and its mode-1 wavenumber
  - (2ω, 2k1)      the BOUND wave -- phase-locked to the primary wave
                    (wavenumber = sum of the interacting wavenumbers, exactly
                    2k1) regardless of whether that satisfies the dispersion
                    relation; doesn't propagate independently
  - (2ω, k2)        the FREE wave -- the actual mode-1 wavenumber at 2ω from
                    the dispersion relation (via sturm_liouville_noneqDZ_norm),
                    a genuine independent normal-mode solution that
                    propagates at its own group velocity
The gap between the (2ω,2k1) and (2ω,k2) markers on the plot IS
epsilon_k = (k2²-(2k1)²)/(2k1)² made visible.

x window: 500-1000 km (well clear of the source at x=80 km and the sponge
past ~1850 km). Time window: last 10 days of the run (spin-up excluded).
Uses u (not v): u is the along-domain component, i.e. the propagation
direction the whole k1/k2 mode-1 analysis is about.
=#

println("number of threads is ",Threads.nthreads())

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, DSP, FFTW

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";
    dirforce = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirfig = string(pth0,"figs/");
    dirforce = string(pth0,"IW/forcingfiles/");
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

figflag = 1

const T2 = 12+25.2/60          # M2 period [hours]
const ω  = 2π / (T2*3600)      # M2 frequency [rad/s]

# analysis window
# x as wide as the clean domain allows (source at ~80km, sponge from ~1850km)
# -- k-resolution dk=1/(Nx*dx) depends on this window LENGTH, not on dx, and
# a wider window is needed to separate the bound (2k1) and free (k2) waves at
# 2ω: at lat=25 k2-2k1 ~ 0.002 cyc/km, same order as dk on the previous
# 500km window (500-1000km), so the two were barely resolved
const xleft_km  = 200.0
const xright_km = 1800.0
const tlast_days = 10.0        # use the last N days of the record

# instead of an arbitrary 10-day window, snap the duration down to the
# largest whole number of M2 periods that fits inside it -- if the window is
# an exact multiple of the forcing period, the (quasi-)periodic tidal signal
# wraps around continuously with no discontinuity at the edges, so the
# time-domain Tukey taper (needed only to suppress the leakage a mismatched
# edge would cause) can be dropped entirely
const T2_days   = T2/24
const n_periods = floor(Int, tlast_days/T2_days)
const tdur_days = n_periods*T2_days
println("using ", n_periods, " whole M2 periods = ", tdur_days, " days (out of ", tlast_days, " requested) -- no time taper")

# run selection -- one run at a time (a k-omega spectrum is inherently
# per-run); swap mainnm/runnm to look at a different one
mainnm = 11
runnm  = 37   # lat=40, F=25 kW/m -- 200m grid, standard (no GM) forcing -- compare against 12.37

row = get_runs(mainnm, [runnm])[1]
LAT = row.lat
fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
filename = string(dirsim, fnames, ".nc")
println(fnames, "; lat=", LAT, " -------------------")

# mode-1 wavenumber at ω and 2ω from the dispersion relation (same convention
# as getkres in IW_nondim_params.jl) -----------------------------------------
fnamegrid = n2_filename(row)
@load string(dirforce,fnamegrid) N2w zfw
fcor = coriolis(LAT)
nonhyd = row.DX < 500 ? 1 : 0

kn1, = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)
k1 = kn1[1]
k1_mode2 = kn1[2]   # mode-2 wavenumber at the SAME forcing frequency ω
kn2, = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, 2ω, nonhyd)
k2 = kn2[1]
println("k1 (at ω) = ", k1, " rad/m;  k1_mode2 (at ω) = ", k1_mode2, " rad/m;  k2 (at 2ω) = ", k2, " rad/m;  2*k1 = ", 2*k1, " rad/m")

# full mode-1 dispersion curves k(ω), hydrostatic and nonhydrostatic, for the
# overlay below. Swept from just above the local inertial cutoff (below which
# no propagating mode-1 IW exists -- the dispersion relation needs
# sqrt(ω²-f²), a DomainError for ω<f) up to 11 cpd, capped at the buoyancy
# frequency N (nonhydrostatic k diverges as ω→N from below)
fcor_cpd = fcor/(2π)*86400
Nmax_cpd = sqrt(maximum(N2w))/(2π)*86400
disp_fmax_cpd = min(11.0, Nmax_cpd*0.98)
disp_freq_cpd = range(fcor_cpd*1.001, disp_fmax_cpd, length=300)
disp_om = disp_freq_cpd .* (2π/86400)
k_disp_nh_cpkm  = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 1)[1][1]/(2π)*1e3 for om_i in disp_om]
k_disp_hy_cpkm  = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 0)[1][1]/(2π)*1e3 for om_i in disp_om]
k_disp_nh_cpkm2 = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 1)[1][2]/(2π)*1e3 for om_i in disp_om]
k_disp_hy_cpkm2 = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, om_i, 0)[1][2]/(2π)*1e3 for om_i in disp_om]
println("f (inertial) = ", fcor_cpd, " cpd;  N (buoyancy) = ", Nmax_cpd, " cpd")

# load surface u over the x/t window only (NCDatasets supports partial reads,
# so this never loads the full field) -----------------------------------------
ds = NCDataset(filename,"r")
xf   = ds["x_faa"][:]
tday = ds["time"][:] ./ (24*3600)

Ix = findall(xleft_km*1e3 .<= xf .<= xright_km*1e3)
It = findall(tday .>= tday[end]-tdur_days)

# keep n odd in both dimensions so the FFT's Nyquist bin is unambiguous:
# fftfreq stores the Nyquist entry as -fs/2 for even n but +fs/2 for odd n,
# so an even n silently mislabels the highest frequency/wavenumber as
# negative -- drop the last sample rather than risk that
isodd(length(Ix)) || (Ix = Ix[1:end-1])
isodd(length(It)) || (It = It[1:end-1])

dx = xf[2]-xf[1]
# dt from WITHIN the selected window, not the first two samples of the whole
# record: the output cadence changes partway through the run (3600 s near the
# start, 300 s later, at least for AMZexpt10.07) -- using the wrong one here
# silently mis-scales the whole frequency axis (a 3600/300=12x error shifted
# the true 1.93 cpd D2 peak to a bogus-looking 0.16 cpd, first caught by the
# Hovmoller in diag_hov.jl showing a single clean, fast oscillation where the
# spectrum wrongly implied a slow ~6-day one)
dt = (tday[It[2]]-tday[It[1]]) * 24*3600      # [s]
dt_check = diff(tday[It]) .* 24*3600
maximum(abs.(dt_check .- dt)) > 1e-3*dt && @warn "time window is not uniformly sampled -- FFT frequency axis will be wrong"

println("x window: ", xf[Ix[1]]/1e3, "-", xf[Ix[end]]/1e3, " km (", length(Ix), " points, dx=", dx, " m)")
println("t window: ", tday[It[1]], "-", tday[It[end]], " days (", length(It), " points, dt=", dt, " s)")

Nz = size(ds["u"], 2)
u_slice = permutedims(ds["u"][Ix, Nz, It], (2,1))   # (Nt, Nx), Nz = surface (z_aac[end] ~ 0 m)
v_slice = permutedims(ds["v"][Ix, Nz, It], (2,1))
close(ds)

# subsample in x (with an anti-alias lowpass filter first) down to just
# above the k-range we ever plot (~1/23 cyc/km) -- the FFT and (worse) the
# CairoMakie heatmap both scale with Nx, and on the 200m grid (mainnm=11,
# Nx~8000) rendering the full-resolution heatmap silently overran CairoMakie
# and dropped every element drawn afterward (colorbar, legend, dispersion
# curves, markers all vanished from the saved PNG with no error). dk depends
# on the window LENGTH (Nx*dx), not on dx itself, so decimating at fixed
# window length changes only the Nyquist wavenumber (which we don't need),
# not the resolution we actually use.
target_dx_km = 5.0                          # -> k-Nyquist = 1/(2*5km) = 0.1 cyc/km, ~2.3x our xmax=1/23
xdec = max(1, floor(Int, target_dx_km*1e3/dx))
if xdec > 1
    cutoff_km = 2*target_dx_km              # antialias cutoff, safely inside the new Nyquist
    for t in axes(u_slice,1)
        u_slice[t,:] = lowhighpass_butter(u_slice[t,:], cutoff_km*1e3, dx, 4, "low")
        v_slice[t,:] = lowhighpass_butter(v_slice[t,:], cutoff_km*1e3, dx, 4, "low")
    end
    u_slice = u_slice[:, 1:xdec:end]
    v_slice = v_slice[:, 1:xdec:end]
    dx = dx*xdec
    println("decimated x by ", xdec, "x (antialiased at ", cutoff_km, " km) -> dx=", dx, " m, Nx=", size(u_slice,2))
end

# 2D k-omega spectrum via the shared komega_spectrum() function
# (functions/komega_spectrum.jl) -- handles de-meaning (both marginals),
# tapering, the 2D FFT, and the rightward-propagation sign-convention fix
# (k-DATA reversed, not the axis labels; see that file's docstring for the
# full derivation, verified against a reference F-K implementation,
# fk_reference.jl). NO time taper (boxcar) since tdur_days is a whole number
# of M2 periods (see note above); Hann in space (lower sidelobes than Tukey
# -- compared directly, Tukey left visibly worse vertical striping)
# pass dt/dx already in the units we want the OUTPUT in (days, km) -- same
# convention as fft_spectra (functions/fft_spectra_vectorized.jl): the
# function then returns freq/k already in cpd / cyc-per-km, with power
# already correctly normalized in matching units (m^2/s^2*day*km), no
# separate axis-only rescaling (which would silently leave power in the
# WRONG units relative to the relabeled axes, as an earlier version of this
# script did -- it never called dt*dx/df*dk at all, just used raw |FFT|^2,
# which is why the numbers were unphysically large, ~1e7 at the M2 peak)
dt_days, dx_km = dt/86400, dx/1e3
freq_cpd, k_cpkm, power  = komega_spectrum(u_slice, dt_days, dx_km; taper=(:none,:hann))
_,        _,      powerv = komega_spectrum(v_slice, dt_days, dx_km; taper=(:none,:hann))
powerKE = power .+ powerv   # Pu+Pv, i.e. m^2/s^2*day*km, a genuine KE spectral density

# crop to non-negative frequencies only (the negative half is the redundant
# Hermitian mirror for this real input -- komega_spectrum() returns the full
# range for generality, but every plot below only ever shows freq>=0)
posf = findall(freq_cpd .>= 0)
freq_cpd = freq_cpd[posf]
power, powerv, powerKE = power[posf,:], powerv[posf,:], powerKE[posf,:]

# fold the discarded negative-frequency half back in: standard one-sided-PSD
# convention doubles every kept row EXCEPT freq=0 (its own unpaired,
# self-conjugate mirror), to conserve total variance/energy now that the
# exact-duplicate negative-frequency half is gone. Wavenumber is NOT folded
# the same way -- +k and -k at a GIVEN frequency are independent physical
# content (rightward vs leftward propagation), not redundant mirrors, so
# both are already kept and correctly summed as-is when integrating over k.
# (This is a uniform x2 scaling of every positive-frequency bin, so it does
# NOT change any relative comparison -- bound/free peak ratios, epsilon_k,
# GM slope shape -- made so far; it only matters for absolute energy units.)
fold = ones(length(freq_cpd)); fold[freq_cpd .> 0] .= 2
power, powerv, powerKE = power.*fold, powerv.*fold, powerKE.*fold

# convert k1/k2 (rad/m) to cycles/km for the same axis convention
k1_cpkm = k1/(2π) * 1e3
k1_mode2_cpkm = k1_mode2/(2π) * 1e3
k2_cpkm = k2/(2π) * 1e3
f1_cpd  = 24/T2       # D2 frequency in cpd
f2_cpd  = 2*f1_cpd    # HH/compound frequency in cpd

# colorrange: restrict to the top ~8 decades below the peak. With the full
# range (down to the FFT's numerical floor, ~1e-15 relative) the true peak
# and genuine-but-much-weaker broadband content render as similar "warm"
# colors, since log10 power here spans >20 decades -- the peak was there in
# the numbers the whole time (verified against a plain sort of power[:]),
# just not visually distinguishable without this
pmax = log10(maximum(power))

# paper-size figure: 11x11 cm at fontsize 11pt. Makie's `size` is in points
# (72 pt/inch) for this purpose; px_per_unit at save-time rasterizes it to
# ~300 dpi for print while keeping the 11cm/11pt physical proportions
cm_to_pt = 72/2.54
fig = Figure(size=(15*cm_to_pt, 11*cm_to_pt), fontsize=9)
ax = Axis(fig[1,1], title=string("u spectrum, ",fnames," (lat=",LAT,")"), titlesize=9,
    xlabel="wavenumber [km⁻¹]", ylabel="frequency [cpd]")
# zoomed in on the region where the dispersion curves are actually visible
xmax = 1/23
ymax = 9.0
kmax_disp = maximum(vcat(k_disp_nh_cpkm, k_disp_hy_cpkm))   # still used for reference/printouts elsewhere

# crop the heatmap's OWN data to the displayed k-range before plotting --
# on the 200m grid (mainnm=11) k_cpkm has ~8000 columns, but xlims below only
# shows ~70 of them; heatmap!() with the full 8000-wide array rasterizes
# ~11M cells in CairoMakie and silently fails to draw anything ELSE added to
# the figure afterward (colorbar, legend, dispersion curves, markers all
# vanished from the saved PNG with no error/warning) -- cropping first keeps
# the actual plotted resolution tiny regardless of the source grid's Nx
kidx = findall(-1e-9 .<= k_cpkm .<= xmax*1.02)
# k on the horizontal axis, frequency on the vertical: power is (Nt,Nx) =
# (length(freq_cpd),length(k_cpkm)), so transpose it to (Nx,Nt) to match
# Makie's heatmap!(x,y,Z) convention (size(Z)==(length(x),length(y)))
# colorrange narrowed to -3..7.5 (vs the full pmax-8..pmax) to push most of
# the background out of the deep-blue floor and show more mid-range texture
# colorrange relative to THIS run's pmax (not a hardcoded absolute value --
# that broke when the power units/magnitude changed after the dt*dx/df*dk
# normalization fix): offsets (-12.2, -1.7) reproduce the same relative
# "mild top saturation + wide bottom for texture" character Maarten
# calibrated earlier against the old (unnormalized) pmax~9.2 case (-3,7.5)
hm = heatmap!(ax, k_cpkm[kidx], freq_cpd, log10.(power[:,kidx])', colormap = Reverse(:Spectral), colorrange=(pmax-12.2,pmax-1.7))
Colorbar(fig[1,2], hm, label="log10 power [m²/s²·day·km]")
ylims!(ax, 0, ymax)
xlims!(ax, 0, xmax)   # positive k (rightward propagation) only

# x ticks as reciprocal-wavelength fractions (1/200, 1/100, ...) instead of
# raw cyc/km decimals; a reciprocal mapping on a LINEAR k-axis always crowds
# the long-wavelength (small-k) ticks together near zero, so candidates
# within minsep of the previous kept tick are dropped rather than plotted
# on top of each other. y ticks as whole cycles/day
wavelen_km_candidates = [2000,1000,500,200,100,50,25,20,15,10]
xtick_k_all = [1/L for L in wavelen_km_candidates]
inrange = findall(0 .< xtick_k_all .< xmax)
minsep = 0.08*xmax
xtick_keep = Int[]; last_k = -Inf
for i in inrange
    if xtick_k_all[i] - last_k >= minsep
        push!(xtick_keep, i); global last_k = xtick_k_all[i]
    end
end
ax.xticks = (xtick_k_all[xtick_keep], ["1/$(wavelen_km_candidates[i])" for i in xtick_keep])
ytick_vals = collect(1:floor(Int, ymax))
ax.yticks = (ytick_vals, string.(ytick_vals))

# mode-1/mode-2 dispersion curves (rightward branch, +k only), hydrostatic
# and nonhydrostatic, plus the inertial cutoff below which no IW mode exists
lines!(ax, k_disp_nh_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, label="nonhydrostatic (mode 1)")
lines!(ax, k_disp_hy_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 1)")
lines!(ax, k_disp_nh_cpkm2,  disp_freq_cpd, color=:gray50, linewidth=1.5, linestyle=:dash, label="nonhydrostatic (mode 2)")
lines!(ax, k_disp_hy_cpkm2,  disp_freq_cpd, color=:gray50, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 2)")
hlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")

# reference markers: (ω,k1), (2ω,2k1) bound, (2ω,k2) free -- open (hollow,
# transparent-fill) symbols so the heatmap shows through: circle for the two
# points that sit ON the dispersion curve (free waves), square for the
# off-curve bound-wave point
open_kw = (color=(:white,0.0), strokecolor=:black, strokewidth=1.5, markersize=12)
scatter!(ax, [k1_cpkm], [f1_cpd]; marker=:circle, open_kw...)
scatter!(ax, [2*k1_cpkm], [f2_cpd]; marker=:rect, open_kw...)
scatter!(ax, [k2_cpkm], [f2_cpd]; marker=:circle, open_kw...)
scatter!(ax, [NaN],[NaN]; marker=:circle, open_kw..., label="(ω, k1) / (2ω, k2) free")
scatter!(ax, [NaN],[NaN]; marker=:rect, open_kw..., label="(2ω, 2k1) bound")
axislegend(ax, position=:lt, labelsize=8, patchsize=(14,6), rowgap=2, patchlabelgap=4)

display(fig)
if figflag==1; save(string(dirfig,"komega_",fnames,".png"), fig; px_per_unit=300/72); end

println("epsilon_k (from the spectrum markers) = ", (k2_cpkm^2-(2*k1_cpkm)^2)/(2*k1_cpkm)^2)

# 1D slices: power(k) at the frequency bins nearest ω (M2) and 2ω -- the 2ω
# curve shows the bound (2k1, locked to the D2 harmonic) and free (k2, on the
# dispersion curve) waves as two separate peaks along k, once dk is fine
# enough to resolve them; the ω curve is shown alongside for comparison, with
# its own wavenumber k1 marked
idx1 = argmin(abs.(freq_cpd .- f1_cpd))
idx2 = argmin(abs.(freq_cpd .- f2_cpd))
fig2 = Figure(size=(700,450))
ax2 = Axis(fig2[1,1], title=string("power(k) at ω and 2ω — ",fnames," (lat=",LAT,")"),
    xlabel="wavenumber [cycles/km] (1/wavelength)", ylabel="log10 power [m²/s²·day·km]")
lines!(ax2, k_cpkm, log10.(power[idx1, :]), color=:seagreen, label=string("P(k) at ω (freq=",round(freq_cpd[idx1],digits=3)," cpd)"))
lines!(ax2, k_cpkm, log10.(power[idx2, :]), color=:black,    label=string("P(k) at 2ω (freq=",round(freq_cpd[idx2],digits=3)," cpd)"))
vlines!(ax2, [k1_cpkm],       color=:seagreen, linestyle=:dash, linewidth=1.5, label="k1 (M2 wavenumber, mode 1)")
vlines!(ax2, [k1_mode2_cpkm], color=:seagreen, linestyle=:dot,  linewidth=1.5, label="k1 mode 2 (M2 wavenumber, mode 2)")
vlines!(ax2, [2*k1_cpkm], color=:red,        linestyle=:dash, linewidth=1.5, label="2k1 (bound)")
vlines!(ax2, [k2_cpkm],   color=:dodgerblue, linestyle=:dash, linewidth=1.5, label="k2 (free)")
xlims!(ax2, 0, kmax_disp*1.05)
axislegend(ax2, position=:rt, labelsize=10)
display(fig2)
if figflag==1; save(string(dirfig,"komega_power_at_2omega_",fnames,".png"), fig2); end

# ---- KE = Pu+Pv versions of the same two figures, for direct comparison
# against the u-only ones above -- does adding v sharpen or blur the
# bound(2k1)/free(k2) separation at 2ω? downside: v also carries whatever
# subtidal (mesoscale/geostrophic) energy is present, which u alone is more
# free of since u is the along-domain propagation direction
pmaxKE = log10(maximum(powerKE))
figKE = Figure(size=(800,600))
axKE = Axis(figKE[1,1], title=string("k-ω spectrum of KE (Pu+Pv) — ",fnames," (lat=",LAT,")"),
    xlabel="wavenumber [cycles/km] (1/wavelength)", ylabel="frequency [cpd]")
hmKE = heatmap!(axKE, k_cpkm, freq_cpd, log10.(powerKE)', colormap = Reverse(:Spectral), colorrange=(pmaxKE-8,pmaxKE))
Colorbar(figKE[1,2], hmKE, label="log10 power [m²/s²·day·km]")
ylims!(axKE, 0, disp_fmax_cpd)
xlims!(axKE, 0, kmax_disp*1.05)
lines!(axKE, k_disp_nh_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, label="nonhydrostatic (mode 1)")
lines!(axKE, k_disp_hy_cpkm,  disp_freq_cpd, color=:black, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 1)")
lines!(axKE, k_disp_nh_cpkm2, disp_freq_cpd, color=:dodgerblue, linewidth=1.5, label="nonhydrostatic (mode 2)")
lines!(axKE, k_disp_hy_cpkm2, disp_freq_cpd, color=:dodgerblue, linewidth=1.5, linestyle=:dot, label="hydrostatic (mode 2)")
hlines!(axKE, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
scatter!(axKE, [k1_cpkm], [f1_cpd], color=:white, marker=:circle, markersize=14, strokecolor=:black, strokewidth=1.5)
scatter!(axKE, [2*k1_cpkm], [f2_cpd], color=:white, marker=:xcross, markersize=14, strokecolor=:black, strokewidth=1.5)
scatter!(axKE, [k2_cpkm], [f2_cpd], color=:white, marker=:utriangle, markersize=14, strokecolor=:black, strokewidth=1.5)
scatter!(axKE, [NaN],[NaN], color=:white, marker=:circle, strokecolor=:black, strokewidth=1.5, label="(ω, k1)")
scatter!(axKE, [NaN],[NaN], color=:white, marker=:xcross, strokecolor=:black, strokewidth=1.5, label="(2ω, 2k1) bound")
scatter!(axKE, [NaN],[NaN], color=:white, marker=:utriangle, strokecolor=:black, strokewidth=1.5, label="(2ω, k2) free")
axislegend(axKE, position=:lt, labelsize=11)
display(figKE)
if figflag==1; save(string(dirfig,"komega_KE_",fnames,".png"), figKE); end

figKE2 = Figure(size=(700,450))
axKE2 = Axis(figKE2[1,1], title=string("KE=Pu+Pv power(k) at ω and 2ω — ",fnames," (lat=",LAT,")"),
    xlabel="wavenumber [cycles/km] (1/wavelength)", ylabel="log10 power [m²/s²·day·km]")
lines!(axKE2, k_cpkm, log10.(powerKE[idx1, :]), color=:seagreen, label=string("P(k) at ω (freq=",round(freq_cpd[idx1],digits=3)," cpd)"))
lines!(axKE2, k_cpkm, log10.(powerKE[idx2, :]), color=:black,    label=string("P(k) at 2ω (freq=",round(freq_cpd[idx2],digits=3)," cpd)"))
vlines!(axKE2, [k1_cpkm],       color=:seagreen, linestyle=:dash, linewidth=1.5, label="k1 (M2 wavenumber, mode 1)")
vlines!(axKE2, [k1_mode2_cpkm], color=:seagreen, linestyle=:dot,  linewidth=1.5, label="k1 mode 2 (M2 wavenumber, mode 2)")
vlines!(axKE2, [2*k1_cpkm], color=:red,        linestyle=:dash, linewidth=1.5, label="2k1 (bound)")
vlines!(axKE2, [k2_cpkm],   color=:dodgerblue, linestyle=:dash, linewidth=1.5, label="k2 (free)")
xlims!(axKE2, 0, kmax_disp*1.05)
axislegend(axKE2, position=:rt, labelsize=10)
display(figKE2)
if figflag==1; save(string(dirfig,"komega_KE_power_at_2omega_",fnames,".png"), figKE2); end

# frequency spectrum P(ω): integrate the KE=Pu+Pv k-omega spectrum over the
# FULL wavenumber range (power/powerKE were never cropped -- only the display
# k_cpkm[kidx] subset used for the heatmaps above was) to check for the
# classic GM continuum slope, P(ω) ~ ω^-2
dk_cpkm = k_cpkm[2] - k_cpkm[1]
Pomega = vec(sum(powerKE, dims=2)) .* dk_cpkm
# x-range: 0.1 to 48 cpd (per Maarten's request -- reads "1/48 cpd" but that
# would be BELOW 0.1, and the requested tick list tops out at 48, so treating
# it as "48 cpd" / a typo for the upper bound, not its reciprocal)
xlo, xhi = 0.1, 48.0
# crop to the displayed range BEFORE plotting, not just via xlims!() --
# Makie's log-scale y-autolimits are computed from the full data passed to
# lines!(), so leaving in the full freq_cpd (up to ~144 cpd Nyquist) blows the
# y-axis out to 1e-20..1e20 even though xlims! only shows xlo-xhi
ipos = findall((freq_cpd .>= xlo) .& (freq_cpd .<= xhi))

# ω^-2 reference line (the GM continuum slope), vertically anchored to match
# the data at a frequency clear of the discrete tidal harmonics (2.6 cpd,
# between M2 at 1.93 and 2ω at 3.87 cpd) so the anchor reflects the
# background continuum, not a harmonic spike
f_anchor = 2.6
i_anchor = argmin(abs.(freq_cpd[ipos] .- f_anchor))
gm_ref = Pomega[ipos][i_anchor] .* (freq_cpd[ipos] ./ freq_cpd[ipos][i_anchor]).^(-2)

# ω^-3 reference line -- this one anchored AT the M2 (fundamental) harmonic
# peak itself, to test whether the DISCRETE harmonic peaks (M2, 2xM2, 3xM2,
# ...) decay envelope-to-envelope like ω^-3, a different question from the
# continuum's own slope between the peaks
i_m2 = argmin(abs.(freq_cpd[ipos] .- f1_cpd))
harm_ref = Pomega[ipos][i_m2] .* (freq_cpd[ipos] ./ freq_cpd[ipos][i_m2]).^(-3)

figPom = Figure(size=(700,450))
axPom = Axis(figPom[1,1], title=string("P(ω) = ∫P(k,ω)dk — ",fnames," (lat=",LAT,")"),
    xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)
lines!(axPom, freq_cpd[ipos], Pomega[ipos], color=:black, label="P(ω), KE=Pu+Pv")
lines!(axPom, freq_cpd[ipos], gm_ref, color=:red, linestyle=:dash, label="ω⁻² (continuum) reference")
lines!(axPom, freq_cpd[ipos], harm_ref, color=:purple, linestyle=:dashdot, label="ω⁻³ (harmonic envelope) reference")
vlines!(axPom, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
vlines!(axPom, [f1_cpd-fcor_cpd], color=:orange, linestyle=:dash, label="M2-f")
xlims!(axPom, xlo, xhi)
# FIXED y-limits (not per-run percentile-based) so 10.37/11.37/12.37 share
# the exact same axes for a direct GM-vs-no-GM comparison. Chosen from
# 12.37's own range (ylo=quantile 0.03 ~2.3e-8, yhi=max*2 ~0.24) with a bit
# of margin -- floored at all rather than the strict per-run minimum since
# Pomega genuinely dips to ~1e-26 at some frequencies (a real spectral null,
# not a bug) that would otherwise blow the log-scale axis out per-run
ylims!(axPom, 1e-8, 0.3)
pomega_xticks = [0.5,1,2,3,4,6,8,10,12,24,48]
axPom.xticks = (pomega_xticks, string.(pomega_xticks))
axislegend(axPom, position=:lb, labelsize=10)
display(figPom)
if figflag==1; save(string(dirfig,"komega_Pomega_",fnames,".png"), figPom); end
