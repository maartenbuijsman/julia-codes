#= IW_komega_2x2_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: a 2x2 panel of wavenumber-frequency (k-omega) spectra of surface
u, 18 x 18 cm, fontsize 10, ONE shared colorbar and zero gap between panels
(dsh = dsv = 0).

  (a) 11.63   200 m NH,   40°N, 25 kW m⁻¹, tide only, fixed N²(50°N)
  (b) 11.34   200 m NH, 28.8°N, 25 kW m⁻¹, tide only
  (c) GMSER.34  200 m NH, 28.8°N, 25 kW m⁻¹, GM + tide (GMSER = 16: GM81 IC at 1x GM81
                over days 10-20; 15: the earlier ~3x GM IC, output names without a tag)
  (d) 11.67   200 m NH,  2.5°N, 50 kW m⁻¹, tide only

Panels run CLOCKWISE from the top left -- (a) top left, (b) top right, (c)
bottom right, (d) bottom left -- so that (b) and (c), the same latitude and
forcing without and with the GM background, sit one above the other in the
right-hand column where they can be compared directly. Latitude then decreases
down the left column.

(a) is poleward of the critical latitude, where PSI is forbidden (ω/2 < f), and
carries the fixed N²(50°N) profile rather than its own latitude's; it separates
the bound and free M4 into two distinct peaks more cleanly than any other panel.
(b)/(c) isolate what the background continuum does at the M2 critical latitude.
(d) is the strongest forcing at the lowest latitude, where the harmonic cascade
(2ω, and 3ω at 5.80 cpd) is most visible and where epsilon_k is so small that
the two M4 arrows coincide.

The companion figure IW_snapshot_4panel_ppr.jl uses the SAME four runs with the
SAME panel letters, so (a)-(d) mean the same run in both.

The two M4 (2ω) reference points are marked with ARROW TIPS rather than
symbols, so nothing of the spectrum is hidden under a marker: grey points at
the BOUND wave (2ω, 2k1) -- phase-locked to the primary, wavenumber the sum of
the interacting wavenumbers whether or not that satisfies the dispersion
relation -- and black at the FREE wave (2ω, k2), the genuine mode-1 solution at
2ω, which sits on the dispersion curve. The gap between the two tips IS
epsilon_k = (k2²-(2k1)²)/(2k1)² made visible.

The GM runs are the mainnm 15 (redistribution IC) series -- the same series the
paper's Te figure uses. Their mainnm 13 twins (k-clamp IC) exist at the same
runnm if that comparison is ever wanted.

ONE colorbar means ONE colorrange for all four panels, so the panels are
directly comparable in absolute power -- which is the point, but note that (c)
carries 4x the forcing flux of the others and is correspondingly brighter. The
range is set from the largest peak over the four panels, with the same
(-12.2, -1.7) offsets the single-panel version (IW_komega_spectrum.jl) uses.

Method is unchanged from IW_komega_spectrum.jl: full-resolution 2D FFT of
surface u over x = 100-1800 km and a whole number of M2 periods at the end of
the record (no time taper needed; Hann in space), then a DISPLAY crop of the
already-computed (freq,k) power matrix before heatmap!(), never an input
filter.

Spectra are cached to dirout/komega_2x2_ppr_cache.jld2 -- the 2D FFTs are the
expensive part and cosmetic edits should not pay for them twice. Set
RECOMPUTE = true to force a re-read of the NetCDF files.
=#

println("number of threads is ", Threads.nthreads())

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, DSP, FFTW

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirsim    = string(pth0, "IW/")
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag   = 1
RECOMPUTE = false

const T2 = 12 + 25.2/60           # M2 period [hours]
const ω  = 2π/(T2*3600)           # M2 frequency [rad/s]

# analysis window -- as in IW_komega_spectrum.jl. The x window sets dk =
# 1/(Nx*dx), which is what has to resolve k2 - 2k1, so it is kept as wide as
# the clean domain allows (source at ~80 km, sponge past ~1850 km).
const xleft_km   = 100.0
const xright_km  = 1800.0
const tlast_days = 10.0
const T2_days    = T2/24
const n_periods  = floor(Int, tlast_days/T2_days)
const tdur_days  = n_periods*T2_days

# display window of the heatmaps
const KMAX = 1/23.0               # cyc/km
const FMAX = 8.0                  # cpd -- 6 covers f, ω, 2ω and 3ω; 8 adds 4ω
# the cache is cropped at FCACHE, NOT at FMAX, so changing the displayed
# frequency range is a free re-plot rather than a re-read of four 82 GB files.
# Raising FCACHE is the only edit that forces a recompute.
const FCACHE = 12.0

# FOLDK=false shows only k >= 0, i.e. RIGHTWARD-propagating energy: the tide
# radiates rightward from the source, so this is the natural view for it, but
# it discards whatever travels leftward -- and the GM background is close to
# isotropic, so roughly half of its energy is simply not on the page.
# FOLDK=true sums P(+k) + P(-k) into a spectrum of |k| (k=0 unpaired), which
# conserves total energy but throws away the propagation direction, so the
# rightward tidal beam and the two-sided GM continuum can no longer be told
# apart. Written to its own filename so the two can be compared.
const FOLDK = true

# USE_KE=true plots Pu+Pv, the same quantity the P(ω) line spectra use, so the
# two sets of figures are consistent. The original single-panel version used u
# alone on the grounds that u is the along-domain propagation direction and is
# therefore freer of subtidal/geostrophic energy, which v carries; that is a
# real advantage for the bound/free M4 story but it makes the two figure sets
# incomparable. Both are computed and cached, so this is a free switch.
const USE_KE = true
# white backing box for the in-panel label, as a fraction of each panel's own
# x/y range (every panel shares the same range, so one set of numbers serves)
const BOXX0        = 0.015                 # left edge
const BOXY0, BOXY1 = 0.860, 0.995          # bottom, top
# the width is set per panel from the longest label line: BOXPAD of padding
# plus BOXCHR per character, calibrated so a 20-character line lands at the
# 0.525 width that looked right before the labels varied in length
const BOXPAD, BOXCHR = 0.055, 0.0235

const ARRLEN = 0.70               # shaft length of the M4 arrow tips [cpd]
const ARRGAP = 0.18               # clearance the tip stops short of 2ω [cpd],
                                  # so the arrow points AT the bound/free peak
                                  # without covering it

# (mainnm, runnm, panel letter, short description, stratification note).
# The note is empty when the run uses its own latitude's N2 (N2source
# "zonalmean"), and names the profile when it does not.
const GMSER = 16   # GM series of panel (c): 16 (GM81 IC, 1x GM81 over days 10-20) or 15
RUNS = [(11, 63, "(a)", "tide only", "N²(50°N)"),
        (11, 34, "(b)", "tide only", ""),
        (GMSER, 34, "(c)", "GM + tide", ""),
        (11, 67, "(d)", "tide only", "")]

# panels are laid out CLOCKWISE from the top left, not row by row, so that the
# two 28.8°N runs -- the tide-only / GM pair that has to be compared directly --
# end up stacked in the right-hand column instead of split across a diagonal.
const POS = [(1,1), (1,2), (2,2), (2,1)]

CACHE = string(dirout, "komega_2x2_ppr_cache.jld2")

## --- spectra -----------------------------------------------------------------
"""
Full-resolution 2D FFT of surface u for one run, returned already cropped to
the displayed (k, freq) window together with everything the overlay needs.
"""
function komega_panel(mainnm, runnm)
    row    = get_runs(mainnm, [runnm])[1]
    LAT    = row.lat
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    println("\n", fnames, "  lat = ", LAT, ", F = ", row.Flux/1e3, " kW/m ", "-"^30)

    # mode-1 wavenumbers at ω and 2ω from the dispersion relation
    @load string(dirforce, n2_filename(row)) N2w zfw
    fcor   = coriolis(LAT)
    nonhyd = row.DX < 500 ? 1 : 0
    k1 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω,  nonhyd)[1][1]
    k2 = sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, 2ω, nonhyd)[1][1]

    # mode-1 and mode-2 dispersion curves, swept from just above the inertial
    # cutoff (sqrt(ω²-f²) is a DomainError below f) to FCACHE, so the curves
    # are long enough for any FMAX the plot is later re-cropped to
    fcor_cpd = fcor/(2π)*86400
    Nmax_cpd = sqrt(maximum(N2w))/(2π)*86400
    fhi      = min(FCACHE, Nmax_cpd*0.98)
    dfreq    = collect(range(fcor_cpd*1.001, fhi, length=300))
    dom      = dfreq .* (2π/86400)
    knh1 = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, o, 1)[1][1]/(2π)*1e3 for o in dom]
    knh2 = [sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, o, 1)[1][2]/(2π)*1e3 for o in dom]

    # surface u over the x/t window only (NCDatasets partial read)
    ds   = NCDataset(string(dirsim, fnames, ".nc"), "r")
    xf   = ds["x_faa"][:]
    tday = ds["time"][:] ./ 86400
    Ix = findall(xleft_km*1e3 .<= xf .<= xright_km*1e3)
    It = findall(tday .>= tday[end]-tdur_days)
    # keep n odd in both dimensions: fftfreq labels the Nyquist bin -fs/2 for
    # even n but +fs/2 for odd n, so an even n silently mislabels it negative
    isodd(length(Ix)) || (Ix = Ix[1:end-1])
    isodd(length(It)) || (It = It[1:end-1])

    dx = xf[2]-xf[1]
    # dt from WITHIN the window: the output cadence changes partway through the
    # run, and the first two samples of the whole record give the wrong one
    dt = (tday[It[2]]-tday[It[1]])*86400
    maximum(abs.(diff(tday[It]).*86400 .- dt)) > 1e-3*dt &&
        @warn "time window is not uniformly sampled -- frequency axis will be wrong"
    @printf("  x: %.0f-%.0f km (%d pts, dx=%.0f m)   t: %.2f-%.2f d (%d pts, dt=%.0f s)\n",
        xf[Ix[1]]/1e3, xf[Ix[end]]/1e3, length(Ix), dx,
        tday[It[1]], tday[It[end]], length(It), dt)

    # u is on x_faa and v on x_caa, so a shared index puts the v point half a
    # cell (100 m) from the u point -- the same convention IW_komega_spectrum.jl
    # and the P(ω) scripts use, kept so the figures stay comparable
    Nz = size(ds["u"], 2)
    u  = permutedims(ds["u"][Ix, Nz, It], (2,1))    # (Nt, Nx), surface, full 200 m
    v  = permutedims(ds["v"][Ix, Nz, It], (2,1))
    close(ds)

    # no time taper (tdur_days is a whole number of M2 periods, so the record
    # wraps continuously); Hann in space
    freq, k, Pu = komega_spectrum(u, dt/86400, dx/1e3; taper=(:none,:hann))
    _,    _, Pv = komega_spectrum(v, dt/86400, dx/1e3; taper=(:none,:hann))

    # keep freq >= 0 and fold the redundant Hermitian half back in (one-sided
    # PSD: double every kept row except freq=0)
    posf = findall(freq .>= 0)
    freq = freq[posf]
    fold = ones(length(freq)); fold[freq .> 0] .= 2
    Pu = Pu[posf, :] .* fold
    Pv = Pv[posf, :] .* fold
    PK = Pu .+ Pv                      # KE spectral density, Pu+Pv

    # k-FOLD: P(|k|) = P(+k) + P(-k), with k=0 unpaired. k is ascending and
    # symmetric about a single zero bin (the point count was forced odd), so
    # the partner of column mid+j is column mid-j. Without this, only the
    # rightward half is ever displayed and the (near-isotropic) GM continuum
    # loses about half its energy off the left of the page.
    mid = argmin(abs.(k))
    np  = length(k) - mid
    kf  = k[mid:end]
    function foldk(P)
        Pf = similar(P, size(P,1), np+1)
        Pf[:,1] .= P[:,mid]
        for j in 1:np
            Pf[:,1+j] .= P[:,mid+j] .+ P[:,mid-j]
        end
        return Pf
    end
    Puf, PKf = foldk(Pu), foldk(PK)

    # DISPLAY crop of the already-computed matrix -- on the 200 m grid k has
    # ~8500 columns but only ~70 are shown; heatmap!() with the full array
    # rasterizes ~11M cells and silently drops everything drawn after it
    ki  = findall(-1e-9 .<= k    .<= KMAX*1.02)
    kif = findall(-1e-9 .<= kf   .<= KMAX*1.02)
    fi  = findall(-1e-9 .<= freq .<= FCACHE*1.02)
    cropu(P, kk)  = log10.(P[fi, kk])'

    return (; mainnm, runnm, lat=LAT, flux=row.Flux, fnames,
              k=k[ki], kf=kf[kif], freq=freq[fi],
              logP   = cropu(Pu,  ki),  pmax   = maximum(Pu),
              logPf  = cropu(Puf, kif), pmaxf  = maximum(Puf),
              logK   = cropu(PK,  ki),  pmaxK  = maximum(PK),
              logKf  = cropu(PKf, kif), pmaxKf = maximum(PKf),
              k1_cpkm=k1/(2π)*1e3, k2_cpkm=k2/(2π)*1e3,
              f1_cpd=24/T2, fcor_cpd, dfreq, knh1, knh2)
end

# cache keyed by (mainnm, runnm), NOT by position: the panel order is a
# presentation choice that gets rearranged, and a positional cache would then
# silently pair a spectrum with the wrong panel's overlay
CACHED = Dict{Tuple{Int,Int}, Any}()
if !RECOMPUTE && isfile(CACHE)
    jldopen(CACHE, "r") do f
        # a cache written at a lower FCACHE was cropped short and cannot serve
        # a taller FMAX, so it is discarded rather than silently plotted
        if haskey(f, "BYRUN") && haskey(f, "FCACHE") && f["FCACHE"] >= FCACHE
            merge!(CACHED, f["BYRUN"])
        else
            println("cache was built at a lower frequency crop -- recomputing")
        end
    end
end
PAN = map(RUNS) do r
    key = (r[1], r[2])
    # an entry written before the k-folding was added has no logPf field and
    # cannot be folded after the fact -- the negative-k half was never stored
    (haskey(CACHED, key) && haskey(CACHED[key], :logKf)) ||
        (CACHED[key] = komega_panel(r[1], r[2]))
    CACHED[key]
end
jldsave(CACHE; BYRUN=CACHED, FCACHE=FCACHE)
println("\nspectra cache (", length(CACHED), " runs) -> ", CACHE,
        "   (set RECOMPUTE=true to redo)")

# ONE colorrange for all four panels, so they are comparable in absolute power.
# The top is tied to the largest peak among them (the single-panel version's
# -1.7 offset, mild top saturation). The floor is an ABSOLUTE log10 level:
# pmax-12.2 puts it near the FFT noise floor and leaves the tide-only panel
# almost entirely in the deep blue; raising it to -8 spends the colormap on the
# range the physics actually occupies. Set CMIN = nothing for the old behaviour.
const CMIN = -8.0
pk(p) = USE_KE ? (FOLDK ? p.pmaxKf : p.pmaxK) : (FOLDK ? p.pmaxf : p.pmax)
pmax_all = log10(maximum(pk(p) for p in PAN))
const CRANGE = (CMIN === nothing ? pmax_all-12.2 : CMIN, pmax_all-1.7)
@printf("\nshared colorrange = (%.2f, %.2f)  [log10 power], global peak %.2f\n",
        CRANGE[1], CRANGE[2], pmax_all)

## --- figure: 18 x 18 cm, fontsize 10 -----------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size=(18*cm_to_pt, 18*cm_to_pt), fontsize=10)

# x ticks as reciprocal wavelengths. A reciprocal mapping on a LINEAR k-axis
# crowds the long-wavelength ticks near zero, so these are hand-picked rather
# than swept; both end ticks are kept clear of the panel edges, because with
# colgap = 0 a tick label at x=KMAX would collide with the next panel's label
# at x=0. Likewise the y ticks omit 0 and FMAX, which with rowgap = 0 would
# print on top of each other across the row boundary.
wl = [100, 50, 30]
XT = ([1/L for L in wl], ["1/$L" for L in wl])
YT = (collect(1:floor(Int, FMAX)-1), string.(1:floor(Int, FMAX)-1))

axs = Axis[]; hms = Any[]
for (n, (p, r)) in enumerate(zip(PAN, RUNS))
    i, j = POS[n]                              # clockwise from the top left
    ax = Axis(fig[i, j], xticks=XT, yticks=YT,
        xlabel = i == 2 ? (FOLDK ? "|wavenumber| [km⁻¹]" : "wavenumber [km⁻¹]") : "",
        ylabel = j == 1 ? "frequency [cpd]"   : "",
        xticklabelsvisible = i == 2, yticklabelsvisible = j == 1)
    push!(axs, ax)
    # second display crop, from the cached FCACHE range down to FMAX: cheaper
    # than re-running the FFT and keeps the rasterized cell count small
    fj = findall(p.freq .<= FMAX*1.02)
    kk = FOLDK ? p.kf : p.k
    PP = USE_KE ? (FOLDK ? p.logKf : p.logK) : (FOLDK ? p.logPf : p.logP)
    push!(hms, heatmap!(ax, kk, p.freq[fj], PP[:, fj],
        colormap=Reverse(:Spectral), colorrange=CRANGE))

    # mode-1 and mode-2 nonhydrostatic dispersion curves, and f
    lines!(ax, p.knh1, p.dfreq, color=:black,  linewidth=1.2)
    lines!(ax, p.knh2, p.dfreq, color=:gray35, linewidth=1.2, linestyle=:dash)
    hlines!(ax, [p.fcor_cpd], color=:gray20, linestyle=:dot, linewidth=1.2)

    # the two M4 (2ω) reference wavenumbers as arrow tips coming down from
    # above -- grey = bound (2k1, off the dispersion curve by construction),
    # black = free (k2, on it). Arrows rather than symbols so the spectrum
    # under the point stays visible; the tip separation IS epsilon_k. Both stop
    # ARRGAP short of 2ω so neither covers its own peak. All arrows are the
    # same length in every panel, so shaft length carries no information and
    # only the horizontal separation of the tips does. The cost is panel (c):
    # at 2.5°N epsilon_k = 0.014, so 2k1 and k2 differ by less than a shaft
    # width and the grey arrow is hidden under the black one -- which is the
    # result there, the bound and free M4 being very nearly the same wave.
    akw = (shaftwidth=1.6, tipwidth=7, tiplength=6)
    for (kk, cc) in ((2p.k1_cpkm, :gray45),   # bound, at 2k1
                     ( p.k2_cpkm, :black))    # free,  at k2
        arrows2d!(ax, [Point2f(kk, 2p.f1_cpd + ARRGAP + ARRLEN)], [Vec2f(0, -ARRLEN)];
            color=cc, akw...)
    end

    xlims!(ax, 0, KMAX); ylims!(ax, 0, FMAX)    # positive k = rightward only

    # panel letter, run id, latitude and forcing in one white-backed block in
    # the top-left corner -- there is room for it, and it saves the reader
    # jumping between a corner letter and a separate caption line. Makie's
    # `text!` has no background attribute, so the backing is an explicit poly
    # in data coordinates (the same fraction-of-range trick text_fignum! uses);
    # the limits above must therefore be frozen before this, or the poly would
    # push the axis range out.
    l1 = @sprintf("%s %d.%d   %.1f°N", r[3], p.mainnm, p.runnm, p.lat)
    l2 = @sprintf("%s, %g kW m⁻¹", r[4], p.flux/1e3) *
         (isempty(r[5]) ? "" : ", " * r[5])
    # width from the longer of the two lines, so a panel carrying a
    # stratification note gets a wider box instead of overflowing a fixed one
    bw = BOXPAD + BOXCHR*maximum(length.((l1, l2)))
    poly!(ax, Rect2f(BOXX0*KMAX, BOXY0*FMAX, bw*KMAX, (BOXY1-BOXY0)*FMAX),
        color=:white, strokewidth=0)
    text!(ax, BOXX0*KMAX + 0.012*KMAX, BOXY1*FMAX - 0.012*FMAX,
        align=(:left,:top), fontsize=9, color=:black, font=:bold,
        text=string(l1, "\n", l2))
end

Colorbar(fig[1:2, 3], hms[1], label = USE_KE ? "log₁₀ KE power  [m² s⁻² day km]" : "log₁₀ power  [m² s⁻² day km]", labelsize=10,
    ticklabelsize=9, width=10)

# one shared legend for the overlay, below the panels
el(; kw...) = LineElement(; kw...)
tip(c)      = MarkerElement(marker=:dtriangle, color=c, markersize=9)
Legend(fig[3, 1:3],
    [el(color=:black, linewidth=1.2),
     el(color=:gray35, linewidth=1.2, linestyle=:dash),
     el(color=:gray20, linewidth=1.2, linestyle=:dot),
     tip(:gray45), tip(:black)],
    ["mode 1 (nonhydrostatic)", "mode 2 (nonhydrostatic)", "inertial frequency f",
     "(2ω, 2k₁) bound M4", "(2ω, k₂) free M4"],
    orientation=:horizontal, nbanks=2, framevisible=false, labelsize=9,
    patchsize=(16,8), colgap=10, rowgap=2)

colgap!(fig.layout, 1, 0)      # dsh = 0 between the two panel columns
colgap!(fig.layout, 2, 6)      # panels -> colorbar
rowgap!(fig.layout, 1, 0)      # dsv = 0 between the two panel rows
rowgap!(fig.layout, 2, 4)      # panels -> legend

display(fig)
if figflag == 1
    # the 6 cpd version keeps the plain name; any other FMAX is written beside
    # it so the two can be compared rather than one overwriting the other
    tag  = string(USE_KE ? "_KE" : "", FOLDK ? "_foldk" : "", GMSER == 15 ? "" : "_GM$(GMSER)")
    fout = string(dirfig, FMAX == 6.0 ? string("komega_2x2_ppr", tag, ".png") :
                          @sprintf("komega_2x2_ppr_%gcpd%s.png", FMAX, tag))
    savefig300(fout, fig)
    println("\nsaved ", fout)
end

## --- numbers ------------------------------------------------------------------
println("\n", "="^92)
println(rpad("panel",7), rpad("run",9), rpad("lat",7), rpad("F [kW/m]",10),
        rpad("f [cpd]",10), rpad("ω/2 [cpd]",11), rpad("PSI?",7), "εk = (k₂²-(2k₁)²)/(2k₁)²")
println("="^92)
for (p, r) in zip(PAN, RUNS)
    εk = (p.k2_cpkm^2 - (2p.k1_cpkm)^2)/(2p.k1_cpkm)^2
    println(rpad(r[3],7), rpad(@sprintf("%d.%d",p.mainnm,p.runnm),9),
        rpad(@sprintf("%.1f",p.lat),7), rpad(@sprintf("%.1f",p.flux/1e3),10),
        rpad(@sprintf("%.3f",p.fcor_cpd),10), rpad(@sprintf("%.3f",p.f1_cpd/2),11),
        rpad(p.f1_cpd/2 >= p.fcor_cpd ? "yes" : "no", 7), @sprintf("%+.4f", εk))
end
