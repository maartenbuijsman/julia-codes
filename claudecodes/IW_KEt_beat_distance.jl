#= IW_KEt_beat_distance.jl
Maarten Buijsman, USM DMS, 2026-8-28
Measure the PSI/wave-wave beat distance directly from the simulated KEt(x)
profile, for comparison against the analytical beat distance (Tbeat*Cg1)
computed in IW_nondim_params.jl's (3,2) subplot.

Borrows the run-selection boilerplate and xc/KEt loading from
IW_analysis_energy_2000km.jl, and the Butterworth filter from
functions/butter_filters.jl.

METHOD (settled interactively 2026-8-27 -- see notes below for what failed):
  1. restrict to x in [100, 1800] km. Everything outside is unusable: the
     Gaussian forcing/nudging patch sits near x=80 km (gausW_center in
     IW_flux_LAT_2000km_bash_cuda.jl) and the sponge/relaxation layer eats
     the last ~200 km, so a "peak" out there is a boundary artifact, not a
     beat crest.
  2. mirror-pad that restricted segment on both ends, THEN lowpass, then
     discard the mirrored ends. Padding after restricting matters: mirroring
     the full 0-2000 km profile folds the near-zero ramp-up/sponge edges onto
     themselves, making a sharp cusp at the seam that the filter rings on
     badly (the filtered result dove below zero there).
  3. Butterworth lowpass, 150 km cutoff, order 4. This removes the ~50-150 km
     ripples riding on the beat envelope. 100 km was tried first and left
     residual ripple leaking through the (imperfect) stopband -- e.g. at
     lat=28.8 that produced a spurious min/max pair near x=950 km with an
     amplitude of ~0.5 out of a ~3066 baseline (~0.02%).
  4. find extrema as sign changes of the derivative of the FILTERED curve.
     No amplitude threshold, no minimum-separation merge: the filtering has
     already removed everything that isn't a genuine turning point, so any
     extra criterion just deletes real structure (an earlier prominence +
     200 km-merge version silently dropped most of the real bumps at
     lat=35/40, which only showed up on a stretched y-axis).
  5. the left edge is always a max (the injection peak), so prepend x=100 km
     as a max if the first detected extremum isn't already one.
  5b. LOW-LATITUDE ONLY (lat < 20): drop any consecutive extrema pair closer
     than 300 km, then snap each surviving min to the lowest point within
     +/-500 km (deepest interior LOCAL min, so it cannot slide onto a monotone
     window edge -- see refine_lowlat). The long-wavelength, broad-flat-trough profiles down there can
     still carry a couple of tiny wiggles in the trough that the derivative
     test reports as a genuine min/max pair (11.66-78 at lat=10: min@796 and
     max@892, 96 km apart, pulling Lbeat to 690 km vs ~1400 km for its
     neighbours). NOT applied at higher latitudes, where the real beat
     wavelength itself approaches 300 km. See refine_lowlat() below.
  6. beat distance = 2 * MEDIAN(spacing between consecutive extrema), since
     each consecutive max->min (or min->max) step is half a beat wavelength.
     Median, not mean: at 40-50N the last one or two gaps are stretched by
     the low group speed there (e.g. lat=45: 104,92,96,96,96,100,104,108,
     124,144,136,276 km -- that final 276 is the artifact). The median
     ignores those automatically and gives results identical to hard-cutting
     everything past x=1500 km, but without discarding otherwise-good data.

Saves one beatdist_AMZexptXX.YY.jld2 per run (same dirout as nondim_*.jld2),
to be loaded back into IW_nondim_params.jl for the (3,2) comparison panel.
=#

println("number of threads is ",Threads.nthreads())

using Printf, CairoMakie, JLD2, ColorSchemes, Statistics, DSP

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirout = string(pth0,"diagout/");
    dirfig = string(pth0,"figs/");
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"butter_filters.jl"))      # bandpass/lowhighpass_butter
include(string(dirparams,"run_master.jl"))         # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

figflag = 1   # save the extrema-detection diagnostic figure
savefl  = 1   # save the per-run beatdist_*.jld2

# analysis window -- see step 1 in the header
const xleft  = 100e3   # m
const xright = 1800e3  # m
const Tcut   = 150e3   # m, lowpass cutoff wavelength
const Nord   = 4       # Butterworth order

# run-ID selection: same block convention as IW_analysis_energy_2000km.jl /
# IW_nondim_params.jl -- swap the active runnms line to pick a different block
mainnm  = 10
runnms  = collect(1:13)   # varying  N2 MERCATOR             F=12.5 kW/m
#runnms  = collect(27:39) # varying  N2 MERCATOR              F=25   kW/m
#runnms  = collect(40:52) # constant N2 MERCATOR 2.5N        F=25   kW/m
#runnms  = collect(53:65) # constant N2 MERCATOR 50N         F=25   kW/m
#runnms  = collect(66:78) # constant N2 MERCATOR             F=50   kW/m

runs = get_runs(mainnm, runnms)
LATS = [r.lat for r in runs]
fnum = string(mainnm,".",runnms[1],"-",runnms[end])

# restrict to [xleft,xright], mirror-pad, lowpass, drop the mirrored ends.
# Returns the indices into the ORIGINAL xc that were kept, plus the filtered
# values on those indices.
function lowpass_mirror_restricted(xc, y, xleft, xright, Tcut, dx, N)
    Ivalid = findall(xleft .<= xc .<= xright)
    ys = y[Ivalid]
    n  = length(ys)
    ypad = vcat(reverse(ys[2:end]), ys, reverse(ys[1:end-1]))
    yf   = lowhighpass_butter(ypad, Tcut, dx, N, "low")[n:2n-1]
    return Ivalid, yf
end

# every local extremum of the (already de-noised) curve, as a sign change of
# the derivative; the left edge is forced to be a max (the injection peak)
function find_extrema_by_slope(y)
    dy = diff(y)
    idxs = Int[]; types = Symbol[]
    for j in 2:length(dy)
        if dy[j-1] > 0 && dy[j] <= 0
            push!(idxs, j); push!(types, :max)
        elseif dy[j-1] < 0 && dy[j] >= 0
            push!(idxs, j); push!(types, :min)
        end
    end
    if isempty(idxs) || types[1] != :max
        pushfirst!(idxs, 1); pushfirst!(types, :max)
    end
    return idxs, types
end

# extra cleanup for the LOW-LATITUDE runs only (lat < LATLOW). There the beat
# wavelength is long (>~1000 km) and the trough is broad and flat, so the
# filtered curve can still carry a couple of tiny wiggles down in the trough
# that the derivative test dutifully reports as a min/max pair -- e.g. 11.66-78
# at lat=10 had min@796 and max@892, only 96 km apart, which dragged the
# median half-wavelength (and hence Lbeat) down to 690 km vs ~1400 km for its
# neighbours. Two rules, applied in this order:
#   A. while the closest consecutive pair is < MINSEP apart, drop that PAIR
#      (dropping both keeps the max/min alternation intact)
#   B. snap each surviving min to the actual lowest point within +/- WINMIN of
#      it, so a min that landed on the shoulder of a broad flat trough gets
#      moved to the real bottom
# Deliberately NOT applied at higher latitudes: there the genuine beat
# wavelength itself approaches (and drops below) MINSEP, so rule A would eat
# real structure -- exactly the failure mode of the earlier global
# prominence+merge attempt.
const LATLOW = 20.0   # deg, apply the rules below this latitude only
const MINSEP = 300e3  # m, rule A
const WINMIN = 500e3  # m, rule B

function refine_lowlat(xc, y, idxs, types; minsep = MINSEP, winmin = WINMIN)
    idxs = copy(idxs); types = copy(types)
    while length(idxs) > 2
        gaps = [xc[idxs[k+1]] - xc[idxs[k]] for k in 1:length(idxs)-1]
        gmin, kmin = findmin(gaps)
        gmin >= minsep && break
        if kmin + 1 == length(idxs)
            # the too-close pair sits at the TAIL: the trailing extremum is a
            # boundary-ish bump, while its partner is usually the real deep
            # trough, so drop only the trailing one. (11.27 at lat=0:
            # max@100, min@1397 (KEt=381), max@1617 (KEt=387) -- deleting the
            # pair threw away the real min and left a single extremum => NaN.)
            pop!(idxs); pop!(types)
        else
            # interior wiggle: dropping BOTH re-joins the monotonic run around it
            deleteat!(idxs, kmin:kmin+1); deleteat!(types, kmin:kmin+1)
        end
    end
    # rule B: snap each min to the deepest INTERIOR LOCAL minimum within
    # +/-winmin. Restricted to genuine turning points (y[j-1] >= y[j] <= y[j+1])
    # rather than simply the lowest sample in the window: at 11.27 lat=0 the
    # trough is flat (KEt ~ 381,384,387,381 at x=1400..1700) and then falls to
    # 364 at x=1800 as the sponge bleeds into the window, so "lowest sample"
    # slid the min onto that boundary artifact and inflated Lbeat to 3400 km
    # (vs 2472 km for the 4 km counterpart 10.27). A monotone edge has no
    # local minimum, so this cannot happen. If the window holds no local
    # minimum at all, the min simply stays where the derivative test put it.
    for k in eachindex(idxs)
        types[k] == :min || continue
        x0 = xc[idxs[k]]
        w  = findall((x0-winmin) .<= xc .<= (x0+winmin))
        cand = [j for j in w if 1 < j < length(y) && y[j] <= y[j-1] && y[j] <= y[j+1]]
        isempty(cand) && continue
        idxs[k] = cand[argmin(y[cand])]
    end
    return idxs, types
end

colorsL = cgrad(:darktest, length(LATS), categorical = true)
figP = Figure(size=(1600,1150))

BeatDistP = zeros(length(runnms))   # 2*median(gaps) [km]
NextrP    = zeros(Int, length(runnms))

for i = 1:length(runnms)
    runnm = runnms[i]; LAT = LATS[i]
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    println(fnames,"; lat=",LAT," -------------------")

    @load string(dirout,"energetics_",fnames,".jld2") xc KEt
    dx = xc[2]-xc[1]

    Ivalid, KEtf = lowpass_mirror_restricted(xc, KEt, xleft, xright, Tcut, dx, Nord)
    xcv = xc[Ivalid]

    idxs, types = find_extrema_by_slope(KEtf)
    idxs_raw = copy(idxs)                       # kept for the diagnostic overlay
    if LAT < LATLOW
        idxs, types = refine_lowlat(xcv, KEtf, idxs, types)
    end
    NextrP[i] = length(idxs)

    xextr_km = xcv[idxs] ./ 1e3
    gaps_km  = diff(xextr_km)            # each gap is a HALF beat wavelength
    BeatDistP[i] = length(gaps_km) >= 1 ? 2*median(gaps_km) : NaN

    if length(gaps_km) < 1
        @warn "only $(length(idxs)) extremum found for $fnames (lat=$LAT) -- beat distance undefined"
    end

    @printf("  n=%d  gaps=[%s]  BeatDist=%.0f km\n", length(idxs),
        join([@sprintf("%.0f",g) for g in gaps_km], ","), BeatDistP[i])

    if savefl == 1
        fnameout = string("beatdist_",fnames,".jld2")
        jldsave(string(dirout,fnameout);
            LAT, xextr_km, extr_types = string.(types), gaps_km,
            BeatDist_km = BeatDistP[i]);
        println("  ",fnameout," data saved ........ ")
    end

    # diagnostic panel: raw (light gray) + filtered (black) + extrema.
    # gray x marks any extremum the low-latitude rules discarded.
    row = div(i-1,4)+1; col = mod(i-1,4)+1
    ax = Axis(figP[row,col], title=string("lat=",LAT), xlabel="x [km]", ylabel="KEt")
    lines!(ax, xc/1e3, KEt, color=:lightgray, linewidth=1)
    lines!(ax, xcv/1e3, KEtf, color=:black, linewidth=1.2)
    idrop = setdiff(idxs_raw, idxs)
    if !isempty(idrop)
        scatter!(ax, xcv[idrop]/1e3, KEtf[idrop], color=:gray, marker=:xcross, markersize=12)
    end
    for (idx,typ) in zip(idxs,types)
        colm = typ==:max ? :blue : :orange
        mk   = typ==:max ? :circle : :utriangle
        scatter!(ax, [xcv[idx]/1e3], [KEtf[idx]], color=colm, marker=mk,
            markersize=10, strokecolor=:black, strokewidth=1)
    end
    xlims!(ax, 0, 2000)
end

display(figP)
if figflag==1; save(string(dirfig,"KEt_beatextrema_",fnum,".png"), figP); end

# summary: measured beat distance vs latitude
figB = Figure(size=(700,450))
axB  = Axis(figB[1,1], title=string("measured KEt beat distance vs latitude (",fnum,")"),
    xlabel="latitude [°]", ylabel="beat distance [km]")
lines!(axB, LATS, BeatDistP, color=:black, linewidth=2)
scatter!(axB, LATS, BeatDistP, color=:black)
display(figB)
if figflag==1; save(string(dirfig,"KEt_beatdist_",fnum,".png"), figB); end

println("\nsummary:")
for i in 1:length(runnms)
    @printf("lat=%5.1f  nextrema=%2d  BeatDist=%.0f km\n", LATS[i], NextrP[i], BeatDistP[i])
end
