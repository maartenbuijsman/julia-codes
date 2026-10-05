#= IW_Pomega_seg3_compare.jl
Maarten Buijsman, USM DMS, 2026-9-5

Compares the LAST x-segment (1100-1600 km) of the segment-averaged P(ω)
across three runs in one plot: 11.37 (200m, D2-forced, no GM), <GM>.37 (200m,
D2-forced, GM-initialized), <GM>.11 (200m, GM-only, Flux=0, no tide) -- all
lat=40. The GM series is set by GMSER below: 13 = k-clamp IC (the original
run of this script), 15 = redistribution IC (what the paper figures use).
The output filename carries the series, so the two do not overwrite. Only loads/computes the 1100-1600km segment (not the full domain
or the 2D k-omega heatmap), so this is cheap regardless of how expensive
the full k-omega analysis is elsewhere.

Same method as the segment-averaged P(ω) in IW_komega_spectrum.jl /
IW_komega_spectrum_notide.jl: fft_spectra() on u,v at every 4km-subsampled
x-point in the segment, POWER (not complex coefficient) averaged across
x.

REVISED 2026-9-5: for a genuinely controlled 3-way comparison, ALL THREE
runs now use the exact same window duration (n whole M2 periods, same n
for every run -- same frequency resolution/bin count) and the exact same
taper (NO taper / boxcar, tukeycf=0.0) -- not each script's own internal
convention (which differed: the D2-forced scripts snap to whole M2
periods AND skip the taper since a periodic tidal signal wraps cleanly;
the no-tide script uses a real Tukey taper since it has no periodicity to
exploit). Using boxcar+same-duration everywhere means 13.11 (no tidal
periodicity) DOES have a genuine, un-tapered edge mismatch -- accepted
deliberately here so every run goes through the literal same pipeline,
per explicit request.
=#

using NCDatasets, Printf, CairoMakie, Statistics, DSP, FFTW

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0 = "/home/mbui/ModelOutput/"
dirsim = string(pth0,"IW/")
dirfig = string(pth0,"figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))

const seg_lo_km, seg_hi_km = 1100.0, 1600.0
const seg_dx_km = 4.0
const tlast_days = 10.0

# snap the window to the largest whole number of M2 periods that fits --
# SAME n (hence same duration, same df) used for all three runs below,
# regardless of whether that run actually has M2 forcing
const T2 = 12 + 25.2/60          # M2 period [hours]
const T2_days = T2/24
const n_periods = floor(Int, tlast_days/T2_days)
const tdur_days = n_periods*T2_days
println("using ", n_periods, " whole M2 periods = ", tdur_days, " days (out of ", tlast_days, " requested) -- no taper, same window for all 3 runs")

# which GM series to compare against the no-GM reference:
#   13 = k-clamp GM initial condition (the original run of this script)
#   15 = redistribution GM initial condition (the series the paper figures use)
# Both have the same layout -- runnm 27:39 is GM + 25 kW/m tide, runnm 1:13 is
# GM only with Flux=0 -- and runnm 37 / 11 are both lat = 40°N in those blocks.
const GMSER = 15

runs = [(11, 37, "11.37 (no GM)"),
        (GMSER, 37, @sprintf("%d.37 (GM+tide)", GMSER)),
        (GMSER, 11, @sprintf("%d.11 (GM, no tide)", GMSER))]
colors = [:black, :red, :dodgerblue]

function seg3_pomega(mainnm, runnm)
    row = get_runs(mainnm, [runnm])[1]
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    ds = NCDataset(string(dirsim, fnames, ".nc"), "r")
    xf = ds["x_faa"][:]
    tday = ds["time"][:] ./ (24*3600)
    Ix = findall(seg_lo_km*1e3 .<= xf .<= seg_hi_km*1e3)
    It = findall(tday .>= tday[end]-tdur_days)
    dx_km = (xf[2]-xf[1])/1e3
    stride = max(1, round(Int, seg_dx_km/dx_km))
    idxseg = Ix[1:stride:end]
    Nz = size(ds["u"], 2)
    t_days = tday[It]

    # BULK-read the whole segment once, then stride over columns in memory.
    # The original version read ds["u"][ix, Nz, It] one x-point at a time --
    # ~126 separate strided netCDF reads per variable per run, which on this
    # storage is dominated by per-read latency (the same fix
    # IW_Pomega_seg12_compare_all13.jl documents). Identical points, identical
    # FFTs, just not one syscall each.
    # NOTE u is on x_faa and v on x_caa, so a shared index puts the v point
    # half a cell (100 m) from the u point. Kept exactly as the original did
    # it, so this result stays directly comparable with the mainnm 13 figure.
    u_seg = ds["u"][Ix[1]:Ix[end], Nz, It]
    v_seg = ds["v"][Ix[1]:Ix[end], Nz, It]
    close(ds)

    Pu_acc = Float64[]; Pv_acc = Float64[]; freq = Float64[]
    for ix in idxseg
        il = ix - Ix[1] + 1
        _, f1d, Pu_x = fft_spectra(t_days, u_seg[il,:]; tukeycf=0.0, numwin=1, linfit=true)
        _,   _, Pv_x = fft_spectra(t_days, v_seg[il,:]; tukeycf=0.0, numwin=1, linfit=true)
        if isempty(Pu_acc)
            Pu_acc = zeros(length(Pu_x)); Pv_acc = zeros(length(Pv_x))
            freq = f1d
        end
        Pu_acc .+= Pu_x; Pv_acc .+= Pv_x
    end
    n = length(idxseg)
    println(fnames, ": segment ", seg_lo_km, "-", seg_hi_km, "km, averaged over ", n, " x-points")
    return freq, (Pu_acc .+ Pv_acc) ./ n
end

cm_to_pt = 72/2.54
fig = Figure(size=(15*cm_to_pt, 10*cm_to_pt), fontsize=10)
ax = Axis(fig[1,1], titlesize=9,
    title=@sprintf("P(ω), %g-%g km, lat 40°N, no taper, %d·T(M2) window",
                   seg_lo_km, seg_hi_km, n_periods),
    xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)

xlo, xhi = 0.1, 48.0
for ((mainnm,runnm,lbl), c) in zip(runs, colors)
    freq, PKE = seg3_pomega(mainnm, runnm)
    ipos = findall((freq .>= xlo) .& (freq .<= xhi))
    lines!(ax, freq[ipos], PKE[ipos], color=c, label=lbl)
end

fcor_cpd = coriolis(40.0)/(2π)*86400
vlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
xlims!(ax, xlo, xhi)
ylims!(ax, 1e-10, 1e0)
xticks_candidates = [1,2,4,8,10,20,30,40]
ax.xticks = (xticks_candidates, string.(xticks_candidates))
axislegend(ax, position=:lb, labelsize=9)
display(fig)
fout = @sprintf("Pomega_seg3_compare_notaper_11.37_%d.37_%d.11.png", GMSER, GMSER)
savefig300(string(dirfig, fout), fig)
println("saved ", fout)
