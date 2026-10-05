#= IW_Pomega_seg3_compare_all13.jl
Maarten Buijsman, USM DMS, 2026-9-5

Generalizes IW_Pomega_seg3_compare.jl (which did lat=40 only) to all 13
LAT13 latitudes. For each of the 13 latitude indices i=1:13, compares the
LAST x-segment (1100-1600 km) segment-averaged P(ω) across three runs
sharing that latitude:
  - GM only, no tide:    mainnm=13, runnm=i        (Flux=0, params_13_noforce.jl)
  - no GM, D2 tide:      mainnm=11, runnm=26+i      (Flux=25kW/m)
  - GM + D2 tide:        mainnm=13, runnm=26+i      (Flux=25kW/m)
(runnm-to-lat mapping verified directly: all three give the same LAT13[i]
for every i=1:13.)

Same controlled-comparison method as IW_Pomega_seg3_compare.jl: ALL THREE
runs use the identical window duration (n whole M2 periods, same n for
every run) and NO taper (boxcar) -- deliberately applied even to the
no-tide run for a literal same-pipeline comparison, per explicit request.
fft_spectra() on u,v at every 4km-subsampled x-point in the segment,
POWER (not complex coefficient) averaged across x. One figure per
latitude (13 total).
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

const T2 = 12 + 25.2/60
const T2_days = T2/24
const n_periods = floor(Int, tlast_days/T2_days)
const tdur_days = n_periods*T2_days
println("using ", n_periods, " whole M2 periods = ", tdur_days, " days -- no taper, same window for all runs")

function seg3_pomega(mainnm, runnm)
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

    Pu_acc = Float64[]; Pv_acc = Float64[]; freq = Float64[]
    for ix in idxseg
        u_x = ds["u"][ix, Nz, It]
        v_x = ds["v"][ix, Nz, It]
        _, f1d, Pu_x = fft_spectra(t_days, u_x; tukeycf=0.0, numwin=1, linfit=true)
        _,   _, Pv_x = fft_spectra(t_days, v_x; tukeycf=0.0, numwin=1, linfit=true)
        if isempty(Pu_acc)
            Pu_acc = zeros(length(Pu_x)); Pv_acc = zeros(length(Pv_x))
            freq = f1d
        end
        Pu_acc .+= Pu_x; Pv_acc .+= Pv_x
    end
    close(ds)
    n = length(idxseg)
    println(fnames, ": segment ", seg_lo_km, "-", seg_hi_km, "km, averaged over ", n, " x-points")
    return freq, (Pu_acc .+ Pv_acc) ./ n
end

function process_lat(i)
    row = get_runs(13, [i])[1]
    LAT = row.lat
    runs = [(11,26+i,"11.$(26+i) (no GM)"), (13,26+i,"13.$(26+i) (GM+tide)"), (13,i,"13.$i (GM, no tide)")]
    colors = [:black, :red, :dodgerblue]

    cm_to_pt = 72/2.54
    fig = Figure(size=(15*cm_to_pt, 10*cm_to_pt), fontsize=10)
    ax = Axis(fig[1,1], title=string("P(ω), segment 1100-1600 km, no taper, n·T(M2) window -- lat=",LAT),
        xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)

    xlo, xhi = 0.1, 48.0
    for ((mainnm,runnm,lbl), c) in zip(runs, colors)
        freq, PKE = seg3_pomega(mainnm, runnm)
        ipos = findall((freq .>= xlo) .& (freq .<= xhi))
        lines!(ax, freq[ipos], PKE[ipos], color=c, label=lbl)
    end

    fcor_cpd = coriolis(LAT)/(2π)*86400
    if fcor_cpd > 0
        vlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
    end
    xlims!(ax, xlo, xhi)
    ylims!(ax, 1e-10, 1e0)
    xticks_candidates = [1,2,4,8,10,20,30,40]
    ax.xticks = (xticks_candidates, string.(xticks_candidates))
    axislegend(ax, position=:lb, labelsize=9)
    display(fig)
    fname_out = string("Pomega_seg3_compare_notaper_lat", LAT, ".png")
    savefig300(string(dirfig,fname_out), fig)
    println("saved ", fname_out)
end

for i in 1:13
    process_lat(i)
end
