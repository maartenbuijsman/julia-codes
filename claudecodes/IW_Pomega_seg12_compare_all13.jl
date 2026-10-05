#= IW_Pomega_seg12_compare_all13.jl
Maarten Buijsman, USM DMS, 2026-9-6

Same 3-way, 13-latitude, no-taper/matched-n*T(M2)-window comparison as
IW_Pomega_seg3_compare_all13.jl, but for segments 1 (100-600 km) and 2
(600-1100 km) instead of segment 3 (1100-1600 km) -- both done together
in ONE pass per latitude.

PERFORMANCE FIX vs IW_Pomega_seg3_compare_all13.jl: that script read u,v
ONE X-POINT AT A TIME directly from the open NCDataset inside the FFT
loop (`ds["u"][ix, Nz, It]`, ~126 separate strided netCDF reads per
segment per file) -- this took ~2.5 hours for 13 latitudes x 3 files x 1
segment, almost certainly dominated by per-read I/O latency on this
storage. Fixed here the same way IW_komega_spectrum.jl already does it:
BULK-read the whole needed x-range ONCE per file into memory
(u_slice/v_slice), then loop over subsampled columns via fast in-memory
array indexing. Also opens each of the 3 files only ONCE per latitude
(covering x=100-1100km, i.e. both segments at once) instead of once per
segment.
=#

using NCDatasets, Printf, CairoMakie, Statistics, DSP, FFTW

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0 = "/home/mbui/ModelOutput/"
dirsim = string(pth0,"IW/")
dirfig = string(pth0,"figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))

const seg_dx_km = 4.0
const tlast_days = 10.0
const segments = [(1,100.0,600.0), (2,600.0,1100.0)]   # (index, lo_km, hi_km)
const xleft_km, xright_km = 100.0, 1100.0               # covers both segments in one bulk read

const T2 = 12 + 25.2/60
const T2_days = T2/24
const n_periods = floor(Int, tlast_days/T2_days)
const tdur_days = n_periods*T2_days
println("using ", n_periods, " whole M2 periods = ", tdur_days, " days -- no taper, same window for all runs")

# returns, for ONE file, the segment-averaged (freq, PKE) for EACH segment,
# via a single bulk read of the whole x=100-1100km window
function seg_pomega_all(mainnm, runnm)
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    ds = NCDataset(string(dirsim, fnames, ".nc"), "r")
    xf = ds["x_faa"][:]
    tday = ds["time"][:] ./ (24*3600)
    Ix = findall(xleft_km*1e3 .<= xf .<= xright_km*1e3)
    It = findall(tday .>= tday[end]-tdur_days)
    isodd(length(Ix)) || (Ix = Ix[1:end-1])
    xseg_km = xf[Ix] ./ 1e3
    dx_km = (xf[2]-xf[1])/1e3
    Nz = size(ds["u"], 2)
    t_days = tday[It]

    u_slice = permutedims(ds["u"][Ix, Nz, It], (2,1))   # (Nt, Nx)
    v_slice = permutedims(ds["v"][Ix, Nz, It], (2,1))
    close(ds)

    stride = max(1, round(Int, seg_dx_km/dx_km))
    results = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for (si, lo, hi) in segments
        idxseg_all = findall(lo .<= xseg_km .<= hi)
        idxseg = idxseg_all[1:stride:end]
        Pu_acc = Float64[]; Pv_acc = Float64[]; freq = Float64[]
        for ix in idxseg
            _, f1d, Pu_x = fft_spectra(t_days, u_slice[:,ix]; tukeycf=0.0, numwin=1, linfit=true)
            _,   _, Pv_x = fft_spectra(t_days, v_slice[:,ix]; tukeycf=0.0, numwin=1, linfit=true)
            if isempty(Pu_acc)
                Pu_acc = zeros(length(Pu_x)); Pv_acc = zeros(length(Pv_x))
                freq = f1d
            end
            Pu_acc .+= Pu_x; Pv_acc .+= Pv_x
        end
        results[si] = (freq, (Pu_acc .+ Pv_acc) ./ length(idxseg))
        println(fnames, ": segment ", lo, "-", hi, "km, averaged over ", length(idxseg), " x-points")
    end
    return results
end

function process_lat(i)
    row = get_runs(13, [i])[1]
    LAT = row.lat
    runs = [(11,26+i,"11.$(26+i) (no GM)"), (13,26+i,"13.$(26+i) (GM+tide)"), (13,i,"13.$i (GM, no tide)")]
    colors = [:black, :red, :dodgerblue]

    res = [seg_pomega_all(mainnm,runnm) for (mainnm,runnm,lbl) in runs]

    for (si, lo, hi) in segments
        cm_to_pt = 72/2.54
        fig = Figure(size=(15*cm_to_pt, 10*cm_to_pt), fontsize=10)
        ax = Axis(fig[1,1], title=string("P(ω), segment ",lo,"-",hi," km, no taper, n·T(M2) window -- lat=",LAT),
            xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)
        xlo, xhi = 0.1, 48.0
        for (ri,(mainnm,runnm,lbl)) in enumerate(runs)
            freq, PKE = res[ri][si]
            ipos = findall((freq .>= xlo) .& (freq .<= xhi))
            lines!(ax, freq[ipos], PKE[ipos], color=colors[ri], label=lbl)
        end
        fcor_cpd = coriolis(LAT)/(2π)*86400
        if fcor_cpd > 0
            vlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
        end
        xlims!(ax, xlo, xhi)
        ylims!(ax, 1e-10, 1e0)
        xtc = [1,2,4,8,10,20,30,40]
        ax.xticks = (xtc, string.(xtc))
        axislegend(ax, position=:lb, labelsize=9)
        display(fig)
        fname_out = string("Pomega_seg",si,"_compare_notaper_lat", LAT, ".png")
        savefig300(string(dirfig,fname_out), fig)
        println("saved ", fname_out)
    end
end

for i in 1:13
    process_lat(i)
end
