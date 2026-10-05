#= IW_Pomega_segments_batch13_notide.jl
Maarten Buijsman, USM DMS, 2026-9-5

Segment-averaged P(ω) (3 x-region lines: 100-600, 600-1100, 1100-1600 km,
same method as IW_komega_spectrum_notide.jl / IW_komega_spectrum.jl) for
ALL 13 runs in the mainnm=13, runnm=1:13 block (params_13_noforce.jl:
200m, GM-spectrum-initialized, Flux=0, no tidal forcing -- LAT13 =
[0,2.5,5,10,15,20,25,28.8,30,35,40,45,50]), one figure per run.

Deliberately SKIPS the full 2D k-omega heatmap (expensive -- would mean
13 full-resolution 2D FFTs) since only the segment-averaged P(ω) was
requested here; reuses the cheap per-x 1D FFT approach from
IW_Pomega_seg3_compare.jl, generalized to compute+plot all 3 segments
instead of just the last one.

Tukey(0.5) taper (not boxcar): no tidal periodicity to exploit here.
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
const segments_km = [(100.0,600.0), (600.0,1100.0), (1100.0,1600.0)]
const seg_colors = [:seagreen, :black, :dodgerblue]
const seg_labels = ["100-600 km", "600-1100 km", "1100-1600 km"]

function process_run(mainnm, runnm)
    row = get_runs(mainnm, [runnm])[1]
    LAT = row.lat
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    ds = NCDataset(string(dirsim, fnames, ".nc"), "r")
    xf = ds["x_faa"][:]
    tday = ds["time"][:] ./ (24*3600)
    It = findall(tday .>= tday[end]-tlast_days)
    xall_km = xf ./ 1e3
    dx_km = (xf[2]-xf[1])/1e3
    Nz = size(ds["u"], 2)
    t_days = tday[It]
    fcor_cpd = coriolis(LAT)/(2π)*86400

    seg_freq = Float64[]
    seg_PKE = Vector{Vector{Float64}}(undef, length(segments_km))
    for (si, (lo,hi)) in enumerate(segments_km)
        idxseg_all = findall(lo .<= xall_km .<= hi)
        stride = max(1, round(Int, seg_dx_km/dx_km))
        idxseg = idxseg_all[1:stride:end]
        Pu_acc = Float64[]; Pv_acc = Float64[]
        for ix in idxseg
            u_x = ds["u"][ix, Nz, It]
            v_x = ds["v"][ix, Nz, It]
            _, f1d, Pu_x = fft_spectra(t_days, u_x; tukeycf=0.5, numwin=1, linfit=true)
            _,   _, Pv_x = fft_spectra(t_days, v_x; tukeycf=0.5, numwin=1, linfit=true)
            if isempty(Pu_acc)
                Pu_acc = zeros(length(Pu_x)); Pv_acc = zeros(length(Pv_x))
                seg_freq = f1d
            end
            Pu_acc .+= Pu_x; Pv_acc .+= Pv_x
        end
        seg_PKE[si] = (Pu_acc .+ Pv_acc) ./ length(idxseg)
        println(fnames, " (lat=", LAT, "): segment ", seg_labels[si], ": averaged over ", length(idxseg), " x-points")
    end
    close(ds)

    xlo, xhi = 0.1, 48.0
    ipos = findall((seg_freq .>= xlo) .& (seg_freq .<= xhi))
    f_anchor = 1.0
    i_anchor = argmin(abs.(seg_freq[ipos] .- f_anchor))
    gm_ref = seg_PKE[2][ipos][i_anchor] .* (seg_freq[ipos] ./ seg_freq[ipos][i_anchor]).^(-2)

    fig = Figure(size=(700,450), fontsize=10)
    ax = Axis(fig[1,1], title=string("P(ω), segment-averaged — ",fnames," (lat=",LAT,", GM only, no tide)"),
        xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)
    for (si, lbl) in enumerate(seg_labels)
        lines!(ax, seg_freq[ipos], seg_PKE[si][ipos], color=seg_colors[si], label=lbl)
    end
    lines!(ax, seg_freq[ipos], gm_ref, color=:red, linestyle=:dash, label="ω⁻² (GM continuum) reference")
    if fcor_cpd > 0
        vlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
    end
    xlims!(ax, xlo, xhi)
    ylims!(ax, 1e-10, 1e0)
    xtc = [1,2,4,8,10,20,30,40]
    ax.xticks = (xtc, string.(xtc))
    axislegend(ax, position=:lb, labelsize=9)
    display(fig)
    save(string(dirfig,"Pomega_segments_",fnames,".png"), fig)
    println("saved Pomega_segments_", fnames, ".png")
end

for runnm in 1:13
    process_run(13, runnm)
end
