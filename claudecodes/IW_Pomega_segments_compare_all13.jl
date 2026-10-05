#= IW_Pomega_segments_compare_all13.jl
Maarten Buijsman, USM DMS, 2026-10-2

Generalizes IW_Pomega_seg12_compare_all13.jl / IW_Pomega_seg3_compare_all13.jl
into one configurable script: ANY set of x-segments (not fixed to 1/2/3),
ANY subsampling step (given directly in GRID CELLS, not a km target
converted to a stride), and ALL of fft_spectra()'s taper/detrend options
exposed as plain settings below -- edit SEGMENTS_KM, CELL_STEP, TUKEYCF,
LINFIT and rerun.

Same 3-way comparison as before (GM-only/no-tide vs no-GM+tide vs GM+tide,
same lat via mainnm=GMSER runnm=i / mainnm=11 runnm=26+i / mainnm=GMSER
runnm=26+i), same "matched window" control (n whole M2 periods, same n for
every run regardless of whether it actually has M2 forcing).

GMSER selects the GM series (13 = k-clamp IC, 15 = redistribution IC) and
LAT_IDX which latitudes to run; both are in the settings block, and the output
filenames carry the series so the two do not overwrite each other. Every run's
per-segment spectrum is cached to diagout/ as it is computed, so a long batch
survives interruption and later figures (e.g. the 2x2 paper panel) re-plot for
free instead of re-running the FFTs.

Still does ONE bulk read per file per latitude (covering the union of all
requested segments), then loops over cell-strided columns in memory --
the per-point-netCDF-read version was ~2.5h/13-lats/1-segment, dominated
by storage I/O latency, not compute.
=#

using NCDatasets, Printf, CairoMakie, Statistics, DSP, FFTW, JLD2

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0 = "/home/mbui/ModelOutput/"
dirsim = string(pth0,"IW/")
dirfig = string(pth0,"figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))

# ---- USER SETTINGS -----------------------------------------------------
# any number of (index, lo_km, hi_km) segments -- index is just used for
# the output filename tag, doesn't need to be 1/2/3
const SEGMENTS_KM = [(1,100.0,600.0), (2,600.0,1100.0), (3,1100.0,1600.0)]
const CELL_STEP   = 20     # subsample every CELL_STEP-th native grid cell
                            # (dx=200m -> CELL_STEP=20 means every 4km, same
                            # as the earlier seg_dx_km=4.0 default)
const TUKEYCF     = 0.0    # fft_spectra() Tukey taper fraction (0=boxcar, 1=Hann)
const LINFIT      = true   # false: just remove the mean; true: remove mean+linear trend
const TLAST_DAYS  = 10.0

# which GM series to compare against the no-GM reference:
#   13 = k-clamp GM initial condition (the original run of this script)
#   15 = redistribution GM initial condition (~3x GM81 at t = 0)
#   16 = GM81 initial condition, 1x GM81 over days 10-20 (what the paper figures use)
# Both blocks are laid out the same way: runnm 26+i is GM + 25 kW/m tide and
# runnm i is GM only with Flux=0, for LAT13[i].
const GMSER   = 16
# which LAT13 indices to process; 1:12 is lat 0-45, i.e. runnm 27-38, dropping
# 50°N as the paper figures do
# Overridable from the command line -- `julia this.jl 4 6` runs LAT13[4:6] --
# so a long batch can be taken in slices. Reading a 10-day surface time series
# touches most of an 82 GB file, so the kernel page cache grows to tens of GB
# per run regardless of how little is kept in Julia arrays; running in slices
# lets each process exit and that pressure drop between chunks.
const LAT_IDX = length(ARGS) == 2 ? (parse(Int,ARGS[1]):parse(Int,ARGS[2])) : 1:12
# the FFTs are the expensive part (one bulk read + ~375 spectra per file), so
# every run's per-segment spectrum is cached. The cache key carries everything
# that changes the numbers; a settings change invalidates it.
const CACHE = string(pth0, "diagout/",
    @sprintf("Pomega_segcache_GM%d_tukey%g_step%d_lin%d.jld2",
             GMSER, TUKEYCF, CELL_STEP, LINFIT))
# --------------------------------------------------------------------------

const T2 = 12 + 25.2/60
const f1_cpd = 24/T2   # M2 frequency in cpd
const T2_days = T2/24
const n_periods = floor(Int, TLAST_DAYS/T2_days)
const tdur_days = n_periods*T2_days
const xleft_km = minimum(s[2] for s in SEGMENTS_KM)
const xright_km = maximum(s[3] for s in SEGMENTS_KM)
println("using ", n_periods, " whole M2 periods = ", tdur_days, " days; CELL_STEP=", CELL_STEP,
    "; tukeycf=", TUKEYCF, "; linfit=", LINFIT, "; x window ", xleft_km, "-", xright_km, "km")

# returns, for ONE file, the segment-averaged (freq, PKE) for EACH requested
# segment, via a single bulk read of the union x-window
function seg_pomega_all(mainnm, runnm)
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    ds = NCDataset(string(dirsim, fnames, ".nc"), "r")
    xf = ds["x_faa"][:]
    tday = ds["time"][:] ./ (24*3600)
    Ix = findall(xleft_km*1e3 .<= xf .<= xright_km*1e3)
    It = findall(tday .>= tday[end]-tdur_days)
    isodd(length(Ix)) || (Ix = Ix[1:end-1])
    xseg_km = xf[Ix] ./ 1e3
    Nz = size(ds["u"], 2)
    t_days = tday[It]

    # Read ONLY the strided columns that are actually used, one segment at a
    # time, as a StepRange. The previous version bulk-read the whole
    # 100-1600 km union (7501 x-points x 2831 times x 2 variables) and then
    # used every 20th column -- 95% of what it read and transposed was thrown
    # away, and the batch was killed by the OOM reaper partway through the
    # second latitude. Same points, same FFTs, ~1/20 the memory and I/O.
    Itr = It[1]:It[end]                 # contiguous, cheaper than an index vector
    results = Dict{Int, Tuple{Vector{Float64},Vector{Float64}}}()
    for (si, lo, hi) in SEGMENTS_KM
        idxseg_all = findall(lo .<= xseg_km .<= hi)
        nseg = length(idxseg_all[1:CELL_STEP:end])
        g0   = Ix[idxseg_all[1]]        # Ix is contiguous, so the strided pick
        grng = g0:CELL_STEP:(g0 + CELL_STEP*(nseg-1))   # is an arithmetic range

        u_seg = ds["u"][grng, Nz, Itr]  # (nseg, Nt)
        v_seg = ds["v"][grng, Nz, Itr]

        Pu_acc = Float64[]; Pv_acc = Float64[]; freq = Float64[]
        for il in 1:nseg
            _, f1d, Pu_x = fft_spectra(t_days, u_seg[il,:]; tukeycf=TUKEYCF, numwin=1, linfit=LINFIT)
            _,   _, Pv_x = fft_spectra(t_days, v_seg[il,:]; tukeycf=TUKEYCF, numwin=1, linfit=LINFIT)
            if isempty(Pu_acc)
                Pu_acc = zeros(length(Pu_x)); Pv_acc = zeros(length(Pv_x))
                freq = f1d
            end
            Pu_acc .+= Pu_x; Pv_acc .+= Pv_x
        end
        u_seg = nothing; v_seg = nothing
        results[si] = (freq, (Pu_acc .+ Pv_acc) ./ nseg)
        println(fnames, ": segment ", lo, "-", hi, "km, averaged over ", nseg, " x-points (step=", CELL_STEP, " cells)")
    end
    close(ds)
    GC.gc()
    return results
end

# cache keyed by (mainnm, runnm); a run already computed is never re-read
CACHED = Dict{Tuple{Int,Int}, Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}}()
isfile(CACHE) && merge!(CACHED, load(CACHE, "BYRUN"))
println("cache ", CACHE, ": ", length(CACHED), " runs already stored")

function cached_pomega(mainnm, runnm)
    key = (mainnm, runnm)
    if !haskey(CACHED, key)
        CACHED[key] = seg_pomega_all(mainnm, runnm)
        jldsave(CACHE; BYRUN=CACHED)     # save as we go: a long batch can be
    end                                   # interrupted without losing the work
    return CACHED[key]
end

function lat_runs(i)
    [(11, 26+i, "11.$(26+i) (no GM)"),
     (GMSER, 26+i, "$GMSER.$(26+i) (GM+tide)"),
     (GMSER, i,    "$GMSER.$i (GM, no tide)")]
end

function process_lat(i)
    row = get_runs(GMSER, [i])[1]
    LAT = row.lat
    runs = lat_runs(i)
    colors = [:black, :red, :dodgerblue]

    res = [cached_pomega(mainnm,runnm) for (mainnm,runnm,lbl) in runs]

    for (si, lo, hi) in SEGMENTS_KM
        cm_to_pt = 72/2.54
        fig = Figure(size=(15*cm_to_pt, 10*cm_to_pt), fontsize=10)
        ax = Axis(fig[1,1], title=string("P(ω), segment ",lo,"-",hi," km, tukey=",TUKEYCF,", linfit=",LINFIT," -- lat=",LAT),
            xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)
        xlo, xhi = 0.3, 48.0
        for (ri,(mainnm,runnm,lbl)) in enumerate(runs)
            freq, PKE = res[ri][si]
            ipos = findall((freq .>= xlo) .& (freq .<= xhi))
            lines!(ax, freq[ipos], PKE[ipos], color=colors[ri], label=lbl)
        end

        # ω⁻² (GM continuum) reference, anchored on the GM-only (13.i, blue)
        # curve at 1 cpd; ω⁻³ (harmonic envelope) reference, anchored on the
        # no-GM (11.x, black) curve's M2 peak -- same anchoring convention as
        # the earlier single-run scripts
        freq_gm, PKE_gm = res[3][si]
        ipos_gm = findall((freq_gm .>= xlo) .& (freq_gm .<= xhi))
        f_anchor = 1.0
        i_anchor = argmin(abs.(freq_gm[ipos_gm] .- f_anchor))
        gm_ref = PKE_gm[ipos_gm][i_anchor] .* (freq_gm[ipos_gm] ./ freq_gm[ipos_gm][i_anchor]).^(-2)
        lines!(ax, freq_gm[ipos_gm], gm_ref, color=:orange, linestyle=:dash, linewidth=1.5, label="ω⁻² reference")

        freq_nogm, PKE_nogm = res[1][si]
        ipos_nogm = findall((freq_nogm .>= xlo) .& (freq_nogm .<= xhi))
        i_m2 = argmin(abs.(freq_nogm[ipos_nogm] .- f1_cpd))
        harm_ref = PKE_nogm[ipos_nogm][i_m2] .* (freq_nogm[ipos_nogm] ./ freq_nogm[ipos_nogm][i_m2]).^(-3)
        lines!(ax, freq_nogm[ipos_nogm], harm_ref, color=:purple, linestyle=:dashdot, linewidth=1.5, label="ω⁻³ reference")

        fcor_cpd = coriolis(LAT)/(2π)*86400
        if fcor_cpd > 0
            vlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
        end
        xlims!(ax, xlo, xhi)
        ylims!(ax, 1e-10, 1e0)
        xtc = [0.5,1,2,4,8,12,24,36,48]
        xtc_labels = [v == 0.5 ? "0.5" : string(Int(v)) for v in xtc]
        ax.xticks = (xtc, xtc_labels)
        axislegend(ax, position=:lb, labelsize=9, framevisible=false)
        display(fig)
        fname_out = string("Pomega_seg",si,"_compare_GM",GMSER,"_tukey",TUKEYCF,"_lat", LAT, ".png")
        savefig300(string(dirfig,fname_out), fig)
        println("saved ", fname_out)
    end
end

for i in LAT_IDX
    @printf("\n===== LAT13[%d] = %.1f°N  (%d/%d) =====\n", i, LAT13[i],
            findfirst(==(i), collect(LAT_IDX)), length(LAT_IDX))
    @time process_lat(i)
end
