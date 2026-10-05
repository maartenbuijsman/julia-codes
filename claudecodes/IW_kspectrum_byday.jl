#= IW_kspectrum_byday.jl
Maarten Buijsman, USM DMS, 2026-9-5

Wavenumber (k) spectrum of surface velocity (KE=Pu+Pv), one spectrum per
day of the run, all overlaid in a single figure -- to see how the spatial
spectrum evolves over time (e.g. cascade to smaller scales / dissipation
in the mainnm=13 runnm=1:13 GM-only, Flux=0 free-decay block).

Uses fft_spectra() (functions/fft_spectra_vectorized.jl, plain 1D FFT) on
the FULL-resolution x series (no lowpass/decimation) at the snapshot
nearest the middle of each day -- learned the hard way in the P(omega)
check just before this that decimating x before the FFT is NOT
frequency/wavenumber-neutral (kills ~50% of true power by k corresponding
to 4 cpd-equivalent scales), so for a real k-spectrum comparison across
days, decimation must be avoided entirely, not just minimized.

Tukey(0.5) taper, numwin=5 (5 overlapping 50%-overlap segments, averaged)
-- trades wavenumber resolution for a less noisy/more converged per-day
spectrum than the numwin=1/2 versions.
=#

using NCDatasets, Printf, CairoMakie, Statistics, DSP, FFTW

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0 = "/home/mbui/ModelOutput/"
dirsim = string(pth0,"IW/")
dirfig = string(pth0,"figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))

mainnm, runnm = 13, 11
row = get_runs(mainnm, [runnm])[1]
LAT = row.lat
fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
filename = string(dirsim, fnames, ".nc")

const xleft_km, xright_km = 200.0, 1800.0

ds = NCDataset(filename,"r")
xf   = ds["x_faa"][:]
tday = ds["time"][:] ./ (24*3600)
Ix = findall(xleft_km*1e3 .<= xf .<= xright_km*1e3)
isodd(length(Ix)) || (Ix = Ix[1:end-1])
xc_km = xf[Ix] ./ 1e3
Nz = size(ds["u"], 2)

days = 1:floor(Int, tday[end])-1     # skip day 0 (initial condition) and the partial last day
println("computing k-spectrum at ", length(days), " daily snapshots (t=", days[1], " to ", days[end], " days), Nx=", length(Ix))

specs = Vector{Vector{Float64}}(undef, length(days))
k_cpkm = Float64[]
for (i,d) in enumerate(days)
    it = argmin(abs.(tday .- (d+0.5)))
    u_x = ds["u"][Ix, Nz, it]
    v_x = ds["v"][Ix, Nz, it]
    _, kk, Pu = fft_spectra(xc_km, u_x; tukeycf=0.5, numwin=5, linfit=true)
    _,  _, Pv = fft_spectra(xc_km, v_x; tukeycf=0.5, numwin=5, linfit=true)
    global k_cpkm = kk
    specs[i] = Pu .+ Pv
    println("day ", d, " (t=", round(tday[it],digits=2), " days): done")
end
close(ds)

cm_to_pt = 72/2.54
fig = Figure(size=(15*cm_to_pt, 11*cm_to_pt), fontsize=10)
ax = Axis(fig[1,1], title=string("k-spectrum (KE=Pu+Pv), one per day — ",fnames," (lat=",LAT,", GM only, no tide)"),
    xlabel="wavenumber [cycles/km] (1/wavelength)", ylabel="power [m²/s²·km]", xscale=log10, yscale=log10)

cmap = cgrad(:viridis, length(days), categorical=true)
for (i,d) in enumerate(days)
    lines!(ax, k_cpkm, specs[i], color=cmap[i], linewidth=1.5)
end
Colorbar(fig[1,2], limits=(days[1], days[end]), colormap=cgrad(:viridis, length(days), categorical=true), label="day")
xlims!(ax, k_cpkm[1], k_cpkm[end])
display(fig)
savefig300(string(dirfig,"kspectrum_byday_",fnames,".png"), fig)
println("saved kspectrum_byday_", fnames, ".png")
