#= IW_Pomega_x1000km_check.jl
Maarten Buijsman, USM DMS, 2026-9-5

Diagnostic: is the steeper-than-omega^-2 slope above ~10 cpd in
komega_Pomega_AMZexpt13.11.png (from IW_komega_spectrum_notide.jl) a real
feature, or an artifact of the spatial anti-alias lowpass filter (cutoff
10km) + x-decimation to dx=5km used there (needed only so the k-omega
HEATMAP doesn't overrun CairoMakie)? That filter removes all k>0.1 cyc/km
content regardless of frequency -- if real high-frequency energy lives at
smaller scales than that, the earlier P(omega) tail would be
artificially steep.

This uses fft_spectra() (functions/fft_spectra_vectorized.jl, a plain 1D
FFT-in-time spectrum) directly on the surface u,v time series at a SINGLE
x location (x=1000km) -- no spatial filtering or decimation at all, so it
can't share that artifact -- as an independent check on the >10 cpd slope.
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

const xtarget_km = 1000.0
const tlast_days = 10.0

ds = NCDataset(filename,"r")
xc = ds["x_caa"][:]
tday = ds["time"][:] ./ (24*3600)
ix = argmin(abs.(xc .- xtarget_km*1e3))
It = findall(tday .>= tday[end]-tlast_days)

Nz = size(ds["u"], 2)
u_t = ds["u"][ix, Nz, It]
v_t = ds["v"][ix, Nz, It]
t_days = tday[It]
close(ds)

println(fnames, "; lat=", LAT, "; x=", xc[ix]/1e3, " km; t window ", t_days[1], "-", t_days[end], " days, n=", length(t_days))

# plain 1D FFT-in-time spectrum, no spatial filtering at all -- default
# Tukey(0.5) taper in time (needed since the window isn't periodic), no
# segment-averaging (numwin=1, we want the finest frequency resolution)
period_u, freq_cpd, Pu = fft_spectra(t_days, u_t; tukeycf=0.5, numwin=1, linfit=true)
_,        _,        Pv = fft_spectra(t_days, v_t; tukeycf=0.5, numwin=1, linfit=true)
PKE = Pu .+ Pv

fcor = coriolis(LAT)
fcor_cpd = fcor/(2π)*86400

xlo, xhi = 0.1, maximum(freq_cpd)
ipos = findall((freq_cpd .>= xlo) .& (freq_cpd .<= xhi))
f_anchor = 1.0
i_anchor = argmin(abs.(freq_cpd[ipos] .- f_anchor))
gm_ref = PKE[ipos][i_anchor] .* (freq_cpd[ipos] ./ freq_cpd[ipos][i_anchor]).^(-2)

fig = Figure(size=(700,450))
ax = Axis(fig[1,1], title=string("P(ω) at x=",xtarget_km," km (1D FFT-in-time, no spatial filter) — ",fnames," (lat=",LAT,")"),
    xlabel="frequency [cpd]", ylabel="power [m²/s²·day]", xscale=log10, yscale=log10)
lines!(ax, freq_cpd[ipos], PKE[ipos], color=:black, label="P(ω), KE=Pu+Pv, x=1000km")
lines!(ax, freq_cpd[ipos], gm_ref, color=:red, linestyle=:dash, label="ω⁻² (GM continuum) reference")
vlines!(ax, [fcor_cpd], color=:gray, linestyle=:dash, label="inertial frequency f")
xlims!(ax, xlo, xhi)
xtick_candidates = [1,2,4,8,10,20,30,40,60,80,100,120,140]
ax.xticks = (xtick_candidates[findall(xlo .<= xtick_candidates .<= xhi)], string.(xtick_candidates[findall(xlo .<= xtick_candidates .<= xhi)]))
axislegend(ax, position=:lb, labelsize=10)
display(fig)
save(string(dirfig,"Pomega_x1000km_",fnames,".png"), fig)
println("saved Pomega_x1000km_", fnames, ".png")
println("Nyquist freq = ", maximum(freq_cpd), " cpd")
