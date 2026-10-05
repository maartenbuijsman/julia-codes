#= IW_KE_zonalavg_13gm_timeseries.jl
Maarten Buijsman, USM DMS, 2026-9-6

Depth-integrated, zonal-mean (x=100-1800 km) KE of the GM-only (no tide,
Flux=0) 13.1-13 series, sampled ONCE PER DAY -- including t=0, the initial
condition -- read directly from the raw .nc output. This is NOT the
post-spinup 10-13-tidal-cycle time-mean used in energetics_AMZexpt13.*.jld2
(IW_KE_zonalavg_13gm_vs_lat.jl); the point here is to see how the initial KE
varies by latitude (GM initialization energy consistency check) and how it
decays over time.

All 13 runs (1-13, lat 0-50N), overlaid in one figure, colored by latitude.
=#

using NCDatasets, Printf, CairoMakie, Statistics

pth0   = "/home/mbui/ModelOutput/"
dirsim = string(pth0, "IW/")
dirfig = string(pth0, "figs/")

const rho0 = 1020.0
const xlo, xhi = 100e3, 1800e3
const fcKE = 1e-3   # J/m^2 -> kJ/m^2

mainnm = 13
runnms = collect(1:13)
LATS   = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]

function ke_timeseries(mainnm, runnm)
    fnames   = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    filename = string(dirsim, fnames, ".nc")
    ds = NCDataset(filename, "r")

    tday = ds["time"][:] ./ 86400
    xc   = ds["x_caa"][:]
    dz   = ds["Δz_aac"][:]
    Ix   = findall(xlo .<= xc .<= xhi)

    # once-per-day sample indices, including t=0
    ndays  = floor(Int, tday[end])
    Idaily = Int[]
    for d in 0:ndays
        _, idx = findmin(abs.(tday .- d))
        push!(Idaily, idx)
    end
    tsel = tday[Idaily]

    dzz = reshape(dz, 1, :)   # (1, Nz) for broadcasting over x
    KEavg_t = zeros(length(Idaily))

    for (k, it) in enumerate(Idaily)
        uf = ds["u"][:, :, it]              # (Nx+1, Nz)
        vc = ds["v"][:, :, it]              # (Nx,   Nz)
        wf = ds["w"][:, :, it]              # (Nx,   Nz+1)

        uc = (uf[1:end-1, :] .+ uf[2:end, :]) ./ 2
        wc = (wf[:, 1:end-1] .+ wf[:, 2:end]) ./ 2

        KEz   = uc.^2 .+ vc.^2 .+ wc.^2
        KEcol = 0.5 * rho0 .* vec(sum(KEz .* dzz, dims=2))   # depth-integrated [J/m^2]

        KEavg_t[k] = mean(KEcol[Ix]) * fcKE                  # zonal mean, kJ/m^2
    end
    close(ds)
    return tsel, KEavg_t, fnames
end

fig = Figure(size=(800,500))
ax = Axis(fig[1,1], title="13.1-13; GM-only, no tide; depth-int zonal-mean KE (x=100-1800 km)",
    xlabel="time [days]", ylabel="KE [kJ/m2]")

cmap = cgrad(:viridis, length(LATS), categorical=true)
KE0  = zeros(length(runnms))

for (i, (runnm, lat)) in enumerate(zip(runnms, LATS))
    tsel, KEavg_t, fnames = ke_timeseries(mainnm, runnm)
    KE0[i] = KEavg_t[1]
    println(fnames,"; lat=",lat,"; KE(t=0)=",@sprintf("%.3f",KE0[i])," kJ/m2, KE(end)=",@sprintf("%.3f",KEavg_t[end])," kJ/m2")
    lines!(ax, tsel, KEavg_t, color=cmap[i], linewidth=2, label=string(lat,"°N"))
end
axislegend(ax, position=:rt, labelsize=9, rowgap=0, nbanks=2)
display(fig)
save(string(dirfig,"KE_zonalavg_timeseries_13.1-13.png"), fig)
println("saved KE_zonalavg_timeseries_13.1-13.png")

# initial KE vs latitude, to see the GM-init energy inconsistency directly
figI = Figure(size=(600,400))
axI = Axis(figI[1,1], title="initial (t=0) depth-int zonal-mean KE — 13.1-13 (GM only)",
    xlabel="latitude [°]", ylabel="KE(t=0) [kJ/m2]")
lines!(axI, LATS, KE0, color=:black, linewidth=2)
scatter!(axI, LATS, KE0, color=:black, markersize=10)
display(figI)
save(string(dirfig,"KE0_vs_lat_13.1-13.png"), figI)
println("saved KE0_vs_lat_13.1-13.png")
