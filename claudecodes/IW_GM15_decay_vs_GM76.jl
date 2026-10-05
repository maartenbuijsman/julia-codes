#= IW_GM15_decay_vs_GM76.jl
Maarten Buijsman, USM DMS, 2026-9-30

Decay of the GM-only (no tide, Flux = 0) series 15.1-12 (lat 0-45 N,
redistribution-fix IC) relative to the GM76 reference energy level of each
latitude's own N(z) profile.

E(t) = depth-integrated, zonal-mean (x = 100-1800 km) KE + APE, read from the
raw .nc output every 12 h (t = 0 is the initial condition):
    KE  = rho0/2 int (u^2 + v^2 + w^2) dz,   APE = rho0/2 int b^2/N^2 dz
GM76 reference (total energy per area, WKB form b^2 N0 N(z) E per unit mass):
    E_GM76 = rho0 b^2 N0 E int N dz,   b = 1300 m, N0 = 5.24e-3 rad/s, E = 6.3e-5
The ratio E/E_GM76 is the "GM level" of the run; the paper diagnoses days 10-20.
Per-run results are cached to diagout/GM15_decay_vs_GM76.jld2 (delete to recompute).
=#

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, Trapz

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirsim    = string(pth0, "IW/")
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1

const rho0 = 1020.0
const Egm, N0gm, bgm = 6.3e-5, 5.24e-3, 1300.0
const xlo, xhi = 100e3, 1800e3
const DTS = 0.5                  # sampling interval [days]

rows = get_runs(15, collect(1:12))
LATS = [r.lat for r in rows]

function energy_timeseries(r)
    @load string(dirforce, n2_filename(r)) N2w zfw
    zord = sortperm(zfw)                                  # .nc z runs bottom -> top
    N2c  = (N2w[zord][1:end-1] .+ N2w[zord][2:end]) ./ 2
    EGM  = rho0 * bgm^2 * N0gm * Egm * abs(trapz(zfw, sqrt.(max.(N2w, 0.0))))

    ds = NCDataset(string(dirsim, @sprintf("AMZexpt%02i.%02i", r.mainnm, r.runnm), ".nc"), "r")
    tday = ds["time"][:] ./ 86400
    xc   = ds["x_caa"][:]
    dzz  = reshape(ds["Δz_aac"][:], 1, :)
    Ix   = findall(xlo .<= xc .<= xhi)
    Isel = unique([argmin(abs.(tday .- d)) for d in 0:DTS:floor(tday[end])])
    KE = zeros(length(Isel)); APE = zeros(length(Isel))
    for (k, it) in enumerate(Isel)
        u = ds["u"][:, :, it]; v = ds["v"][:, :, it]; w = ds["w"][:, :, it]; b = ds["b"][:, :, it]
        uc = (u[1:end-1, :] .+ u[2:end, :]) ./ 2
        wc = (w[:, 1:end-1] .+ w[:, 2:end]) ./ 2
        KE[k]  = mean(0.5rho0 .* vec(sum((uc.^2 .+ v.^2 .+ wc.^2) .* dzz, dims = 2))[Ix])
        APE[k] = mean(0.5rho0 .* vec(sum(b.^2 ./ max.(N2c, 1e-12)' .* dzz, dims = 2))[Ix])
    end
    t = tday[Isel]
    close(ds)
    return t, KE, APE, EGM
end

fcache = string(dirout, "GM15_decay_vs_GM76.jld2")
if isfile(fcache)
    @load fcache tt KEs APEs EGMs
else
    tt = Vector{Vector{Float64}}(); KEs = similar(tt); APEs = similar(tt); EGMs = Float64[]
    for r in rows
        t, KE, APE, EGM = energy_timeseries(r)
        push!(tt, t); push!(KEs, KE); push!(APEs, APE); push!(EGMs, EGM)
        @printf("15.%02i lat %4.1f done\n", r.runnm, r.lat)
    end
    jldsave(fcache; tt, KEs, APEs, EGMs, LATS)
end

## table -------------------------------------------------------------------
ratio   = [(KEs[i] .+ APEs[i]) ./ EGMs[i] for i in eachindex(rows)]
I1020   = [findall(10 .<= tt[i] .<= 20) for i in eachindex(rows)]
r0      = [ratio[i][1] for i in eachindex(rows)]
r10     = [ratio[i][argmin(abs.(tt[i] .- 10))] for i in eachindex(rows)]
r20     = [ratio[i][end] for i in eachindex(rows)]
rm1020  = [mean(ratio[i][I1020[i]]) for i in eachindex(rows)]
@printf("%5s %8s %8s %8s %8s %8s %8s %8s\n", "lat", "E_GM76", "E(0)", "E/GM t=0", "t=10", "t=20", "mean10-20", "E20/E0")
for i in eachindex(rows)
    E = KEs[i] .+ APEs[i]
    @printf("%5.1f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", LATS[i], EGMs[i]/1e3, E[1]/1e3,
            r0[i], r10[i], r20[i], rm1020[i], E[end]/E[1])
end

## figure -----------------------------------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size = (18cm_to_pt, 8.5cm_to_pt), fontsize = 10)
cmap = cgrad(:viridis, length(LATS), categorical = true)
axA = Axis(fig[1, 1], xlabel = "time [days]", ylabel = "E / E_GM76",
           title = "(a) GM only (15.1-12): (KE+APE) / E_GM76", titlesize = 10,
           xticks = 0:5:20, xtickalign = 1, ytickalign = 1)
vspan!(axA, 10, 20, color = (:gray, 0.15))
hlines!(axA, [1.0], color = :black, linestyle = :dash, linewidth = 1)
for i in eachindex(rows)
    lines!(axA, tt[i], ratio[i], color = cmap[i], linewidth = 1.5, label = @sprintf("%.1f°N", LATS[i]))
end
ylims!(axA, 0, nothing)
axB = Axis(fig[1, 2], xlabel = "latitude [°N]", ylabel = "E / E_GM76",
           title = "(b) by latitude", titlesize = 10, xticks = 0:10:40,
           xtickalign = 1, ytickalign = 1)
hlines!(axB, [1.0], color = :black, linestyle = :dash, linewidth = 1)
scatterlines!(axB, LATS, r0,     color = :black,  markersize = 6, label = "t = 0")
scatterlines!(axB, LATS, rm1020, color = :tomato, markersize = 6, label = "mean days 10-20")
scatterlines!(axB, LATS, r20,    color = :dodgerblue, markersize = 6, label = "t = 20 d")
ylims!(axB, 0, nothing)
# legend floats in the empty band between the day-10-20 curves (<= 1.65) and
# the t = 0 curve (>= 2.99); clearance checked against the data below
legB = axislegend(axB, position = (0.95, 0.62), labelsize = 8, framevisible = false, rowgap = 0,
                  patchsize = (14, 6), padding = (2, 2, 2, 2))
Legend(fig[1, 3], axA, labelsize = 8, framevisible = false, rowgap = 0, patchsize = (14, 6))
colsize!(fig.layout, 1, Relative(0.47))
display(fig)
# legend clearance: legend box (pixels) -> data coords of axB, vs every curve inside its x-range
let bb = legB.layoutobservables.computedbbox[], vp = axB.scene.viewport[], lim = axB.finallimits[]
    tox(px) = lim.origin[1] + (px - vp.origin[1]) / vp.widths[1] * lim.widths[1]
    toy(py) = lim.origin[2] + (py - vp.origin[2]) / vp.widths[2] * lim.widths[2]
    x0, x1 = tox(bb.origin[1]), tox(bb.origin[1] + bb.widths[1])
    y0, y1 = toy(bb.origin[2]), toy(bb.origin[2] + bb.widths[2])
    Ix = findall(x0 - 2.5 .<= LATS .<= x1 + 2.5)     # include the segments entering the box
    hit = any(y -> y0 <= y <= y1, vcat(r0[Ix], rm1020[Ix], r20[Ix], [1.0]))
    @printf("legend (b): x %.1f-%.1f, y %.2f-%.2f -> %s\n", x0, x1, y0, y1, hit ? "[OVERLAP]" : "[CLEAR]")
end
if figflag == 1
    fout = string(dirfig, "GM15_decay_vs_GM76.png")
    savefig300(fout, fig)
    println("saved ", fout)
end
