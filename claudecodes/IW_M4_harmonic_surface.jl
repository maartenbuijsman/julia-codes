#= IW_M4_harmonic_surface.jl
Maarten Buijsman, USM DMS, 2026-9-16

Step 1 of the M4 beat-wavelength analysis: least-squares harmonic analysis of
the SURFACE velocities at the M4 frequency, for the 11.27-38 series (D2 tide
only, no GM, F=25 kW/m, DX=200 m, 12 latitudes), and the resulting M4 kinetic
energy as a function of along-transect distance.

Purpose: the M4 signal is a superposition of the BOUND (forced) harmonic that
travels with the M2 parent at wavenumber 2*k1(M2), and FREE M4 modes at
k_n(M4). Those two wavenumbers differ, so |u_M4|^2 beats in x with wavelength
2*pi/Delta_k. Extracting the beat wavelength from KE_M4(x) and comparing it
with 2*pi/Delta_k computed from the eigenvalue problem is the goal of the
following step; this script produces the KE_M4(x) profiles it needs.

Sampling
  - last 10 days only (t >= 10 d). The model writes 5-minute output over that
    window (the first 10 days are hourly), giving 2881 samples and ~74.5 per
    M4 cycle -- Nyquist is 144 cpd against M4 at 3.86 cpd, so M4 is very
    comfortably resolved and there is no aliasing concern.
  - every 5th grid point in x (DX=200 m -> 1 km), 2000 points over 2000 km.
    Beat wavelengths are O(100 km), so 1 km is ample.

Fit
  u(t) = a0 + SUM_j [ a_j cos(w_j t) + b_j sin(w_j t) ],  w_j = M2, M4, M6, M8
  solved by least squares. M2 and M4 are separated by 1.93 cpd while the
  Rayleigh resolution over 10 days is 0.1 cpd, so they are independent many
  times over; M6/M8 are included only to stop higher harmonics leaking into
  the M4 estimate. Amplitude of constituent j is sqrt(a_j^2 + b_j^2).

  The design matrix depends only on time, so all 2000 x-positions are solved
  in one backslash (G \ U with matrix RHS).

Energy
  For u = U cos(wt + phi) the time-mean of u^2/2 is U^2/4, so the M4 kinetic
  energy per unit volume at the surface is
      KE_M4 = rho0 * ( |U_M4|^2 + |V_M4|^2 ) / 4     [J/m^3]

Note on staggering: u lives on x-faces and v on cell centres. Both are
subsampled with the same stride, leaving u offset from v by half a cell
(100 m). Against beat wavelengths of O(100 km) that is a 0.1% effect and is
not corrected (correcting it would cost a second strided read per run).

Writes one m4harm_AMZexptXX.YY.jld2 per run to dirout, and a stacked
KE_M4(x) figure over all latitudes.
=#

using NCDatasets, Printf, JLD2, Statistics, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirsim    = string(pth0, "IW/")
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1
savefl  = 1

const T2     = 12 + 25.2/60      # M2 period [h]
const rho0   = 1020.0
const TSPIN  = 10.0              # days; analysis window is t >= TSPIN
const XTARGET = 1000.0           # target along-transect sampling [m]; the
                                 # stride is derived from the run's own dx, so
                                 # this works for the 200 m and 4 km grids alike
                                 # (200 m -> stride 5 = 1 km; 4 km -> stride 1)
const CONSTIT = ["M2","M4","M6","M8"]
const PERIODS = [T2, T2/2, T2/3, T2/4]      # hours
const IM4     = 2                            # index of M4 in CONSTIT

mainnm = 10
runnms = collect(27:38)
runs   = get_runs(mainnm, runnms)
LATS   = [r.lat for r in runs]
fnum   = string(mainnm, ".", runnms[1], "-", runnms[end])

# design matrix for the harmonic fit: [1, cos w1 t, sin w1 t, cos w2 t, ...]
function design_matrix(t_sec)
    nt = length(t_sec)
    G  = ones(nt, 1 + 2*length(PERIODS))
    for (j, Th) in enumerate(PERIODS)
        w = 2π / (Th*3600)
        G[:, 2j]   = cos.(w .* t_sec)
        G[:, 2j+1] = sin.(w .* t_sec)
    end
    return G
end

# amplitude of constituent j from the coefficient matrix (ncoef x nx)
constit_amp(C, j) = sqrt.(C[2j, :].^2 .+ C[2j+1, :].^2)

results = Vector{Any}(undef, length(runnms))

for (ir, runnm) in enumerate(runnms)
    LAT    = LATS[ir]
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    println("\n", fnames, "; lat=", LAT, " -------------------")

    ds = NCDataset(string(dirsim, fnames, ".nc"), "r")
    t  = ds["time"][:]
    It = findall(>=(TSPIN*86400), t)
    nz = length(ds["z_aac"][:])          # surface = topmost cell centre
    dx0 = ds["Δx_caa"][1]
    xstride = max(1, round(Int, XTARGET/dx0))
    xc = ds["x_caa"][1:xstride:end]

    @printf("  window: %.1f-%.1f d, %d samples, dt=%.1f min, %.1f samples/M4 cycle\n",
        t[It[1]]/86400, t[It[end]]/86400, length(It),
        (t[It[2]]-t[It[1]])/60, (T2/2)/((t[It[2]]-t[It[1]])/3600))
    @printf("  grid dx=%.0f m, stride %d -> %.1f km sampling, %d points\n",
        dx0, xstride, dx0*xstride/1e3, length(xc))

    print("  reading surface u ... "); @time U = ds["u"][1:xstride:end, nz, It]
    print("  reading surface v ... "); @time V = ds["v"][1:xstride:end, nz, It]
    close(ds)

    nx = length(xc)
    U  = permutedims(Float64.(U[1:nx, :]))      # nt x nx  (u truncated to v's length)
    V  = permutedims(Float64.(V[1:nx, :]))

    G  = design_matrix(t[It] .- t[It[1]])
    CU = G \ U                                   # ncoef x nx, all positions at once
    CV = G \ V

    ampU4 = constit_amp(CU, IM4);  ampV4 = constit_amp(CV, IM4)
    ampU2 = constit_amp(CU, 1);    ampV2 = constit_amp(CV, 1)

    ke4 = rho0 .* (ampU4.^2 .+ ampV4.^2) ./ 4    # [J/m^3]
    ke2 = rho0 .* (ampU2.^2 .+ ampV2.^2) ./ 4

    # QC: variance explained by the whole harmonic fit, x-averaged
    # columns with zero variance (the boundary cell, where u is identically 0)
    # would give 0/0; they are left as NaN and skipped in the summary
    resU = U .- G*CU
    vtot = vec(sum((U .- mean(U,dims=1)).^2, dims=1))
    vexp = 1 .- vec(sum(resU.^2, dims=1)) ./ vtot
    vexp[vtot .== 0] .= NaN
    vok  = filter(!isnan, vexp)
    @printf("  |U_M4| mean/max = %.4f / %.4f m/s;  KE_M4 mean/max = %.4f / %.4f J/m3\n",
        mean(ampU4), maximum(ampU4), mean(ke4), maximum(ke4))
    @printf("  M4/M2 amplitude ratio (x-mean) = %.3f;  fit explains %.1f%% of u variance (median over x)\n",
        mean(ampU4)/mean(ampU2), 100*median(vok))

    results[ir] = (; lat=LAT, runnm, xc, ampU4, ampV4, ke4, ampU2, ampV2, ke2, vexp)

    if savefl == 1
        @save string(dirout, "m4harm_", fnames, ".jld2") xc ampU4 ampV4 ke4 ampU2 ampV2 ke2 vexp LAT
    end
    U = nothing; V = nothing; CU = nothing; CV = nothing; resU = nothing; GC.gc()
end

## --- stacked KE_M4(x), one row per latitude ---------------------------------
nrun = length(runnms)
fig  = Figure(size=(1000, 110*nrun + 60), fontsize=10)
for ir in 1:nrun
    r  = results[ir]
    ax = Axis(fig[ir, 1],
        ylabel = @sprintf("%.1f°", r.lat),
        xlabel = ir == nrun ? "x [km]" : "",
        xticklabelsvisible = ir == nrun)
    lines!(ax, r.xc./1e3, r.ke4, color=:crimson, linewidth=1.2)
    xlims!(ax, 0, 2000)
    ir == 1 && (ax.title = string("surface M4 kinetic energy, ", fnum,
        "  [J/m³]   (least-squares harmonic fit, days ", Int(TSPIN), "-20)"))
end
display(fig)
figflag == 1 && savefig300(string(dirfig, "M4_KE_vs_x_", fnum, ".png"), fig)

## --- summary -----------------------------------------------------------------
println("\n", "="^78)
println(rpad("lat",8), rpad("mean|U_M4|",13), rpad("max|U_M4|",13),
        rpad("mean KE_M4",13), rpad("max KE_M4",13), "M4/M2")
for r in results
    println(rpad(@sprintf("%.1f",r.lat),8),
        rpad(@sprintf("%.4f",mean(r.ampU4)),13), rpad(@sprintf("%.4f",maximum(r.ampU4)),13),
        rpad(@sprintf("%.4f",mean(r.ke4)),13),   rpad(@sprintf("%.4f",maximum(r.ke4)),13),
        @sprintf("%.3f", mean(r.ampU4)/mean(r.ampU2)))
end
