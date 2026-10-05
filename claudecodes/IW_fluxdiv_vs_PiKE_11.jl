#= IW_fluxdiv_vs_PiKE_11.jl
Maarten Buijsman, USM DMS, 2026-10-3

Check of the band energy budget for the tide-only runs 11.27-30 (lat 0, 2.5,
5, 10 N; 25 kW/m): does the cross-scale transfer from coarse-graining (which
contains KE terms only, IW_coarsegraining_tile.jl) account for the loss of
D2 energy flux? Time means over the analysis window, depth-integrated.

  top row     -dF_D2/dx, +dF_HH/dx and rho0*Pi_KE   [mW/m2]
  bottom row  cumulative from x0 = 100 km:
              F_D2(x0) - F_D2(x),  F_HH(x) - F_HH(x0),  int rho0 Pi_KE dx  [kW/m]
F = Fx + FKx + FAx (pressure, KE-advective and APE-advective flux; the APE is
the nonlinear APE of IW_total_energetics_tile.jl), band t = D2 (tidal), h = HH
(supertidal). The divergences are taken from fluxes smoothed with a Gaussian
of SIG = 20 km (raw band fluxes carry short-wavelength interference
wiggles); Pi is smoothed with the same SIG in the top row only.
=#

using Printf, JLD2, Statistics, Trapz, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1
const rho0 = 1020.0
const SIG  = 20e3
const X0, X1 = 100e3, 1800e3
rows = get_runs(11, collect(27:30))

cm_to_pt = 72/2.54
fig = Figure(size = (18cm_to_pt, 12cm_to_pt), fontsize = 10)
for (c, r) in enumerate(rows)
    fn = @sprintf("AMZexpt%02i.%02i", r.mainnm, r.runnm)
    e  = load(string(dirout, "energetics_", fn, ".jld2"))
    t  = load(string(dirout, "Etran_", fn, ".jld2"))
    xe = e["xc"]; xp = t["xc"]
    Ft = gaussfilt(xe, e["Fxt"] .+ e["FKxt"] .+ e["FAxt"], SIG)
    Fh = gaussfilt(xe, e["Fxh"] .+ e["FKxh"] .+ e["FAxh"], SIG)
    Pi = rho0 .* (t["Πnhxa"] .+ t["Πxxa"] .+ t["Πzxa"])          # W/m2
    xm = (xe[1:end-1] .+ xe[2:end]) ./ 2
    divt = -diff(Ft) ./ diff(xe); divh = diff(Fh) ./ diff(xe)       # W/m2
    Ie = findall(X0 + 2SIG .<= xm .<= X1); Ip = findall(X0 .<= xp .<= X1)   # divergence: skip 2 SIG next to x0 (smoothing edge)

    ax1 = Axis(fig[1, c], title = @sprintf("%s  %.1f°N", "11.$(r.runnm)", r.lat), titlesize = 10,
               ylabel = c == 1 ? "[mW/m²]" : "", xticklabelsvisible = false,
               xtickalign = 1, ytickalign = 1, xticks = (0:500:2000, ["0", "", "1000", "", "2000"]))
    hlines!(ax1, [0.0], color = :gray70, linewidth = 0.8)
    lines!(ax1, xm[Ie] ./ 1e3, 1e3 .* divt[Ie], color = :red, linewidth = 1.2, label = "−dF_D2/dx")
    lines!(ax1, xm[Ie] ./ 1e3, 1e3 .* divh[Ie], color = :green, linewidth = 1.2, label = "+dF_HH/dx")
    lines!(ax1, xp[Ip] ./ 1e3, 1e3 .* gaussfilt(xp, Pi, SIG)[Ip], color = :black, linewidth = 1.5, label = "ρ₀Π_KE")

    cumt = (Ft[findfirst(>=(X0), xe)] .- Ft) ./ 1e3
    cumh = (Fh .- Fh[findfirst(>=(X0), xe)]) ./ 1e3
    Ix   = findall(X0 .<= xe .<= X1)
    cumP = [rho0 * trapz(xp[Ip[1]:i], (t["Πnhxa"] .+ t["Πxxa"] .+ t["Πzxa"])[Ip[1]:i]) for i in Ip] ./ 1e3
    ax2 = Axis(fig[2, c], xlabel = "x [km]", ylabel = c == 1 ? "cumulative [kW/m]" : "",
               xtickalign = 1, ytickalign = 1, xticks = (0:500:2000, ["0", "", "1000", "", "2000"]))
    lines!(ax2, xe[Ix] ./ 1e3, cumt[Ix], color = :red, linewidth = 1.2)
    lines!(ax2, xe[Ix] ./ 1e3, cumh[Ix], color = :green, linewidth = 1.2)
    lines!(ax2, xp[Ip] ./ 1e3, cumP, color = :black, linewidth = 1.5)
    lines!(ax2, xp[Ip] ./ 1e3, 2 .* cumP, color = :black, linewidth = 1.0, linestyle = :dash)
    for ax in (ax1, ax2); xlims!(ax, 0, 2000); end
    @printf("%s lat %4.1f: at 1800 km  D2 loss %.1f, HH gain %.1f, int rho0 Pi_KE %.1f kW/m (ratio D2/Pi = %.2f)\n",
            fn, r.lat, cumt[Ix[end]], cumh[Ix[end]], cumP[end], cumt[Ix[end]] / cumP[end])
end
Legend(fig[3, 1:4], [LineElement(color = :red), LineElement(color = :green), LineElement(color = :black, linewidth = 1.5),
                     LineElement(color = :black, linestyle = :dash)],
       ["D2 flux loss  (−dF_D2/dx; cumulative)", "HH flux gain (+dF_HH/dx; cumulative)", "ρ₀Π_KE", "2 × cumulative ρ₀Π_KE"],
       orientation = :horizontal, nbanks = 2, framevisible = false, labelsize = 8, patchsize = (16, 6), colgap = 14)
rowgap!(fig.layout, 6)
display(fig)
if figflag == 1
    fout = string(dirfig, "fluxdiv_vs_PiKE_11.27-30.png")
    savefig300(fout, fig)
    println("saved ", fout)
end
