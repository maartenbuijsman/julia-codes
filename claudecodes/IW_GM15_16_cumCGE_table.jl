#= IW_GM15_16_cumCGE_table.jl
Maarten Buijsman, USM DMS, 2026-10-3

Table: GM level and cumulative cross-scale energy transfer for the two GM
series (15: earlier IC, ~3x GM81 at t = 0; 16: GM81 IC, 1x GM81 over days
10-20), per latitude (0-45 N), F = 25 kW/m.

  GM level   <E/E_GM81> over days 10-20 of the GM-only runs (x.1-12), from the
             caches of IW_GM15_decay_vs_GM76.jl / IW_GM16_decay_vs_GM81.jl
  C_tot      cumulative transfer of the GM + tide run (x.27-38),
                 C = int_{100 km}^{1800 km} rho0 Pi dx   [W/m],  shown as % of F
  C_res      the same for the GM-tide interaction residual (de-GM), as in
             IW_GM15_KEt_CGE_deGM_2x2_ppr.jl:
                 dPi = Pi(GM + tide) - Pi(GM only) - Pi(tide only, 11.27-38)
  R          C_res / C_tide = [(tide+GM) - GM - tide] / tide, the GM-tide
             interaction transfer relative to the tide-only transfer
             (C_tide = same integral for 11.27-38) -- the table printed below.
             Ill-conditioned where the tide alone transfers almost nothing
             (C_tide < ~1% of F at 25-45 N).
  %chg       100 (series 16 - series 15) / |series 15|
  sens       %chg(C_res) / %chg(GM level): about 1 if the interaction scales
             linearly with the GM energy level (full-table output only)

Pi = Pinhxa + Pixxa + Pizxa (depth-integrated, time mean over the analysis
window) from the Etran_*.jld2 files, Gaussian-smoothed at 1600 m as in the
figures. The integral starts at 100 km to leave out the source region.
Prints the table as markdown and LaTeX; the LaTeX is also written to
diagout/GM15_16_cumCGE_table.tex.
=#

using Printf, JLD2, Statistics, Trapz

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

const rho0    = 1020.0
const Lsmooth = 1600.0
const F0      = 25e3              # W/m, mode-1 flux of the tidal runs
const XLO, XHI = 100e3, 1800e3
const RN_TIDE = collect(27:38)
const RN_GM   = collect(1:12)
LATS = [r.lat for r in get_runs(11, RN_TIDE)]
@assert LATS == [r.lat for r in get_runs(16, RN_TIDE)] == [r.lat for r in get_runs(15, RN_TIDE)]

function load_Pi(mainnm, runnms)
    fn0 = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "Etran_", fn0, ".jld2") xc
    A = zeros(length(runnms), length(xc))
    for (i, rn) in enumerate(runnms)
        @load string(dirout, "Etran_", @sprintf("AMZexpt%02i.%02i", mainnm, rn), ".jld2") Πnhxa Πxxa Πzxa
        p = Πnhxa .+ Πxxa .+ Πzxa
        (xc[2] - xc[1]) < 500 && (p = gaussfilt(xc, p, Lsmooth))
        A[i, :] = p
    end
    return xc, A
end

xc, P11 = load_Pi(11, RN_TIDE)
_,  P15 = load_Pi(15, RN_TIDE); _, P15g = load_Pi(15, RN_GM)
_,  P16 = load_Pi(16, RN_TIDE); _, P16g = load_Pi(16, RN_GM)
Ix = findall(XLO .<= xc .<= XHI)
cum(P) = [rho0 * trapz(xc[Ix], P[i, Ix]) for i in axes(P, 1)]     # W/m
pct(v) = 100 .* v ./ F0

Ctot15, Ctot16 = pct(cum(P15)), pct(cum(P16))
Cres15, Cres16 = pct(cum(P15 .- P15g .- P11)), pct(cum(P16 .- P16g .- P11))
Ctide          = pct(cum(P11))

gmlev(c) = [mean(((c["KEs"][i] .+ c["APEs"][i]) ./ c["EGMs"][i])[10 .<= c["tt"][i] .<= 20]) for i in eachindex(LATS)]
G15 = gmlev(load(string(dirout, "GM15_decay_vs_GM76.jld2")))
G16 = gmlev(load(string(dirout, "GM16_decay_vs_GM81.jld2")))

chg(a, b) = 100 .* (b .- a) ./ abs.(a)
dG, dCtot, dCres = chg(G15, G16), chg(Ctot15, Ctot16), chg(Cres15, Cres16)
sens = dCres ./ dG

R15, R16 = 100 .* Cres15 ./ Ctide, 100 .* Cres16 ./ Ctide       # % of the tide-only transfer
dR = chg(R15, R16)

## markdown (simplified table) --------------------------------------------------
println("| lat | GM 15 | GM 16 | ΔGM % | R 15 [%] | R 16 [%] | ΔR % | C_tide [% of F] |")
println("|---|---|---|---|---|---|---|---|")
for i in eachindex(LATS)
    @printf("| %g | %.2f | %.2f | %.0f | %.0f | %.0f | %.0f | %.1f |\n", LATS[i],
            G15[i], G16[i], dG[i], R15[i], R16[i], dR[i], Ctide[i])
end

## LaTeX (simplified table) -------------------------------------------------------
io = IOBuffer()
println(io, "\\begin{table}[h]")
println(io, "\\caption{GM level and GM--tide interaction transfer for the two GM series. GM level: mean \$E/E_{\\mathrm{GM81}}\$ over days 10--20 of the GM-only runs. \$R\$: cumulative cross-scale transfer \$\\int\\rho_0\\Pi\\mathrm{d}x\$ from 100 to 1800~km of the residual (GM~+~tide) \$-\$ (GM only) \$-\$ (tide only), relative to that of the tide-only run, in \\%. \$\\Delta\$: change from series 15 to 16 in \\%.}")
println(io, "\\label{tab:GM1516}")
println(io, "\\begin{tabular}{rrrrrrr}")
println(io, "\\hline")
println(io, " & \\multicolumn{3}{c}{GM level} & \\multicolumn{3}{c}{\$R\$ [\\%]} \\\\")
println(io, "lat [\$^\\circ\$N] & 15 & 16 & \$\\Delta\$ [\\%] & 15 & 16 & \$\\Delta\$ [\\%] \\\\")
println(io, "\\hline")
for i in eachindex(LATS)
    @printf(io, "%g & %.2f & %.2f & %.0f & %.0f & %.0f & %.0f \\\\\n", LATS[i],
            G15[i], G16[i], dG[i], R15[i], R16[i], dR[i])
end
println(io, "\\hline")
println(io, "\\end{tabular}")
println(io, "\\end{table}")
tex = String(take!(io))
println("\n", tex)
write(string(dirout, "GM15_16_cumCGE_table.tex"), tex)
println("written ", dirout, "GM15_16_cumCGE_table.tex")
