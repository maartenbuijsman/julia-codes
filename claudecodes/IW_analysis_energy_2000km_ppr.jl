#= IW_analysis_energy_2000km_ppr.jl
Maarten Buijsman, USM DMS, 2026-9-26
Paper figures, for the 11.27-38 series (12 sims, F=25 kW/m, varying Mercator
N2, mainnm=11):

1) 2x2 grid of x-vs-latitude heatmaps of simulated tidal-band (KEt) and
   supertidal-band (KEh) depth-integrated KE -- top row KEt and KEt/KEtmax,
   bottom row KEh and KEh/KEtmax, based on IW_analysis_energy_2000km.jl.
   KEtmax is the theoretical depth-integrated mode-1 KE (not a function of
   x), computed with the same F=E*Cg / KE-APE-polarization-split formula as
   mode1_theory() in IW_mode1_theory_vs_lat.jl.

2) 2x2 grid of x-vs-latitude heatmaps of the coarse-graining cross-scale
   energy transfer Π (and its cumulative x-integral), based on
   IW_analysis_coarsegr_2000km.jl -- top row rho0*Π and rho0*Π/F, bottom row
   rho0*ΣΠdx and rho0*ΣΠdx/F. Π as saved is a depth-integrated MASS-SPECIFIC
   power (W/kg·m); scaling by rho0 converts it to an actual power density
   comparable to dF/dx (and its x-integral to something comparable to F
   itself), so dividing by the run's prescribed flux F gives, respectively,
   a local fractional flux-divergence rate [1/m] and the cumulative fraction
   of F converted by cross-scale transfer at each x [dimensionless].
=#

println("number of threads is ",Threads.nthreads())

using Pkg, NCDatasets, Printf, CairoMakie, Statistics, JLD2

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";
    dirEIG = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirfig = string(pth0,"figs/");
    dirout = string(pth0,"diagout/");
    dirforce = string(pth0,"IW/forcingfiles/");
    dirEIG = string(pth0,"IW/forcingfiles/");
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename()

# print figures
figflag = 1

const T2 = 12+25.2/60
const rho0=1020;
const ω  = 2π/(T2*3600)

# run names --------------------------------

# 200m, D2-tide-only series (mainnm=11), F=25 kW/m, varying Mercator N2,
# excluding the lat=50 transect (last runnm in the block) -- 12 sims
mainnm  = 11
runnms  = collect(27:38)

runs = get_runs(mainnm, runnms)   # errors immediately if a runnm isn't in RUN_TABLE
LATS = [r.lat for r in runs]

fnum = string(mainnm,".",runnms[1],"-",runnms[end])


## load simulated tidal-band (KEt) and supertidal-band (KEh) KE per run --------
fnames0  = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
@load string(dirout, "energetics_", fnames0, ".jld2") xc
KEtr = zeros(length(runnms), length(xc))   # tidal-band KE
KEhr = zeros(length(runnms), length(xc))   # supertidal-band KE

for i=1:length(runnms)
    runnm = runnms[i]; LAT = LATS[i];

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
    println(fnames,"; lat=",LAT," -------------------")

    fnameout = string("energetics_",fnames,".jld2")
    @load string(dirout,fnameout) KEt KEh
    KEtr[i,:] = KEt;
    KEhr[i,:] = KEh;
end


## theoretical depth-integrated mode-1 KEtmax per run, same F=E*Cg + KE/APE
## polarization-split formula as mode1_theory() in IW_mode1_theory_vs_lat.jl
## (only the KE half is needed here -- not a function of x) --------------------
function mode1_KEtmax(row)
    Fx     = row.Flux
    fcor   = coriolis(row.lat)
    nonhyd = row.DX < 500 ? 1 : 0

    fnamegrid = n2_filename(row)
    @load string(dirforce, fnamegrid) N2w zfw

    kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
        sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)

    Etot = Fx / Cgn[1]                  # F = E*Cg
    fw2  = (fcor/ω)^2
    rat  = (1-fw2) / (1+fw2)            # APE/KE, from IW_Energy_scenarios.jl
    KE   = Etot / (1+rat)
    return KE
end

KEtmax = [mode1_KEtmax(row) for row in runs]   # [J/m²], one value per run/latitude


## 2x2 heatmaps (x vs latitude): (top) KEt, KEt/KEtmax; (bottom) KEh, KEh/KEtmax -
LdomH   = 2000e3;
fcKE    = 1e-3               # J/m² -> kJ/m² (KE depth-integrated: kg/m³·m²/s²·m = J/m²)
cmapKE  = :thermal

KEtrn = KEtr ./ KEtmax        # tidal KE as fraction of theoretical mode-1 KEtmax
KEhrn = KEhr ./ KEtmax        # supertidal KE as fraction of theoretical mode-1 KEtmax

# paper size: 18 cm wide x 16 cm tall at fontsize 10pt, same convention as IW_mode1_theory_vs_lat.jl
cm_to_pt = 72/2.54
fig_w = 18*cm_to_pt
fig_h = 16*cm_to_pt
fig = Figure(size=(fig_w, fig_h), fontsize=10)

# explicit bbox layout (see functions/subplot_hor_vertpos.jl for the vers/vere
# swap gotcha) instead of GridLayout -- panels AND colorbars are both placed
# via bbox (colorbar_bbox, ported from Maarten's MATLAB colorbar_pos.m),
# bypassing the Figure's own layout entirely, so they stay aligned together
pos = subplot_hor_vertpos(2, 2, 0.1, 0.1, 0.1, 0.05, 0.15, 0.08)
bb(i)  = BBox(subplot_bbox(pos[i], fig_w, fig_h)...)
cbb(i) = BBox(colorbar_bbox(pos[i], fig_w, fig_h, 0.02, 0.02, 1.0, 0.0)...)

# units folded into the panel title instead of a colorbar label, to save
# horizontal space for the colorbar itself
ax1 = Axis(fig.scene, bbox=bb(1), title = string("(a) KEt, ",fnum," [kJ/m²]"), ylabel = "latitude [°]")
ax2 = Axis(fig.scene, bbox=bb(2), title = "(b) KEt / KEtmax")
ax3 = Axis(fig.scene, bbox=bb(3), title = "(c) KEh [kJ/m²]", xlabel = "x [km]", ylabel = "latitude [°]")
ax4 = Axis(fig.scene, bbox=bb(4), title = "(d) KEh / KEtmax", xlabel = "x [km]")

hm1 = heatmap!(ax1, xc/1e3, LATS, (KEtr*fcKE)', colormap = cmapKE)
Colorbar(fig.scene, hm1, bbox = cbb(1))

hm2 = heatmap!(ax2, xc/1e3, LATS, KEtrn', colormap = cmapKE, colorrange = (0, 1))
Colorbar(fig.scene, hm2, bbox = cbb(2))

hm3 = heatmap!(ax3, xc/1e3, LATS, (KEhr*fcKE)', colormap = cmapKE)
Colorbar(fig.scene, hm3, bbox = cbb(3))

hm4 = heatmap!(ax4, xc/1e3, LATS, KEhrn', colormap = cmapKE, colorrange = (0, 1))
Colorbar(fig.scene, hm4, bbox = cbb(4))

# bbox-parented axes don't auto-update their limits from plotted data
for ax in (ax1, ax2, ax3, ax4)
    autolimits!(ax)
    xlims!(ax, 0, LdomH/1e3)
end

display(fig)

if figflag==1; savefig300(string(dirfig,"KEt_KEh_norm_",fnum,".png"), fig)
end


## load cross-scale energy transfer (coarse-graining flux) Π per run, same
## source files/smoothing as IW_analysis_coarsegr_2000km.jl --------------------
fnames0c = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
@load string(dirout, "Etran_", fnames0c, ".jld2") xc
CGE      = zeros(length(runnms), length(xc))   # Π=Πnh+Πx+Πz, depth-integrated [W·m/kg]
dxC      = xc[2]-xc[1]
LsmoothC = 1600   # Gaussian σ [m], same convention as IW_analysis_coarsegr_2000km.jl

for i=1:length(runnms)
    runnm = runnms[i]; LAT = LATS[i];

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
    println(fnames,"; lat=",LAT," -------------------")

    fnameout = string("Etran_",fnames,".jld2")
    @load string(dirout,fnameout) Πnhxa Πxxa Πzxa
    pietot = Πnhxa .+ Πxxa .+ Πzxa
    CGE[i,:] = dxC < 500 ? gaussfilt(xc, pietot, LsmoothC) : pietot
end

CGEsum = zeros(length(runnms), length(xc))     # cumulative x-integral of Π [W·m²/kg]
for i=1:length(runnms)
    CGEsum[i,:] = cumtrapz(xc, CGE[i,:])
end

FLUXES = [r.Flux for r in runs]   # prescribed mode-1 flux per run [W/m]

## Π is a depth-integrated MASS-SPECIFIC power (W/kg · m, since the per-depth
## terms are W/kg and get integrated over z in meters) -- multiplying by rho0
## [kg/m³] converts it to an actual power density, comparable to dF/dx:
##   rho0*Π      [W/m²]  ~ dF/dx  (one more x-integral of Π turns W/kg·m into
##                                  W/kg·m², so rho0*ΣΠdx works out to W/m)
##   rho0*ΣΠdx   [W/m]   ~ F      (directly comparable to the prescribed flux)
## dividing those by F gives, respectively, a local fractional flux-divergence
## rate [1/m] and the cumulative fraction of F converted by cross-scale
## transfer at each x [dimensionless] -- this is the "give it a try" part
CGEphys    = rho0 .* CGE            # [W/m²],  ~ dF/dx
CGEsumphys = rho0 .* CGEsum         # [W/m],   ~ F

CGEnorm    = CGEphys    ./ FLUXES   # (rho0*Π)/F      [1/m]
CGEsumnorm = CGEsumphys ./ FLUXES   # (rho0*ΣΠdx)/F   [-]

println("rho0*Π    min/max = ", @sprintf("%.2e", minimum(CGEphys)),    " / ", @sprintf("%.2e", maximum(CGEphys)),    " W/m²")
println("rho0*ΣΠdx min/max = ", @sprintf("%.2e", minimum(CGEsumphys)), " / ", @sprintf("%.2e", maximum(CGEsumphys)), " W/m")


## 2x2 heatmaps (x vs latitude): (top) rho0*Π, rho0*Π/F; (bottom) rho0*ΣΠdx, rho0*ΣΠdx/F
cmapCGE = Reverse(:RdBu_5)   # diverging -- Π can transfer energy either up- or down-scale
cmax1 = maximum(abs.(CGEphys))
cmax2 = maximum(abs.(CGEnorm))
cmax3 = maximum(abs.(CGEsumphys))
cmax4 = maximum(abs.(CGEsumnorm))

# (b) and (c) span several orders of magnitude, so Makie's per-tick scientific
# notation gets clipped by the narrow colorbar; pull out a single common
# power-of-10 instead, scale the data by it, and fold the exponent into the
# title so the colorbar itself just shows plain O(1) numbers
pow10_exponent(x) = floor(Int, log10(x))
y2 = pow10_exponent(cmax2); scale2 = 10.0^y2; cmax2s = cmax2/scale2
y3 = pow10_exponent(cmax3); scale3 = 10.0^y3; cmax3s = cmax3/scale3

figC = Figure(size=(fig_w, fig_h), fontsize=10)

ax1c = Axis(figC.scene, bbox=bb(1), title = string("(a) ρ₀Π, ",fnum," [W/m²]"), ylabel = "latitude [°]")
ax2c = Axis(figC.scene, bbox=bb(2), title = rich("(b) ρ₀Π / F [×10", superscript(string(y2)), " 1/m]"))
ax3c = Axis(figC.scene, bbox=bb(3), title = rich("(c) ρ₀ΣΠdx [×10", superscript(string(y3)), " W/m]"), xlabel = "x [km]", ylabel = "latitude [°]")
ax4c = Axis(figC.scene, bbox=bb(4), title = "(d) ρ₀ΣΠdx / F [-]", xlabel = "x [km]")

hm1c = heatmap!(ax1c, xc/1e3, LATS, CGEphys',        colormap = cmapCGE, colorrange = (-cmax1,  cmax1))
Colorbar(figC.scene, hm1c, bbox = cbb(1))

hm2c = heatmap!(ax2c, xc/1e3, LATS, (CGEnorm./scale2)',    colormap = cmapCGE, colorrange = (-cmax2s, cmax2s))
Colorbar(figC.scene, hm2c, bbox = cbb(2))

hm3c = heatmap!(ax3c, xc/1e3, LATS, (CGEsumphys./scale3)', colormap = cmapCGE, colorrange = (-cmax3s, cmax3s))
Colorbar(figC.scene, hm3c, bbox = cbb(3))

hm4c = heatmap!(ax4c, xc/1e3, LATS, CGEsumnorm',     colormap = cmapCGE, colorrange = (-cmax4,  cmax4))
Colorbar(figC.scene, hm4c, bbox = cbb(4))

for ax in (ax1c, ax2c, ax3c, ax4c)
    autolimits!(ax)
    xlims!(ax, 0, LdomH/1e3)
end

display(figC)

if figflag==1; savefig300(string(dirfig,"CGE_CGEsum_norm_",fnum,".png"), figC)
end


## ============================================================================
## Comparison figures: 4-row (KEt/KEh) and 3-row (CGE/Π) grids, one column per
## run block, shared color scale per row so columns are directly comparable.
## Used twice below with different block/label sets -- once holding F fixed
## and varying the N² treatment, once holding N² fixed (variable Mercator
## N²(z)) and varying the forcing amplitude F -------------------------------
function load_KE_block(mainnm, runnms)
    runs = get_runs(mainnm, runnms)
    LATS = [r.lat for r in runs]

    fnames0b = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "energetics_", fnames0b, ".jld2") xc
    KEtrb = zeros(length(runnms), length(xc))
    KEhrb = zeros(length(runnms), length(xc))
    for i=1:length(runnms)
        fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnms[i])
        @load string(dirout,"energetics_",fnames,".jld2") KEt KEh
        KEtrb[i,:] = KEt
        KEhrb[i,:] = KEh
    end
    KEtmaxb = [mode1_KEtmax(row) for row in runs]
    return xc, LATS, KEtrb, KEhrb, KEtmaxb
end

function load_CGE_block(mainnm, runnms, runs)
    fnames0d = @sprintf("AMZexpt%02i.%02i", mainnm, runnms[1])
    @load string(dirout, "Etran_", fnames0d, ".jld2") xc
    dxCb = xc[2]-xc[1]
    CGEb = zeros(length(runnms), length(xc))
    for i=1:length(runnms)
        fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnms[i])
        @load string(dirout,"Etran_",fnames,".jld2") Πnhxa Πxxa Πzxa
        pietot = Πnhxa .+ Πxxa .+ Πzxa
        CGEb[i,:] = dxCb < 500 ? gaussfilt(xc, pietot, 1600) : pietot
    end
    CGEsumb = zeros(length(runnms), length(xc))
    for i=1:length(runnms)
        CGEsumb[i,:] = cumtrapz(xc, CGEb[i,:])
    end
    FLUXESb = [r.Flux for r in runs]
    CGEphysb    = rho0 .* CGEb
    CGEsumphysb = rho0 .* CGEsumb
    return xc, CGEphysb, CGEsumphysb, CGEphysb./FLUXESb, CGEsumphysb./FLUXESb
end

## --- shared layout (both comparison figures below reuse these) ---------------
fig_w2 = 18*cm_to_pt
fig_h2 = 18*cm_to_pt                      # 4-row (KEt/KEh) figure
# Dsh=Dsv=0 -- converted to the zero-gap convention (see
# IW_energy_flux_CGE_2col_ppr.jl / the compareF merged figure); this function
# is now called only for "compareN2", so the change is isolated to that file.
pos2 = subplot_hor_vertpos(3, 4, 0.08, 0.15, 0.08, 0.04, 0.0, 0.0)
const TICKSIZE2 = 4.0
xticks2(c, nblk) = (collect(0.0:500:2000),
                    c==nblk ? string.(0:500:2000) : vcat(string.(0:500:1500), ""))
# It is now TWO rows: the ρ₀ΣΠdx/F row was dropped (it is the ρ₀ΣΠdx row divided
# by a constant per column, so it carried no information the row above did not).
# The two surviving rows keep exactly the panel size they had as part of the
# 3-row figure -- PANELH3 is that height in points -- so the canvas simply loses
# one row's worth of height instead of the panels growing to fill it. The
# margins are given in POINTS and converted to fractions, which is what keeps
# them fixed in physical size as the row count changes; the bottom one is sized
# for the x tick labels PLUS the "x [km]" label, which used to be clipped off
# the canvas entirely.
# Dsh=Dsv=0 -- converted to the zero-gap convention (was 0.02/0.02, matching
# pos2's OLD value; now both are 0, matching pos2's current value above and the
# compareF merged figure). Shared by both compareN2 and compareF.
const NROW3   = 2
const PANELH3 = 104.0                       # panel height [pt], as in the 3-row version
const MARGB3  = 42.0                        # bottom margin [pt]: x tick labels + xlabel (56*0.75)
const MARGT3  = 21.0                        # top margin [pt]: column titles
fig_h3 = NROW3*PANELH3 + MARGB3 + MARGT3
pos3 = subplot_hor_vertpos(3, NROW3, 0.08, 0.15, MARGB3/fig_h3, MARGT3/fig_h3, 0.0, 0.0)
const TICKSIZE3 = 4.0
bb2(i)  = BBox(subplot_bbox(pos2[i], fig_w2, fig_h2)...)
# hbar<1 + vertoffbar>0: small gap top/bottom of each row's colorbar so their
# tick labels don't collide at the shared row seams (same fix as posM's cbbM)
cbb2(i) = BBox(colorbar_bbox(pos2[i], fig_w2, fig_h2, 0.015, 0.015, 0.92, 0.04)...)
bb3(i)  = BBox(subplot_bbox(pos3[i], fig_w2, fig_h3)...)
cbb3(i) = BBox(colorbar_bbox(pos3[i], fig_w2, fig_h3, 0.015, 0.015, 0.92, 0.04)...)
panelidx(r,c) = (r-1)*3 + c   # row r, col c (1-3) -> pos2/pos3 index (row-major, matches subplot_hor_vertpos fill order)

# letters numbered vertically (down each column first): (1,1)=a,(2,1)=b,...
fignumletter(r,c)  = string("(", Char('a' + (c-1)*4 + (r-1)), ")")   # 4-row figure -> a-l
fignumletter3(r,c) = string("(", Char('a' + (c-1)*NROW3 + (r-1)), ")")  # CGE figure -> a-f

cmapNorm = :linear_wcmr_100_45_c42_n256   # white -> purple -> red; normalized (0-1)
                                          # quantities get their own colormap + fixed
                                          # range, so they stay comparable across series

## --- load all per-block data needed by both comparison figures ---------------
function load_comparison_data(mainnm, blocks)
    nb = length(blocks)
    xc_c, LATS_c = Vector{Any}(undef,nb), Vector{Any}(undef,nb)
    KEtr_c, KEhr_c, KEtrn_c, KEhrn_c = [Vector{Any}(undef,nb) for _ in 1:4]
    CGEphys_c, CGEsumphys_c, CGEnorm_c, CGEsumnorm_c = [Vector{Any}(undef,nb) for _ in 1:4]

    for (b, rn) in enumerate(blocks)
        runsb = get_runs(mainnm, rn)
        xcb, LATSb, KEtrb, KEhrb, KEtmaxb = load_KE_block(mainnm, rn)
        xc_c[b] = xcb; LATS_c[b] = LATSb
        KEtr_c[b]  = KEtrb;            KEhr_c[b]  = KEhrb
        KEtrn_c[b] = KEtrb ./ KEtmaxb; KEhrn_c[b] = KEhrb ./ KEtmaxb

        _, CGEphysb, CGEsumphysb, CGEnormb, CGEsumnormb = load_CGE_block(mainnm, rn, runsb)
        CGEphys_c[b] = CGEphysb;       CGEsumphys_c[b] = CGEsumphysb
        CGEnorm_c[b] = CGEnormb;       CGEsumnorm_c[b] = CGEsumnormb
    end
    return xc_c, LATS_c, KEtr_c, KEhr_c, KEtrn_c, KEhrn_c, CGEphys_c, CGEsumphys_c, CGEnorm_c, CGEsumnorm_c
end

## --- KEt/KEh comparison: 4 rows (a-d/e-h/...), 1 column per block ------------
function make_KE_compare_figure(xc_cmp, LATS_cmp, KEtr_cmp, KEhr_cmp, KEtrn_cmp, KEhrn_cmp, labels, savename)
    nblk = length(labels)
    cmaxKEt = maximum(maximum(m) for m in KEtr_cmp) * fcKE
    cmaxKEh = maximum(maximum(m) for m in KEhr_cmp) * fcKE

    figKEcmp = Figure(size=(fig_w2, fig_h2), fontsize=10)
    axKEcmp = Matrix{Axis}(undef, 4, 3)
    for r in 1:4, c in 1:3
        axKEcmp[r,c] = Axis(figKEcmp.scene, bbox=bb2(panelidx(r,c)),
            title  = r==1 ? labels[c] : "",
            xtickalign = 1, ytickalign = 1, xticksize = TICKSIZE2, yticksize = TICKSIZE2,
            xticks = xticks2(c, nblk), yticks = 0:10:40,
            xlabel = r==4 ? "x [km]" : "",
            ylabel = c==1 ? "latitude [°]" : "",
            xticklabelsvisible = r==4, yticklabelsvisible = c==1)
    end

    for c in 1:nblk
        hma = heatmap!(axKEcmp[1,c], xc_cmp[c]/1e3, LATS_cmp[c], (KEtr_cmp[c]*fcKE)',  colormap=cmapKE,   colorrange=(0,cmaxKEt))
        hmb = heatmap!(axKEcmp[2,c], xc_cmp[c]/1e3, LATS_cmp[c], KEtrn_cmp[c]',        colormap=cmapNorm, colorrange=(0,1))
        hmc = heatmap!(axKEcmp[3,c], xc_cmp[c]/1e3, LATS_cmp[c], (KEhr_cmp[c]*fcKE)',  colormap=cmapKE,   colorrange=(0,cmaxKEh))
        hmd = heatmap!(axKEcmp[4,c], xc_cmp[c]/1e3, LATS_cmp[c], KEhrn_cmp[c]',        colormap=cmapNorm, colorrange=(0,1))
        if c==nblk
            Colorbar(figKEcmp.scene, hma, bbox=cbb2(panelidx(1,c)), label="KEt [kJ/m²]")
            Colorbar(figKEcmp.scene, hmb, bbox=cbb2(panelidx(2,c)), label="KEt / KEtmax")
            Colorbar(figKEcmp.scene, hmc, bbox=cbb2(panelidx(3,c)), label="KEh [kJ/m²]")
            Colorbar(figKEcmp.scene, hmd, bbox=cbb2(panelidx(4,c)), label="KEh / KEtmax")
        end
        for r in 1:4
            autolimits!(axKEcmp[r,c])
            xlims!(axKEcmp[r,c], 0, LdomH/1e3)
            text_fignum!(axKEcmp[r,c], fignumletter(r,c); bckclr=:white)
        end
    end

    display(figKEcmp)
    if figflag==1; savefig300(string(dirfig,"KEt_KEh_norm_",savename,".png"), figKEcmp)
    end
    return figKEcmp
end

## --- CGE/Π comparison: 2 rows, 1 column per block. Two normalized rows have
## been dropped over time, both for the same reason -- dividing a row by a
## constant does not make a new row: rho0*Π/F (constant F within a column) and
## now rho0*ΣΠdx/F ------------------------------------------------------------
function make_CGE_compare_figure(xc_cmp, LATS_cmp, CGEphys_cmp, CGEsumphys_cmp, labels, savename)
    nblk = length(labels)
    cmaxA_cmp = maximum(maximum(abs.(m)) for m in CGEphys_cmp)
    cmaxC_cmp = maximum(maximum(abs.(m)) for m in CGEsumphys_cmp)
    yC_cmp = pow10_exponent(cmaxC_cmp); scaleC_cmp = 10.0^yC_cmp; cmaxC_cmps = cmaxC_cmp/scaleC_cmp

    figCGEcmp = Figure(size=(fig_w2, fig_h3), fontsize=10)
    axCGEcmp = Matrix{Axis}(undef, NROW3, 3)
    for r in 1:NROW3, c in 1:3
        axCGEcmp[r,c] = Axis(figCGEcmp.scene, bbox=bb3(panelidx(r,c)),
            title  = r==1 ? labels[c] : "",
            xtickalign = 1, ytickalign = 1, xticksize = TICKSIZE3, yticksize = TICKSIZE3,
            xticks = (0:500:2000, c==nblk ? string.(0:500:2000) : vcat(string.(0:500:1500), "")),
            yticks = 0:10:40,
            xlabel = r==NROW3 ? "x [km]" : "",
            ylabel = c==1 ? "latitude [°]" : "",
            xticklabelsvisible = r==NROW3, yticklabelsvisible = c==1)
    end

    for c in 1:nblk
        hma = heatmap!(axCGEcmp[1,c], xc_cmp[c]/1e3, LATS_cmp[c], CGEphys_cmp[c]',                colormap=cmapCGE, colorrange=(-cmaxA_cmp,  cmaxA_cmp))
        hmc = heatmap!(axCGEcmp[2,c], xc_cmp[c]/1e3, LATS_cmp[c], (CGEsumphys_cmp[c]./scaleC_cmp)',colormap=cmapCGE, colorrange=(-cmaxC_cmps, cmaxC_cmps))
        if c==nblk
            Colorbar(figCGEcmp.scene, hma, bbox=cbb3(panelidx(1,c)), label="ρ₀Π [W/m²]")
            Colorbar(figCGEcmp.scene, hmc, bbox=cbb3(panelidx(2,c)), label=rich("ρ₀ΣΠdx [×10", superscript(string(yC_cmp)), " W/m]"))
        end
        for r in 1:NROW3
            autolimits!(axCGEcmp[r,c])
            xlims!(axCGEcmp[r,c], 0, LdomH/1e3)
            text_fignum!(axCGEcmp[r,c], fignumletter3(r,c); bckclr=:white)
        end
    end

    display(figCGEcmp)
    if figflag==1; savefig300(string(dirfig,"CGE_CGEsum_norm_",savename,".png"), figCGEcmp)
    end
    return figCGEcmp
end


## ============================================================================
## Comparison 1: N² treatment varies, F=25 kW/m held fixed -- 11.27-38
## (Mercator N²(z), varies by run/latitude), 11.40-51 (fixed N² profile from
## 2.5°N), 11.53-64 (fixed N² profile from 50°N)
blocks_cmp = [collect(27:38), collect(40:51), collect(53:64)]
labels_cmp = ["variable N²(z), 11.27-38", "fixed N² 2.5°N, 11.40-51", "fixed N² 50°N, 11.53-64"]

xc_cmp, LATS_cmp, KEtr_cmp, KEhr_cmp, KEtrn_cmp, KEhrn_cmp,
    CGEphys_cmp, CGEsumphys_cmp, CGEnorm_cmp, CGEsumnorm_cmp = load_comparison_data(mainnm, blocks_cmp)

make_KE_compare_figure(xc_cmp, LATS_cmp, KEtr_cmp, KEhr_cmp, KEtrn_cmp, KEhrn_cmp, labels_cmp, "compareN2")
make_CGE_compare_figure(xc_cmp, LATS_cmp, CGEphys_cmp, CGEsumphys_cmp, labels_cmp, "compareN2")


## ============================================================================
## Comparison 2: forcing amplitude F varies, N² treatment held fixed
## (variable Mercator N²(z), i.e. the "zonalmean" N2source) -- 11.1-12
## (F=12.5 kW/m), 11.27-38 (F=25 kW/m), 11.66-77 (F=50 kW/m)
function flux_label(mainnm, rn)
    F = get_runs(mainnm, rn)[1].Flux / 1e3   # kW/m
    fnum = string(mainnm,".",rn[1],"-",rn[end])
    if F == 12.5
        Fstr = "12.5"
    elseif F == 25
        Fstr = "25"
    elseif F == 50
        Fstr = "50"
    else
        Fstr = string(F)
    end
    return string("F=", Fstr, " kW/m, ", fnum)
end

blocksF = [collect(1:12), collect(27:38), collect(66:77)]
labelsF = [flux_label(mainnm, rn) for rn in blocksF]

xc_cmpF, LATS_cmpF, KEtr_cmpF, KEhr_cmpF, KEtrn_cmpF, KEhrn_cmpF,
    CGEphys_cmpF, CGEsumphys_cmpF, CGEnorm_cmpF, CGEsumnorm_cmpF = load_comparison_data(mainnm, blocksF)

for (b, lbl) in enumerate(labelsF)
    println(lbl, ": max(ρ₀ΣΠdx/F) = ", @sprintf("%.3f", maximum(CGEsumnorm_cmpF[b])),
            ", max(ρ₀Π) = ", @sprintf("%.2e", maximum(abs.(CGEphys_cmpF[b]))), " W/m²")
end

make_CGE_compare_figure(xc_cmpF, LATS_cmpF, CGEphys_cmpF, CGEsumphys_cmpF, labelsF, "compareF")


## --- merged KEt/KEh + ρ0Π/F comparison (forcing-amplitude sweep only) --------
## KEt_KEh_norm_compareF.png (4 rows, make_KE_compare_figure) and
## CGE_norm_PiF_check_compareF.png (the standalone ρ0Π/F check, dropped from
## the 3-row CGE figure because F is constant per column there) are merged
## into ONE figure: ρ0Π/F becomes row 5 here, sharing the same 3 forcing-block
## columns as the KE rows above it, so the paper size grows by one row instead
## of the two figures being cross-referenced separately.
##
## This figure switches to ZERO inter-panel gaps (Dsh = Dsv = 0), the
## convention from IW_energy_flux_CGE_2col_ppr.jl -- panels share edges, so
## every Axis below uses inward ticks. The OTHER comparison figures in this
## file (make_KE_compare_figure/make_CGE_compare_figure, Dsh=Dsv=0.02) have
## NOT been converted to this convention yet; that is still to do.
const NROWM   = 5
const PANELHM = 104.0   # pt -- same panel height as every other comparison figure in this file
const MARGBM  = 42.0     # pt: x tick labels + xlabel
const MARGTM  = 21.0     # pt: column titles
fig_hM = NROWM*PANELHM + MARGBM + MARGTM   # paper size grows with the row count; panels don't shrink
posM = subplot_hor_vertpos(3, NROWM, 0.08, 0.15, MARGBM/fig_hM, MARGTM/fig_hM, 0.0, 0.0)
const TICKSIZEM = 4.0
bbM(i)  = BBox(subplot_bbox(posM[i], fig_w2, fig_hM)...)
# hbar<1 + vertoffbar>0 -- small gap top/bottom of EACH row's colorbar, else
# their tick labels collide at the shared row seams now that Dsv=0 stacks the
# colorbars flush too (heatmap panels stay full-height and edge-to-edge; only
# the colorbars get this margin)
cbbM(i) = BBox(colorbar_bbox(posM[i], fig_w2, fig_hM, 0.015, 0.015, 0.92, 0.04)...)
panelidxM(r,c)     = (r-1)*3 + c
fignumletterM(r,c) = string("(", Char('a' + (c-1)*NROWM + (r-1)), ")")   # a-o, down each column

function make_KE_CGE_merged_figure(xc_cmp, LATS_cmp, KEtr_cmp, KEhr_cmp, KEtrn_cmp, KEhrn_cmp,
                                    PiFnorm_cmp, labels, savename)
    nblk = length(labels)
    cmaxKEt = maximum(maximum(m) for m in KEtr_cmp) * fcKE
    cmaxKEh = maximum(maximum(m) for m in KEhr_cmp) * fcKE
    cmaxPiF  = maximum(maximum(abs.(m)) for m in PiFnorm_cmp)
    yPiF = pow10_exponent(cmaxPiF); scalePiF = 10.0^yPiF; cmaxPiFs = cmaxPiF/scalePiF

    figM = Figure(size=(fig_w2, fig_hM), fontsize=10)
    axM = Matrix{Axis}(undef, NROWM, 3)
    for r in 1:NROWM, c in 1:3
        axM[r,c] = Axis(figM.scene, bbox=bbM(panelidxM(r,c)),
            title = r==1 ? labels[c] : "",
            xtickalign = 1, ytickalign = 1, xticksize = TICKSIZEM, yticksize = TICKSIZEM,
            # panels share edges (Dsh=0): "2000" is blanked on every column but
            # the last so it can't collide with the next column's "0"
            xticks = (0:500:2000, c==nblk ? string.(0:500:2000) : vcat(string.(0:500:1500), "")),
            yticks = 0:10:40,
            xlabel = r==NROWM ? "x [km]" : "",
            ylabel = c==1 ? "latitude [°]" : "",
            xticklabelsvisible = r==NROWM, yticklabelsvisible = c==1)
    end

    for c in 1:nblk
        hma = heatmap!(axM[1,c], xc_cmp[c]/1e3, LATS_cmp[c], (KEtr_cmp[c]*fcKE)',  colormap=cmapKE,   colorrange=(0,cmaxKEt))
        hmb = heatmap!(axM[2,c], xc_cmp[c]/1e3, LATS_cmp[c], KEtrn_cmp[c]',        colormap=cmapNorm, colorrange=(0,1))
        hmc = heatmap!(axM[3,c], xc_cmp[c]/1e3, LATS_cmp[c], (KEhr_cmp[c]*fcKE)',  colormap=cmapKE,   colorrange=(0,cmaxKEh))
        hmd = heatmap!(axM[4,c], xc_cmp[c]/1e3, LATS_cmp[c], KEhrn_cmp[c]',        colormap=cmapNorm, colorrange=(0,1))
        hme = heatmap!(axM[5,c], xc_cmp[c]/1e3, LATS_cmp[c], (PiFnorm_cmp[c]./scalePiF)',
                        colormap=cmapCGE, colorrange=(-cmaxPiFs, cmaxPiFs))
        if c==nblk
            Colorbar(figM.scene, hma, bbox=cbbM(panelidxM(1,c)), label="KEt [kJ/m²]")
            Colorbar(figM.scene, hmb, bbox=cbbM(panelidxM(2,c)), label="KEt / KEtmax")
            Colorbar(figM.scene, hmc, bbox=cbbM(panelidxM(3,c)), label="KEh [kJ/m²]")
            Colorbar(figM.scene, hmd, bbox=cbbM(panelidxM(4,c)), label="KEh / KEtmax")
            Colorbar(figM.scene, hme, bbox=cbbM(panelidxM(5,c)),
                label=rich("ρ₀Π/F [×10", superscript(string(yPiF)), " 1/m]"))
        end
        for r in 1:NROWM
            autolimits!(axM[r,c])
            xlims!(axM[r,c], 0, LdomH/1e3)
            text_fignum!(axM[r,c], fignumletterM(r,c); bckclr=:white)
        end
    end

    display(figM)
    if figflag==1; savefig300(string(dirfig,"KEt_KEh_norm_",savename,".png"), figM)
    end
    return figM
end

make_KE_CGE_merged_figure(xc_cmpF, LATS_cmpF, KEtr_cmpF, KEhr_cmpF, KEtrn_cmpF, KEhrn_cmpF,
                           CGEnorm_cmpF, labelsF, "compareF")


## --- max ρ₀Π within x<=1000km & lat<=25°, vs forcing amplitude F -------------
Fvals = [get_runs(mainnm, rn)[1].Flux/1e3 for rn in blocksF]   # kW/m
maxPi_vsF = zeros(length(blocksF))
for (b, rn) in enumerate(blocksF)
    Ilat = findall(LATS_cmpF[b] .<= 25)
    Ix   = findall(xc_cmpF[b]/1e3 .<= 1000)
    maxPi_vsF[b] = maximum(CGEphys_cmpF[b][Ilat, Ix])
end
println("max ρ₀Π (x≤1000km, lat≤25°) vs F [kW/m]: ", collect(zip(Fvals, maxPi_vsF)))

figFscale = Figure(size=(10*cm_to_pt, 8*cm_to_pt), fontsize=10)
axFscale = Axis(figFscale[1,1], xlabel="F [kW/m]", ylabel="max ρ₀Π [W/m²]\n(x≤1000 km, lat≤25°)")
scatterlines!(axFscale, Fvals, maxPi_vsF, color=:black, marker=:circle, markersize=10, linewidth=2, label="data")

# F^1.7 reference (the superlinear scaling estimated from the 3 aggregate
# points above), anchored to pass through the F=25 kW/m data point
Fline   = range(minimum(Fvals), maximum(Fvals), length=50)
refline = maxPi_vsF[2] .* (Fline./25).^1.7
lines!(axFscale, Fline, refline, color=:gray, linestyle=:dash, linewidth=2, label="F^1.7")
axislegend(axFscale, position=:lt, framevisible=false)

display(figFscale)

if figflag==1; savefig300(string(dirfig,"CGE_maxPi_vs_F.png"), figFscale)
end


## --- same, but one point per (latitude, F) pair instead of a single max
## across all latitudes ≤25° -- more data points, and shows whether the
## superlinear F-scaling seen above is uniform across latitude or not --------
LATref = LATS_cmpF[1]                 # same latitudes for all 3 F-blocks
IlatS  = findall(LATref .<= 25)
nlat   = length(IlatS)

maxPi_latF = zeros(nlat, length(blocksF))   # [latitude, F-block]
for (b, rn) in enumerate(blocksF)
    Ix = findall(xc_cmpF[b]/1e3 .<= 1000)
    for (li, i) in enumerate(IlatS)
        maxPi_latF[li, b] = maximum(CGEphys_cmpF[b][i, Ix])
    end
end

colorsLat = cgrad(:darktest, nlat, categorical = true)

figFscale2 = Figure(size=(12*cm_to_pt, 9*cm_to_pt), fontsize=10)
axFscale2 = Axis(figFscale2[1,1], xlabel="F [kW/m]", ylabel="max ρ₀Π [W/m²] (x≤1000 km)")
for (li, i) in enumerate(IlatS)
    scatterlines!(axFscale2, Fvals, maxPi_latF[li,:], color=colorsLat[li], marker=:circle, markersize=8, linewidth=1.5)
end
# same F^1.7 reference as the aggregate plot, anchored to the all-latitude max at F=25 kW/m
lines!(axFscale2, Fline, refline, color=:black, linestyle=:dash, linewidth=2, label="F^1.7")
axislegend(axFscale2, position=:lt, framevisible=false)
Colorbar(figFscale2[1,2], colormap=colorsLat, limits=(0.5, nlat+0.5),
    ticks=(1:nlat, string.(LATref[IlatS])), label="latitude [°]")

display(figFscale2)

if figflag==1; savefig300(string(dirfig,"CGE_maxPi_vs_F_bylat.png"), figFscale2)
end
