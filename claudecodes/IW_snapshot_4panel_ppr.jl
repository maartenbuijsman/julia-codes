#= IW_snapshot_4panel_ppr.jl
Maarten Buijsman, USM DMS, 2026-10-2

Paper figure: zonal velocity with superposed isopycnals at day 20, as a 4x1
stack, 18 x 18 cm, fontsize 10, ONE shared colorbar. Companion to
IW_komega_2x2_ppr.jl -- the SAME four runs, in the same order, so the (k,omega)
spectra and the physical-space transects can be read side by side.

  (a) 11.63   200 m NH,   40°N, 25 kW m⁻¹, tide only, fixed N²(50°N)
  (b) 11.34   200 m NH, 28.8°N, 25 kW m⁻¹, tide only
  (c) GMSER.34  200 m NH, 28.8°N, 25 kW m⁻¹, GM + tide (GMSER = 16: GM81 IC at 1x GM81
                over days 10-20; 15: the earlier ~3x GM IC, output names without a tag)
  (d) 11.67   200 m NH,  2.5°N, 50 kW m⁻¹, tide only

Latitude decreases down the stack. (b) and (c) are the same latitude and
forcing without and with the GM background, so they sit adjacent. The k-omega
companion lays the same four runs out clockwise in a 2x2 so that pair ends up
stacked in its right-hand column; the panel letters mean the same run in both
figures.

Method follows IW_single_snapshot.jl:
  - u is read on x faces and averaged to cell centres
  - the reference density comes from the run's OWN N2 forcing profile
    (n2_filename), integrated as breff = cumtrapz(zfw, N2w) and converted with
    rho = -b*rho0/g; the plotted field is rho0 + rho_pert + rho_ref
  - contours are drawn at the reference densities of a fixed set of DEPTHS
    (LEVEL_DEPTHS), so every panel's contours start at the same depths even
    though panel (c) uses a different stratification and therefore different
    density values

One colorbar means one colour range for all four panels. (d) carries 4x the
forcing flux of the others and saturates first; CLIMS is the knob.

First pass covers x = 0-750 km, the near field where the beam structure is
still coherent. XWIN is the knob for that.
=#

println("number of threads is ", Threads.nthreads())

using NCDatasets, Printf, CairoMakie, Statistics, JLD2, Interpolations

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirsim    = string(pth0, "IW/")
dirfig    = string(pth0, "figs/")
dirforce  = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

figflag = 1
const LBL2 = false       # true adds the configuration/forcing line to the label

const RHO0 = 1020.0
const GRAV = 9.81

# display window / scales
const XWIN  = (0.0, 800.0)          # km
const ZLIM  = (-1500.0, 0.0)        # m
const CLIMS = (-0.3, 0.3)           # u [m/s]
const CMAP  = Reverse(:RdBu_9)
const LEVEL_DEPTHS = [100, 300, 500, 750, 1000, 1250]   # m

# white backing box for the in-panel label, as a fraction of the panel's range.
# Bottom-left: the deep water at small x is the quietest corner of every panel,
# so the box hides least there -- top-left sat on the 100 and 300 m isopycnals.
const BOXX0          = 0.012
const BOXY0, BOXY1   = 0.050, 0.210
const BOXPAD, BOXCHR = 0.025, 0.0092
const BOXFAC         = 1.10     # widen the box to the right by this factor

# (mainnm, runnm, panel letter, short description, stratification note)
const GMSER = 16   # GM series of panel (c): 16 (GM81 IC, 1x GM81 over days 10-20) or 15
RUNS = [(11, 63, "(a)", "tide only", "N²(50°N)"),
        (11, 34, "(b)", "tide only", ""),
        (GMSER, 34, "(c)", "GM + tide", ""),
        (11, 67, "(d)", "tide only", "")]

## --- load ---------------------------------------------------------------------
function snapshot(mainnm, runnm)
    row    = get_runs(mainnm, [runnm])[1]
    fnames = @sprintf("AMZexpt%02i.%02i", mainnm, runnm)
    ds = NCDataset(string(dirsim, fnames, ".nc"), "r")

    tday = ds["time"][:] ./ 86400
    It   = length(tday)                       # last output = day 20
    xc   = ds["x_caa"][:]
    zc   = ds["z_aac"][:]

    # read only the displayed x range. u lives on faces, so the face slice runs
    # one index past the last centre it has to average into.
    Ic = findall(XWIN[1]*1e3 .<= xc .<= XWIN[2]*1e3)
    uf = ds["u"][Ic[1]:Ic[end]+1, :, It]
    bc = ds["b"][Ic, :, It]
    close(ds)

    uc = 0.5 .* (uf[1:end-1, :] .+ uf[2:end, :])   # faces -> centres

    # reference density from this run's own N2 profile, then total density
    @load string(dirforce, n2_filename(row)) N2w zfw
    breff   = cumtrapz(zfw, N2w)
    intzc   = interpolate((zfw,), breff, Gridded(Linear()))
    rhorefc = -intzc.(zc) .* RHO0/GRAV
    rhoc    = (-bc .* RHO0/GRAV) .+ reshape(rhorefc, 1, :) .+ RHO0

    # contour levels = the reference density AT the fixed depths, so the
    # contours label the same depths in every panel
    levels = sort([-intzc(-d)*RHO0/GRAV + RHO0 for d in LEVEL_DEPTHS])

    @printf("%s  lat %.1f  F %.1f kW/m  day %.2f  |u|max %.3f m/s\n",
            fnames, row.lat, row.Flux/1e3, tday[It], maximum(abs, uc))
    return (; mainnm, runnm, lat=row.lat, flux=row.Flux, tsnap=tday[It],
              x=xc[Ic], z=zc, uc, rhoc, levels, umax=maximum(abs, uc))
end

PAN = [snapshot(r[1], r[2]) for r in RUNS]

## --- figure: 18 x 18 cm, fontsize 10 -------------------------------------------
cm_to_pt = 72/2.54
fig = Figure(size=(18*cm_to_pt, 18*cm_to_pt), fontsize=10)

# 0 is ticked; ZLIM[1] is not. With rowgap = 0 a tick at the very bottom of one
# panel would print on top of the 0 at the top of the panel below, but -1200 is
# far enough up the axis to clear it.
ZT = ([0, -400, -800, -1200], ["0", "-400", "-800", "-1200"])

hms = Any[]
for (n, (p, r)) in enumerate(zip(PAN, RUNS))
    ax = Axis(fig[n, 1], ylabel="z [m]", yticks=ZT, xticks=XWIN[1]:100:XWIN[2],
        xlabel = n == 4 ? "x [km]" : "",
        xticklabelsvisible = n == 4)
    push!(hms, heatmap!(ax, p.x./1e3, p.z, p.uc, colormap=CMAP, colorrange=CLIMS))
    contour!(ax, p.x./1e3, p.z, p.rhoc; levels=p.levels, color=:black,
        linewidth=0.6, labels=false)
    xlims!(ax, XWIN...); ylims!(ax, ZLIM...)

    # same label block as IW_komega_2x2_ppr.jl: `text!` has no background
    # attribute, so the white backing is an explicit poly in data coordinates.
    # The limits above must be frozen first or the poly would push them out.
    # one line only: the run id already carries the configuration (mainnm 11 =
    # tide only, 15 = GM + tide) and the full description is in the caption and
    # in the companion k-omega figure. Set LBL2 = true to put it back.
    l1 = @sprintf("%s %d.%d   %.1f°N", r[3], p.mainnm, p.runnm, p.lat)
    l2 = @sprintf("%s, %g kW m⁻¹", r[4], p.flux/1e3) *
         (isempty(r[5]) ? "" : ", " * r[5])
    lbl = LBL2 ? string(l1, "\n", l2) : l1
    bw  = BOXFAC * (BOXPAD + BOXCHR*(LBL2 ? maximum(length.((l1, l2))) : length(l1)))
    bh  = (BOXY1-BOXY0) * (LBL2 ? 2.0 : 1.0)
    dx, dz = XWIN[2]-XWIN[1], ZLIM[2]-ZLIM[1]
    poly!(ax, Rect2f(XWIN[1] + BOXX0*dx, ZLIM[1] + BOXY0*dz,
                     bw*dx, bh*dz), color=:white, strokewidth=0)
    text!(ax, XWIN[1] + (BOXX0+0.008)*dx, ZLIM[1] + (BOXY0+0.045)*dz,
        align=(:left,:bottom), fontsize=9, color=:black, font=:bold, text=lbl)
end

Colorbar(fig[1:4, 2], hms[1], label="u [m s⁻¹]", labelsize=10,
    ticklabelsize=9, width=10)

# no in-figure footnote: the isopycnal depths and the snapshot time belong in
# the paper caption, which the printout below restates for copying
rowgap!(fig.layout, 0)            # panels flush, shared x axis
colgap!(fig.layout, 6)

display(fig)
if figflag == 1
    fout = string(dirfig, @sprintf("snapshot_u_rho_4panel_ppr_%03d-%03dkm%s.png",
                                   XWIN[1], XWIN[2], GMSER == 15 ? "" : "_GM$(GMSER)"))
    savefig300(fout, fig)
    println("\nsaved ", fout)
end

## --- numbers --------------------------------------------------------------------
println("\n", "="^70)
println(rpad("panel",7), rpad("run",9), rpad("lat",7), rpad("F [kW/m]",10),
        rpad("day",7), "max|u| [m/s]")
println("="^70)
for (p, r) in zip(PAN, RUNS)
    println(rpad(r[3],7), rpad(@sprintf("%d.%d",p.mainnm,p.runnm),9),
        rpad(@sprintf("%.1f",p.lat),7), rpad(@sprintf("%.1f",p.flux/1e3),10),
        rpad(@sprintf("%.2f",p.tsnap),7), @sprintf("%.3f", p.umax))
end
