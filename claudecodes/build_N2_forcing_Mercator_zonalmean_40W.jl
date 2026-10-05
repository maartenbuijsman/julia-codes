# build_N2_forcing_Mercator_zonalmean_40W.jl
# MCB/Claude, USM, 2026-9-26
#
# Final N2 forcing profiles for IW_Amz_200m_2000km_bash_cuda.jl, based ONLY
# on Mercator data (dropping WOCE -- Mercator's open-Atlantic zonal mean
# looked cleaner/less noisy and only differed from WOCE in a few spots).
#
# Source: MERC_N2_zonalmean_Atl_offshelf_monthly.nc (the refined Range-2,
# coastline-anchored/off-shelf, monthly-then-annual-then-zonal-mean product
# from extract_TS_N2_Atlantic_zonalmean_offshelf_monthly.jl), superseding the
# earlier fixed-60W-20W-band MERC_N2_zonalmean_Atl_40W.nc.
#
# NOTE: these are NOT single-column 40 W profiles -- they are zonally
# averaged over the open Atlantic (per-latitude adaptive, off-shelf window),
# at the same 14 latitudes originally chosen for the 40 W transect. Output
# files/plots are named accordingly ("ZonalMeanAtl"/"zonalmean", not "40W").
#
# Recipe:
#  - Each latitude's native (~42-48 point) zonal-mean N2 profile is first
#    PCHIP-splined (shape-preserving -- no overshoot/ringing near sharp
#    gradients, unlike a plain cubic spline) onto a uniform dz=1 m grid, to
#    remove the "angular" piecewise-linear look below ~100 m that comes from
#    computing N2 as a derivative across Mercator's increasingly coarse native
#    spacing at depth. (Splining N2 itself, not T/S: this is a zonal MEAN of
#    many independently-computed per-column N2 profiles, so there is no
#    single T,S profile behind it to spline instead.) Outside each latitude's
#    own native valid range, the profile is extended flat (nearest-neighbor)
#    rather than let the spline extrapolate.
#  - From that fine, smooth profile, N2 is then sampled via LINEAR
#    INTERPOLATION onto the coarser WKB grid. (Earlier versions used
#    nearest-neighbor here, justified by Mercator's very fine native
#    near-surface spacing -- but since every latitude is now routed through
#    the dense, smooth 1 m spline first, linear interpolation from that fine
#    grid is effectively just as accurate and is the more standard choice.)
#    Extrapolation is never actually triggered: spline_then_extend already
#    extends every latitude's fine profile flat out to mindepth=-4000 m using
#    its own deepest valid value where that latitude's zonal-mean data runs
#    out shallower (e.g. 55N/60N), so zfw always falls within the fine grid's
#    domain.
#  - The WKB-scaled face grid (zfw) is computed ONCE from the 2.5 N profile,
#    using two explicit knobs (rather than letting the grid size/near-surface
#    dz emerge automatically): `nzWKB` sets the number of WKB-space layers
#    (dzwkb2 = H/nzWKB), and `dzminfix` caps the minimum near-surface layer
#    thickness -- the near-surface fix now finds the deepest point where the
#    raw WKB-derived dz first drops below dzminfix and replaces everything
#    shallower with a uniform dzminfix grid, rather than replicating whatever
#    the automatic minimum dz happened to be (which could eat into the
#    layer budget and leave the bottom layer too thick). Every other
#    latitude's N2 is then linearly interpolated onto that SAME zfw -- so
#    every latitude ends up with identical Nz, directly comparable.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using Interpolations
using DataInterpolations
using Trapz
using JLD2
using CairoMakie
using ColorSchemes
using Printf

# ---------------------------------------------------------------------------
# paths / parameters

dirmerc = "/home/mbui/ModelOutput/IW/mercator/"
dirout  = "/home/mbui/ModelOutput/IW/forcingfiles/"
dirfig  = "/home/mbui/ModelOutput/figs/"

mindepth = -4000.0   # m, bottom of the model domain
nzWKB    = 110         # number of WKB-space layers (dzwkb2 = H/nzWKB)
dzminfix = 5.0         # m, capped minimum near-surface layer thickness
lat0     = 2.5         # reference latitude whose WKB grid all others share

# ---------------------------------------------------------------------------
# load the Mercator open-Atlantic zonal-mean N2 (Range-2/off-shelf/monthly)

ds = NCDataset(string(dirmerc,"MERC_N2_zonalmean_Atl_offshelf_monthly.nc"),"r")
depth_mid = ds["depth_mid"][:]
lat       = ds["latitude"][:]
N2_all    = coalesce.(ds["N2_zonalmean"][:,:], NaN)   # (depth_mid, latitude)
close(ds)

nlat = length(lat)
k0   = findfirst(==(lat0), lat)

# ---------------------------------------------------------------------------
# linear-interpolation lookup, Flat() beyond the source range as a safety net
# (never actually triggered -- see note above)

function interp_lookup(z_native::Vector{Float64}, N2_native::Vector{Float64})
    return linear_interpolation(z_native, N2_native, extrapolation_bc=Flat())
end

# ---------------------------------------------------------------------------
# WKB-regrid helper (AMZ_stratification_profile.jl recipe, generalized with
# explicit nzWKB/dzminfix knobs instead of a nominal dzwkb2 + automatic
# near-surface-dz fix)

function wkb_regrid(zf::Vector{Float64}, N2::Vector{Float64}, nzWKB::Int, dzminfix::Float64)
    Nzf = length(zf)
    H = abs(zf[1])
    Nave = trapz(zf, sqrt.(N2)) / H

    zwkb = zeros(size(zf))
    for i in Nzf-1:-1:1
        zwkb[i] = trapz(zf[Nzf:-1:i], sqrt.(N2[Nzf:-1:i])) / Nave
    end
    for i in 2:Nzf
        zwkb[i] <= zwkb[i-1] && (zwkb[i] = nextfloat(zwkb[i-1]))
    end

    zwkb2 = collect(range(-H, 0, length=nzWKB+1))
    interp_linextr = linear_interpolation(zwkb, zf, extrapolation_bc=Line())
    zfd = interp_linextr.(zwkb2)

    # near-surface fix: cap the minimum layer thickness at dzminfix, rather
    # than replicating whatever the automatic minimum dz happened to be
    dzd = diff(zfd)
    dzmin, Imin = findmin(dzd)
    Iminfix = findlast(>(dzminfix), dzd[1:Imin])
    isnothing(Iminfix) && error("no point in the WKB grid has dz > dzminfix=", dzminfix, " -- lower dzminfix or increase nzWKB")
    len = 1 + Int(ceil(abs(zfd[Iminfix])/dzminfix))
    zfdadd = collect(range(zfd[Iminfix], 0, len))
    zfw = vcat(zfd[1:Iminfix], zfdadd[2:end])

    itp = interp_lookup(zf, N2)
    N2w = itp.(zfw)

    return zfw, N2w
end

# ---------------------------------------------------------------------------
# per-latitude native (valid-trimmed) profile, ascending z

zM = Vector{Vector{Float64}}(undef, nlat)
NM = Vector{Vector{Float64}}(undef, nlat)

for k in 1:nlat
    ivalid = findall(!isnan, N2_all[:,k])
    zM[k] = reverse(-depth_mid[ivalid])
    NM[k] = reverse(N2_all[ivalid,k])
    println("lat=",lat[k]," native valid range: ", zM[k][1], " to ", zM[k][end], " m (", length(zM[k]), " points)")
end

# ---------------------------------------------------------------------------
# PCHIP-spline each latitude's native profile onto a uniform 1 m grid over
# [mindepth,0]; outside that latitude's own native valid range, extend flat
# (nearest-neighbor) rather than let the spline extrapolate

dzspline = 1.0
zfine = collect(mindepth:dzspline:0.0)

function spline_then_extend(z_native::Vector{Float64}, N2_native::Vector{Float64}, zfine::Vector{Float64})
    itp = PCHIPInterpolation(N2_native, z_native)
    zmin, zmax = z_native[1], z_native[end]
    N2fine = similar(zfine)
    for (i,zi) in enumerate(zfine)
        N2fine[i] = zi <= zmin ? N2_native[1] :
                    zi >= zmax ? N2_native[end] :
                    itp(zi)
    end
    return N2fine
end

zMs = Vector{Vector{Float64}}(undef, nlat)   # all == zfine, kept per-latitude for uniform interface below
NMs = Vector{Vector{Float64}}(undef, nlat)

for k in 1:nlat
    zMs[k] = zfine
    NMs[k] = spline_then_extend(zM[k], NM[k], zfine)
    nneg = count(<(0), NMs[k])
    nneg > 0 && println("  lat=",lat[k]," WARNING: ", nneg, " negative points after spline (unexpected for PCHIP)")
end

# ---------------------------------------------------------------------------
# build the 2.5 N reference WKB grid, then linearly interpolate every
# latitude's fine spline profile onto it

zfw_ref, N2w_ref = wkb_regrid(zMs[k0], NMs[k0], nzWKB, dzminfix)
Nz_ref = length(zfw_ref)-1
dz_ref = diff(zfw_ref)
println("reference (", lat[k0], " N) WKB grid: Nz = ", Nz_ref, " layers, ", length(zfw_ref), " faces, ",
        "min/max thickness = ", round(minimum(dz_ref),digits=2), "/", round(maximum(dz_ref),digits=2), " m")

isdir(dirout) || error("output dir missing: ", dirout)

for k in 1:nlat
    N2w = k == k0 ? N2w_ref : interp_lookup(zMs[k], NMs[k]).(zfw_ref)

    fname = @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", lat[k])
    path_fname = string(dirout, fname)
    jldsave(path_fname; N2w, zfw=zfw_ref, lonsel=-40.0, latsel=lat[k])
    println(fname, " saved (Nz=", length(zfw_ref)-1, ") ........ ")
end

# ---------------------------------------------------------------------------
# diagnostic plot: 3 latitude-group subplots, full depth + upper 400 m

# 55N/60N dropped from these two diagnostic figures (still fully processed and
# saved above for every latitude incl. 55/60 -- this only trims what's PLOTTED
# here), leaving 13 latitudes split 4/5/4 across the 3 panels (15N moved into
# panel 2).
groups = [1:4, 5:9, 10:13]
grouptitles = ["0-10°N", "15-30°N", "35-50°N"]

# per-subplot x-axis ranges matching N2_zonalmean_Atl_offshelf_monthly_3panels(_upper400m).png
# (plot_N2_zonalmean_Atl_offshelf_monthly.jl let each panel auto-scale to its
# own group's data, restricted to that panel's own depth range) -- recomputed
# here directly from that same source file/grouping/depth-restriction so the
# two figures are directly, panel-by-panel comparable
dsref = NCDataset(string(dirmerc,"MERC_N2_zonalmean_Atl_offshelf_monthly.nc"),"r")
depth_mid_ref = dsref["depth_mid"][:]
N2_ref        = coalesce.(dsref["N2_zonalmean"][:,:], NaN)
close(dsref)

function group_xmax(depth_mid_ref, N2_ref, rng, maxdepth)
    idx = findall(depth_mid_ref .<= maxdepth)
    1.05*maximum(filter(!isnan, N2_ref[idx,rng]))
end

# full-depth version: unchanged plain GridLayout figure with a shared title
for (ylim, suffix, ttl) in [((-4000,0), "", "full depth")]
    fig = Figure(size = (1500, 800))
    for (g,rng) in enumerate(groups)
        ax = Axis(fig[1,g], xlabel = "N² [s⁻²]", ylabel = "Depth [m]", title = grouptitles[g])
        ng = length(rng)
        cols = get(ColorSchemes.viridis, range(0,1,length=ng))
        for (i,k) in enumerate(rng)
            N2w = k == k0 ? N2w_ref : interp_lookup(zMs[k], NMs[k]).(zfw_ref)
            lines!(ax, N2w, zfw_ref, color = cols[i], label = string(lat[k],"°N"))
        end
        #xlims!(ax, 0, group_xmax(depth_mid_ref, N2_ref, rng, -ylim[1]))
        xlims!(ax,[0 0.00052])
        ylims!(ax, ylim...)
        axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
    end
    Label(fig[0,1:3], "Mercator-only N² forcing profiles (open-Atlantic zonal mean, 60°W-20°W) at 40°W-transect latitudes ($ttl), shared Nz=$(length(zfw_ref)-1) grid", fontsize = 14)
    save(string(dirfig,"N2_forcing_Mercator_zonalmean_3panels$suffix.png"), fig)
    println("saved: ", string(dirfig,"N2_forcing_Mercator_zonalmean_3panels$suffix.png"))    
    #display(fig)
end

# ---------------------------------------------------------------------------
# upper-400m version: paper-ready 3-panel figure, restyled to match the
# comparison figures in IW_analysis_energy_2000km_ppr.jl -- explicit bbox
# layout (subplot_hor_vertpos), Dsh=0 (panels share their vertical edges, so
# ticks go inward), no shared figure Label (removed entirely -- the panel
# titles below carry the group info instead), and the panel letter folded
# directly into each axis's own title rather than a separate corner box.
cm_to_pt = 72/2.54
fig_wN = 18*cm_to_pt
fig_hN = 12*cm_to_pt          # single row, so Dsv is irrelevant (numpv=1)
const MARGBN = 42.0           # pt: x tick labels + xlabel
const MARGTN = 21.0           # pt: panel title (folds in the (a)/(b)/(c) letter)
posN = subplot_hor_vertpos(3, 1, 0.09, 0.02, MARGBN/fig_hN, MARGTN/fig_hN, 0.0, 0.0)
bbN(i) = BBox(subplot_bbox(posN[i], fig_wN, fig_hN)...)
const TICKSIZEN = 4.0

# fixed x-range (0, 0.00052 s⁻²) is identical across all 3 panels (see xlims!
# below), so with Dsh=0 an outermost tick would sit exactly on the shared
# column edge -- these ticks are kept well clear of both edges instead
# (same "inside ticks" convention as the row-1/row-4 forced ticks in
# IW_energy_flux_CGE_2col_ppr.jl), so no tick from one panel can be mistaken
# for its neighbour's.
# Grid (tick marks/gridlines) runs 0:1e-4:5e-4, unchanged across panels.
# Labels are the bare integer multiple of 1e-4 (0,1,2,3,4[,5]) rather than the
# full decimal -- the xlabel below carries the "×10⁻⁴" scale instead, so
# narrow single-digit labels are all that's needed and every tick from 0-4
# fits with no crowding. Panel (c) also labels the last tick (5, i.e. 0.0005):
# its curves sit closer to the left edge than (a)/(b)'s, so there's no
# crowding risk there and its own max is worth reading directly.
xtickvalsN = collect(0.0:1e-4:5e-4)
xticklabelsN(g) = [(idx = round(Int, v/1e-4); (idx<=4 || g==3) ? string(idx) : "")
                    for v in xtickvalsN]

figN = Figure(size = (fig_wN, fig_hN), fontsize = 10)
letters = ("(a)", "(b)", "(c)")
for (g,rng) in enumerate(groups)
    ax = Axis(figN.scene, bbox = bbN(g),
        xtickalign = 1, ytickalign = 1, xticksize = TICKSIZEN, yticksize = TICKSIZEN,
        xticks = (xtickvalsN, xticklabelsN(g)),
        xlabel = "N² [×10⁻⁴ s⁻²]", ylabel = "Depth [m]",
        ylabelvisible = g==1, yticklabelsvisible = g==1,
        title = string(letters[g], " ", grouptitles[g]))
    ng = length(rng)
    cols = get(ColorSchemes.viridis, range(0,1,length=ng))
    for (i,k) in enumerate(rng)
        N2w = k == k0 ? N2w_ref : interp_lookup(zMs[k], NMs[k]).(zfw_ref)
        lines!(ax, N2w, zfw_ref, color = cols[i], label = string(lat[k],"°N"))
    end
    xlims!(ax, [0 0.00052])
    ylims!(ax, -400, 0)
    axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
end
savefig300(string(dirfig,"N2_forcing_Mercator_zonalmean_3panels_upper400m.png"), figN)
println("saved: ", string(dirfig,"N2_forcing_Mercator_zonalmean_3panels_upper400m.png"))
