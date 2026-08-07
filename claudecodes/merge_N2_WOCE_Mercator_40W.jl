# merge_N2_WOCE_Mercator_40W.jl
# MCB/Claude, USM, 2026-8-4 (v2: monthly-averaged N2, shared vertical grid)
#
# Merge WOCE N2 (accurate near-surface/thermocline structure, but only valid to
# ~1900 m at most latitudes) with Mercator N2 (extends to full depth) along
# 40 W, blend across a narrow transition band, and save one N2w/zfw .jld2 file
# per latitude -- drop-in compatible with the `@load path_fname N2w zfw` used
# in IW_Amz_200m_2000km_bash_cuda.jl.
#
# v2 changes from the first iteration:
#  - N2 sources are now the annual mean of the 12 *monthly* N2 profiles
#    (MERC_N2_monthly_40W.nc / WOCE_N2_monthly_40W.nc), not N2 computed from
#    annual-mean T,S -- averaging N2 itself is more representative of the
#    real (nonlinear) stratification. Both files already share the exact same
#    latitude grid, so no re-extraction/interpolation across latitude is
#    needed here (unlike v1, which had to reconcile WOCE's native 2 deg grid
#    against Mercator's arbitrary-latitude grid).
#  - WOCE N2<0 is clamped to zeroval = 1e-12 before blending.
#  - Boundary conditions on the merged profile: N2 = zeroval exactly at the
#    surface (z=0), and N2 = nearest-neighbor Mercator value at the bottom
#    (z=mindepth=-4000 m), matching AMZ_stratification_profile.jl's original
#    convention (zz=[0; zmid; mindepth], N2c=[0; N2b; N2b[end]]) rather than
#    linearly interpolating/extrapolating those two endpoints.
#  - All 14 latitudes are regridded onto ONE shared set of WKB-scaled faces
#    (zfw), computed once from the 0 N profile only (dz kept constant above
#    the depth of maximum stratification, as in the original recipe) -- so
#    every latitude ends up with the same Nz, letting them be compared
#    directly. Other latitudes' N2 is then just linearly interpolated onto
#    that fixed zfw, not independently WKB-scaled.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using Interpolations
using Trapz
using JLD2
using CairoMakie
using ColorSchemes
using Printf

# ---------------------------------------------------------------------------
# paths / parameters

dirwoce = "/home/mbui/ModelOutput/IW/stratification/"
dirmerc = "/home/mbui/ModelOutput/IW/mercator/"
dirout  = "/home/mbui/ModelOutput/IW/forcingfiles/"
dirfig  = "/home/mbui/ModelOutput/figs/"

blend_end_default = 2100.0    # m
blend_width        = 400.0    # m
mindepth            = -4000.0 # m, bottom of the merged/model domain
dzwkb2              = 40.0    # m, nominal WKB-space spacing (matches N2_amz1.jld2 recipe)
zeroval             = 1e-12
lat0                = 0.0     # reference latitude whose WKB grid all others share

# ---------------------------------------------------------------------------
# load the monthly-averaged N2 products (same latitude grid in both files)

dsw = NCDataset(string(dirwoce,"WOCE_N2_monthly_40W.nc"),"r")
depth_mid_w = dsw["depth_mid"][:]
lat         = dsw["latitude"][:]
N2_w_all    = coalesce.(dsw["N2_annual_mean"][:,:], NaN)   # (depth_mid, latitude)
close(dsw)

dsm = NCDataset(string(dirmerc,"MERC_N2_monthly_40W.nc"),"r")
depth_mid_m = dsm["depth_mid"][:]
lat_m       = dsm["latitude"][:]
N2_m_all    = coalesce.(dsm["N2_annual_mean"][:,:], NaN)   # (depth_mid, latitude)
lon_used    = dsm.attrib["longitude"]
close(dsm)

@assert lat == lat_m "WOCE and Mercator monthly products must share the same latitude grid"
nlat = length(lat)
k0   = findfirst(==(lat0), lat)

# clamp WOCE N2<0 to zeroval before blending
nneg_w = count(x -> !isnan(x) && x < 0, N2_w_all)
println("clamping ", nneg_w, " negative WOCE monthly-mean N2 point(s) to zeroval = ", zeroval)
N2_w_all[.!isnan.(N2_w_all) .& (N2_w_all .< 0)] .= zeroval

# Below each latitude's own deepest valid Mercator point, N2 is held flat
# (nearest-neighbor) via itp_m's Flat() extrapolation below -- several
# latitudes (e.g. 45N/60N) run out as shallow as ~2083 m, well short of
# mindepth=-4000, since the Mercator land/bathymetry mask is identical
# between the monthly and annual-mean-T,S products (confirmed: same cutoff
# depth in both). There is no deeper real Mercator data to fall back on.

# ---------------------------------------------------------------------------
# WKB-regrid helper (identical recipe to AMZ_stratification_profile.jl),
# hardened against long flat/duplicate-valued stretches in N2 (which make the
# WKB coordinate zwkb non-strictly-increasing -- expected here given the flat
# abyssal extrapolation) by nudging ties up to the next representable float
# rather than erroring.

function make_strictly_increasing!(x::Vector{Float64})
    for i in 2:length(x)
        if x[i] <= x[i-1]
            x[i] = nextfloat(x[i-1])
        end
    end
    return x
end

function wkb_regrid(zf::Vector{Float64}, N2::Vector{Float64}, dzwkb2::Float64)
    Nzf = length(zf)
    H = abs(zf[1])
    Nave = trapz(zf, sqrt.(N2)) / H

    zwkb = zeros(size(zf))
    for i in Nzf-1:-1:1
        zwkb[i] = trapz(zf[Nzf:-1:i], sqrt.(N2[Nzf:-1:i])) / Nave
    end
    make_strictly_increasing!(zwkb)

    zwkb2 = collect(-H:dzwkb2:0)
    interp_linextr = linear_interpolation(zwkb, zf, extrapolation_bc=Line())
    zfd = interp_linextr.(zwkb2)

    dzd = diff(zfd)
    dzmin, Imin = findmin(dzd)
    zfdadd = collect(range(zfd[Imin], 0, length=Int(ceil(abs(zfd[Imin])/dzmin))))
    zfw = vcat(zfd[1:Imin], zfdadd[2:end])

    intzc = linear_interpolation(zf, N2, extrapolation_bc=Line())
    N2w = intzc.(zfw)

    return zfw, N2w
end

# ---------------------------------------------------------------------------
# build each latitude's blended N2(z) on a common fine grid, with the
# zeroval / nearest-neighbor boundary conditions

zc = collect(mindepth:5.0:0.0)   # common merging grid, bottom -> surface

N2_merged_common = fill(NaN, length(zc), nlat)
blend_starts = zeros(nlat)
blend_ends   = zeros(nlat)

for k in 1:nlat
    Nw = N2_w_all[:,k]
    ivalid = findall(!isnan, Nw)
    zW = reverse(-depth_mid_w[ivalid])
    NW = reverse(Nw[ivalid])
    maxWOCEdepth = depth_mid_w[ivalid[end]]

    Nm = N2_m_all[:,k]
    jvalid = findall(!isnan, Nm)
    zM = reverse(-depth_mid_m[jvalid])
    NM = reverse(Nm[jvalid])

    itp_w = linear_interpolation(zW, NW, extrapolation_bc=Flat())
    itp_m = linear_interpolation(zM, NM, extrapolation_bc=Flat())

    blend_end   = min(blend_end_default, maxWOCEdepth)
    blend_start = blend_end - blend_width
    blend_starts[k] = blend_start
    blend_ends[k]   = blend_end

    N2c = zeros(length(zc))
    for (i,zi) in enumerate(zc)
        depth_i = -zi
        w = depth_i <= blend_start ? 1.0 :
            depth_i >= blend_end   ? 0.0 :
            (blend_end - depth_i) / (blend_end - blend_start)
        N2c[i] = w*itp_w(zi) + (1-w)*itp_m(zi)
    end

    nneg = count(<(0), N2c)
    nneg > 0 && println("  lat=",lat[k]," clamping ", nneg, " negative merged N2 point(s) to zeroval")
    N2c[N2c .< 0] .= zeroval

    # boundary conditions (as in the original recipe): surface -> zeroval.
    # The bottom (z=mindepth) is NOT separately overridden here -- it's
    # already the nearest-neighbor Mercator value by construction, since
    # w=0 (pure Mercator) that deep and itp_m flat-extrapolates below this
    # latitude's own deepest valid Mercator point.
    N2c[end] = zeroval   # zc[end] = 0 (surface)

    N2_merged_common[:,k] = N2c
end

# ---------------------------------------------------------------------------
# shared vertical grid: WKB-scale ONLY the reference (0 N) profile

zfw_ref, N2w_ref = wkb_regrid(zc, N2_merged_common[:,k0], dzwkb2)
println("reference (", lat[k0], " N) WKB grid: Nz = ", length(zfw_ref)-1, " layers, ", length(zfw_ref), " faces")

# ---------------------------------------------------------------------------
# regrid every latitude's merged N2 onto the shared zfw_ref (same Nz for all)

for k in 1:nlat
    N2w = if k == k0
        N2w_ref
    else
        itp = linear_interpolation(zc, N2_merged_common[:,k], extrapolation_bc=Flat())
        itp.(zfw_ref)
    end

    fname = @sprintf("N2_40W_lat%04.1f.jld2", lat[k])
    path_fname = string(dirout, fname)
    jldsave(path_fname; N2w, zfw=zfw_ref, lonsel=lon_used, latsel=lat[k])
    println(fname, " saved (blend ", round(blend_starts[k],digits=0), "-", round(blend_ends[k],digits=0),
            " m, Nz=", length(zfw_ref)-1, ") ........ ")
end

# ---------------------------------------------------------------------------
# diagnostic plot: final merged N2 profiles (on the fine common grid, pre-WKB),
# 14 latitudes distributed over three subplots (low/mid/high latitude groups)

groups = [1:5, 6:10, 11:nlat]   # 0-15N, 20-40N, 45-60N
grouptitles = ["0-15°N", "20-40°N", "45-60°N"]

fig = Figure(size = (1500, 800))

for (g,rng) in enumerate(groups)
    ax = Axis(fig[1,g], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = grouptitles[g])
    ng = length(rng)
    cols = get(ColorSchemes.viridis, range(0,1,length=ng))
    for (i,k) in enumerate(rng)
        lines!(ax, N2_merged_common[:,k], zc, color = cols[i], label = string(lat[k],"°N"))
    end
    ylims!(ax, -4000, 0)
    axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
end

Label(fig[0,1:3], "Merged (WOCE + Mercator, monthly-averaged N2) along 40°W -- shared Nz=$(length(zfw_ref)-1) grid", fontsize = 16)

save(string(dirfig,"N2_merged_40W_3panels.png"), fig)
println("saved: ", string(dirfig,"N2_merged_40W_3panels.png"))

# ---------------------------------------------------------------------------
# same, zoomed to the upper 400 m

fig2 = Figure(size = (1500, 800))

for (g,rng) in enumerate(groups)
    ax = Axis(fig2[1,g], xlabel = "N² (s⁻²)", ylabel = "Depth (m)", title = grouptitles[g])
    ng = length(rng)
    cols = get(ColorSchemes.viridis, range(0,1,length=ng))
    for (i,k) in enumerate(rng)
        lines!(ax, N2_merged_common[:,k], zc, color = cols[i], label = string(lat[k],"°N"))
    end
    ylims!(ax, -400, 0)
    axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
end

Label(fig2[0,1:3], "Merged (WOCE + Mercator, monthly-averaged N2) along 40°W, upper 400 m", fontsize = 16)

save(string(dirfig,"N2_merged_40W_3panels_upper400m.png"), fig2)
println("saved: ", string(dirfig,"N2_merged_40W_3panels_upper400m.png"))
