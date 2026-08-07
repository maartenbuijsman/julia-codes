# extract_WOCE_TS_N2_Atlantic_zonalmean_40W.jl
# MCB/Claude, USM, 2026-8-4
#
# Same idea as extract_TS_N2_Atlantic_zonalmean_40W.jl (Mercator), applied to
# the WOCE-based NATL climatology (Maarten_NATL_mean_TS.nc): compute N2
# (TEOS-10, annual-mean T,S, in-situ T -> gsw_ct_from_t) at every WOCE column
# in the same open-Atlantic longitude band (60W-20W), and zonally average --
# at each of WOCE's native 2 deg latitude rows first, then linearly
# interpolate the (already smooth) zonal-mean profiles onto the exact
# requested latitudes (2.5, 15, 25, ...), same as extract_WOCE_TS_N2_40W.jl /
# extract_WOCE_TS_N2_monthly_40W.jl.
#
# NOTE: unlike the Mercator version, there is NO minimum-bathymetry filter
# here. For WOCE, "deepest valid data level" reflects the Gregg Jacobs
# objective analysis's historical CTD/cast coverage, not real sea floor
# depth (confirmed: nearly every requested latitude at 40W alone had an
# oddly-uniform ~1960 m cutoff regardless of true bathymetry, while a few
# better-sampled latitude bands reached deeper) -- so a ">2000 m" filter
# would just select a sparse, geographically biased subset of well-sampled
# casts, not "deep ocean" columns. All available columns are used instead,
# with the standard per-depth-level NaN-aware averaging (a column only
# contributes at depths where it actually has data).
#
# WOCE's native longitude grid is only 2 deg spaced (~21 columns in this
# band vs. Mercator's 481), so this yields far fewer columns per average,
# but should still smooth out single-column noise.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater
using Statistics

dirin   = "/home/mbui/ModelOutput/IW/stratification/"
fnamein = "Maarten_NATL_mean_TS.nc"
dirout  = dirin
fnameout = "WOCE_N2_zonalmean_Atl_40W.nc"

lonband_e = (300.0, 340.0)   # degrees_east, = 60W to 20W
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat    = length(latsels)
nmonths = 12
fillthresh = 1f30

ds = NCDataset(string(dirin,fnamein),"r")
longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])
nz        = length(depth)
nlon_native = length(longitude)
nlat_native = length(latitude)

ilo, _ = nearest_index(longitude, lonband_e[1])
ihi, _ = nearest_index(longitude, lonband_e[2])
println("longitude band: ", longitude[ilo]-360, " to ", longitude[ihi]-360, " W (", ihi-ilo+1, " native columns)")

depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2

# annual-mean T,S at a single native (lon index, lat index) column
function woce_annual_col(ds, i, j, nz)
    Tm = Float64.(ds["mean_temp"][i,j,:,:])   # (depth, month)
    Sm = Float64.(ds["mean_salt"][i,j,:,:])
    Tm[Tm .> fillthresh] .= NaN
    Sm[Sm .> fillthresh] .= NaN
    Tp = fill(NaN, nz); Sp = fill(NaN, nz)
    for z in 1:nz
        Tz = filter(!isnan, Tm[z,:]); Sz = filter(!isnan, Sm[z,:])
        if !isempty(Tz) && !isempty(Sz)
            Tp[z] = sum(Tz)/length(Tz)
            Sp[z] = sum(Sz)/length(Sz)
        end
    end
    return Tp, Sp
end

# ---------------------------------------------------------------------------
# zonal mean at every native latitude row

N2_zonalmean_native = fill(NaN, nz-1, nlat_native)
ncolumns_native = zeros(Int, nlat_native)

for j in 1:nlat_native
    N2cols = fill(NaN, nz-1, ihi-ilo+1)
    kept = 0
    for (c,i) in enumerate(ilo:ihi)
        Tp, Sp = woce_annual_col(ds, i, j, nz)
        igood = findall(!isnan, Tp .+ Sp)
        isempty(igood) && continue

        Tsg = Tp[igood]; Ssg = Sp[igood]; dpk = depth[igood]
        nzk = length(dpk)

        p  = gsw_p_from_z.(-dpk, latitude[j])
        SA = gsw_sa_from_sp.(Ssg, p, longitude[i]-360.0, latitude[j])
        CT = gsw_ct_from_t.(SA, Tsg, p)

        N2k = zeros(nzk-1); Pmidk = zeros(nzk-1)
        Lats = fill(latitude[j], nzk)
        gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

        N2cols[1:nzk-1,c] = N2k
        kept += 1
    end

    ncolumns_native[j] = kept
    for z in 1:nz-1
        vals = filter(!isnan, N2cols[z,:])
        if !isempty(vals)
            N2_zonalmean_native[z,j] = mean(vals)
        end
    end
    println("native lat = ", latitude[j], ": ", kept, " of ", ihi-ilo+1, " columns had any valid data")
end

close(ds)

# ---------------------------------------------------------------------------
# interpolate the (already-smooth) zonal-mean profiles onto the exact
# requested latitudes

function bracket(grid, x)
    if x <= grid[1]
        return 1, 1, 0.0
    elseif x >= grid[end]
        return length(grid), length(grid), 0.0
    end
    jhi = findfirst(>=(x), grid)
    grid[jhi] == x && return jhi, jhi, 0.0
    jlo = jhi - 1
    return jlo, jhi, (x - grid[jlo]) / (grid[jhi] - grid[jlo])
end

N2_zonalmean = fill(NaN, nz-1, nlat)
ncolumns_lo  = zeros(Int, nlat)
ncolumns_hi  = zeros(Int, nlat)

for k in 1:nlat
    jlo, jhi, frac = bracket(latitude, latsels[k])
    ncolumns_lo[k] = ncolumns_native[jlo]
    ncolumns_hi[k] = ncolumns_native[jhi]
    if jlo == jhi
        N2_zonalmean[:,k] = N2_zonalmean_native[:,jlo]
    else
        Nlo = N2_zonalmean_native[:,jlo]; Nhi = N2_zonalmean_native[:,jhi]
        for z in 1:nz-1
            if !isnan(Nlo[z]) && !isnan(Nhi[z])
                N2_zonalmean[z,k] = (1-frac)*Nlo[z] + frac*Nhi[z]
            elseif !isnan(Nlo[z]) && frac == 0.0
                N2_zonalmean[z,k] = Nlo[z]
            elseif !isnan(Nhi[z]) && frac == 1.0
                N2_zonalmean[z,k] = Nhi[z]
            end
        end
    end
end

# ---------------------------------------------------------------------------
# save

isfile(string(dirout,fnameout)) && rm(string(dirout,fnameout))
dsout = NCDataset(string(dirout,fnameout),"c")

defDim(dsout,"depth_mid",nz-1)
defDim(dsout,"latitude",nlat)

v = defVar(dsout,"depth_mid",Float64,("depth_mid",)); v[:] = depth_mid
v.attrib["units"] = "m"; v.attrib["long_name"] = "Depth at N2 mid-points"; v.attrib["positive"] = "down"

v = defVar(dsout,"latitude",Float64,("latitude",)); v[:] = latsels
v.attrib["units"] = "degrees_north"; v.attrib["long_name"] = "Latitude"

v = defVar(dsout,"N2_zonalmean",Float64,("depth_mid","latitude"), fillvalue=NaN); v[:,:] = N2_zonalmean
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Zonal-mean N2 over all open-Atlantic WOCE columns (no bathymetry filter), interpolated to the requested latitude"

v = defVar(dsout,"ncolumns_lo",Int32,("latitude",)); v[:] = ncolumns_lo
v.attrib["long_name"] = "Number of columns with any valid data at the native latitude row below/at the requested latitude"

v = defVar(dsout,"ncolumns_hi",Int32,("latitude",)); v[:] = ncolumns_hi
v.attrib["long_name"] = "Number of columns with any valid data at the native latitude row above/at the requested latitude"

dsout.attrib["longitude_band"] = "60W to 20W"
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "WOCE N2 (TEOS-10, in-situ T -> gsw_ct_from_t), zonally averaged over all open-Atlantic (60W-20W) columns (no bathymetry filter -- WOCE's depth cutoff reflects cast coverage, not sea floor) at each native 2 deg latitude row, then linearly interpolated in latitude to the requested latitude"

close(dsout)
println(fnameout," saved ........ ")
