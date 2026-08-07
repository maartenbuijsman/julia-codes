# extract_WOCE_TS_N2_40W.jl
# MCB/Claude, USM, 2026-8-4
# Extract annual-mean in-situ T,S profiles from the WOCE-based NATL climatology
# (Maarten_NATL_mean_TS.nc, cf. ATL_stratification_profile.jl) along 40 W at the
# same latitudes used for the Mercator extraction, and compute N2 (TEOS-10).
# Note: mean_temp here is IN-SITU temperature (unlike Mercator's thetao, which is
# potential temperature) -> use gsw_ct_from_t (with pressure), not gsw_ct_from_pt.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater

# ---------------------------------------------------------------------------
# input/output

dirin   = "/home/mbui/ModelOutput/IW/stratification/"
fnamein = "Maarten_NATL_mean_TS.nc"

dirout  = dirin
fnameTS = "WOCE_TS_profiles_40W.nc"
fnameN2 = "WOCE_N2_profiles_40W.nc"

lonsel  = -40.0 + 360.0   # file uses degrees_east, 0-360
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat    = length(latsels)

# how to handle the noisy near-surface (< 10 m) N2 levels:
#   :value_at_10m -> clamp to the N2 value at the level nearest 10 m (default)
#   :zeroval      -> map to a small positive placeholder, matching the
#                     zeroval = 1e-12 convention used for N2<0 / mixed-layer
#                     points in AMZ_stratification_profile.jl / ATL_stratification_profile.jl
shallow_method = :zeroval
zeroval = 1e-12

fillthresh = 1f30   # WOCE file uses the netCDF default fill (9.96921e36), no _FillValue attrib

# ---------------------------------------------------------------------------
# read grid and locate longitude / latitude indices

ds = NCDataset(string(dirin,fnamein),"r")

longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])
nz        = length(depth)

is, dlon = nearest_index(longitude, lonsel)
println("requested lon = ", lonsel-360, " W, nearest lon = ", longitude[is]-360, " W, dlon = ", dlon)

lon_used = Float64(longitude[is]) - 360.0
lat_used = latsels   # WOCE is linearly interpolated onto the exact requested latitudes below

# annual-mean profile (masking the fill value) at a native WOCE latitude index j
function woce_annual_profile(ds, is, j, nz)
    Tm = Float64.(ds["mean_temp"][is,j,:,:])   # (depth, month)
    Sm = Float64.(ds["mean_salt"][is,j,:,:])
    Tm[Tm .> fillthresh] .= NaN
    Sm[Sm .> fillthresh] .= NaN

    Tp = fill(NaN, nz); Sp = fill(NaN, nz)
    for z in 1:nz
        Tz = filter(!isnan, Tm[z,:])
        Sz = filter(!isnan, Sm[z,:])
        if !isempty(Tz) && !isempty(Sz)
            Tp[z] = sum(Tz)/length(Tz)
            Sp[z] = sum(Sz)/length(Sz)
        end
    end
    return Tp, Sp
end

# bracket a native grid (assumed sorted ascending) around x; returns (jlo,jhi,frac)
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

# ---------------------------------------------------------------------------
# extract T,S profiles at the exact requested latitudes: annual mean over the
# 12 months at the two bracketing native (2 deg spaced) WOCE latitudes, then
# linearly interpolated in latitude (nearest-neighbor snapping would put e.g.
# 2.5 N at the native 2.0 N point, up to 1 deg off)

Tprof = fill(NaN, nz, nlat)
Sprof = fill(NaN, nz, nlat)

for k in 1:nlat
    jlo, jhi, frac = bracket(latitude, latsels[k])
    println("requested lat = ", latsels[k], ", bracket = [", latitude[jlo], ", ", latitude[jhi], "], frac = ", round(frac,digits=2))

    Tlo, Slo = woce_annual_profile(ds, is, jlo, nz)
    if jlo == jhi
        Tprof[:,k] = Tlo
        Sprof[:,k] = Slo
    else
        Thi, Shi = woce_annual_profile(ds, is, jhi, nz)
        Tprof[:,k] = (1-frac).*Tlo .+ frac.*Thi
        Sprof[:,k] = (1-frac).*Slo .+ frac.*Shi
    end
end

close(ds)

# ---------------------------------------------------------------------------
# save T,S profiles

isfile(string(dirout,fnameTS)) && rm(string(dirout,fnameTS))

dsout = NCDataset(string(dirout,fnameTS),"c")

defDim(dsout,"depth",nz)
defDim(dsout,"latitude",nlat)

v = defVar(dsout,"depth",Float64,("depth",)); v[:] = depth
v.attrib["units"] = "m"; v.attrib["long_name"] = "Depth"; v.attrib["positive"] = "down"

v = defVar(dsout,"latitude",Float64,("latitude",)); v[:] = lat_used
v.attrib["units"] = "degrees_north"; v.attrib["long_name"] = "Latitude"

v = defVar(dsout,"temp",Float64,("depth","latitude"), fillvalue=NaN); v[:,:] = Tprof
v.attrib["units"] = "degrees_C"; v.attrib["long_name"] = "In-situ temperature (annual mean)"

v = defVar(dsout,"salt",Float64,("depth","latitude"), fillvalue=NaN); v[:,:] = Sprof
v.attrib["units"] = "PSU"; v.attrib["long_name"] = "Salinity (annual mean)"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "WOCE (Gregg Jacobs / WOD) in-situ T,S annual-mean profiles along 40 W"

close(dsout)
println(fnameTS," saved ........ ")

# ---------------------------------------------------------------------------
# compute buoyancy frequency N2 using TEOS-10 (GibbsSeaWater.jl)
# mean_temp is IN-SITU temperature here -> gsw_ct_from_t (with pressure)

N2prof   = fill(NaN, nz-1, nlat)
Pmidprof = fill(NaN, nz-1, nlat)

for k in 1:nlat
    Ts = Tprof[:,k]
    Ss = Sprof[:,k]
    dpk = depth

    igood = findall(!isnan, Ts .+ Ss)
    if length(igood) < 2
        println("lat = ", lat_used[k], ": not enough valid data, skipping N2")
        continue
    end
    Ts = Float64.(Ts[igood])
    Ss = Float64.(Ss[igood])
    dpk = dpk[igood]
    nzk = length(dpk)

    p   = gsw_p_from_z.(-dpk, lat_used[k])
    SA  = gsw_sa_from_sp.(Ss, p, lon_used, lat_used[k])
    CT  = gsw_ct_from_t.(SA, Ts, p)

    N2k   = zeros(nzk-1)
    Pmidk = zeros(nzk-1)
    Lats  = fill(lat_used[k], nzk)
    gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

    N2prof[1:nzk-1,k]   = N2k
    Pmidprof[1:nzk-1,k] = Pmidk

    println("lat = ", lat_used[k], ": max valid depth = ", dpk[end], " m")
end

depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2

# ---------------------------------------------------------------------------
# the shallowest N2 levels (< 10 m) are noisy (very sparse WOCE sampling near
# the surface); map them per shallow_method above

idx10 = argmin(abs.(depth_mid .- 10.0))
ishallow = findall(depth_mid .< 10.0)

if shallow_method == :zeroval
    println("mapping depth_mid = ", depth_mid[ishallow], " m to zeroval = ", zeroval)
    for k in 1:nlat
        N2prof[ishallow,k] .= zeroval
    end
elseif shallow_method == :value_at_10m
    println("clamping depth_mid = ", depth_mid[ishallow], " m to N2 at depth_mid = ", depth_mid[idx10], " m")
    for k in 1:nlat
        N2prof[ishallow,k] .= N2prof[idx10,k]
    end
else
    error("unknown shallow_method: ", shallow_method, " (use :value_at_10m or :zeroval)")
end

# ---------------------------------------------------------------------------
# save N2 profiles

isfile(string(dirout,fnameN2)) && rm(string(dirout,fnameN2))

dsout = NCDataset(string(dirout,fnameN2),"c")

defDim(dsout,"depth_mid",nz-1)
defDim(dsout,"latitude",nlat)

v = defVar(dsout,"depth_mid",Float64,("depth_mid",)); v[:] = depth_mid
v.attrib["units"] = "m"; v.attrib["long_name"] = "Depth at N2 mid-points"; v.attrib["positive"] = "down"

v = defVar(dsout,"latitude",Float64,("latitude",)); v[:] = lat_used
v.attrib["units"] = "degrees_north"; v.attrib["long_name"] = "Latitude"

v = defVar(dsout,"N2",Float64,("depth_mid","latitude"), fillvalue=NaN); v[:,:] = N2prof
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Buoyancy (Brunt-Vaisala) frequency squared"

v = defVar(dsout,"pressure_mid",Float64,("depth_mid","latitude"), fillvalue=NaN); v[:,:] = Pmidprof
v.attrib["units"] = "dbar"; v.attrib["long_name"] = "Pressure at N2 mid-points"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "WOCE N2 (TEOS-10, in-situ T -> gsw_ct_from_t) along 40 W"

close(dsout)
println(fnameN2," saved ........ ")
