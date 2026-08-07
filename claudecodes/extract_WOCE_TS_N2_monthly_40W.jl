# extract_WOCE_TS_N2_monthly_40W.jl
# MCB/Claude, USM, 2026-8-4
#
# Extract WOCE-based (Maarten_NATL_mean_TS.nc) monthly in-situ T,S profiles
# along 40 W at the same latitudes used before (linearly interpolated onto the
# exact requested latitudes from the native 2 deg WOCE grid, as in
# extract_WOCE_TS_N2_40W.jl), compute N2 (TEOS-10) per month, then average the
# 12 monthly N2 profiles to get the annual mean.
#
# NOTE: this iteration does NOT clamp/zero negative N2 or noisy near-surface
# values (unlike extract_WOCE_TS_N2_40W.jl's shallow_method option) -- raw
# monthly N2, including any negative values, is averaged as-is. To be revisited.
#
# mean_temp is IN-SITU temperature -> gsw_ct_from_t (with pressure).

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater
using Printf

# ---------------------------------------------------------------------------
# input/output

dirin   = "/home/mbui/ModelOutput/IW/stratification/"
fnamein = "Maarten_NATL_mean_TS.nc"

dirout  = dirin
fnameTS = "WOCE_TS_monthly_40W.nc"
fnameN2 = "WOCE_N2_monthly_40W.nc"

lonsel  = -40.0 + 360.0   # file uses degrees_east, 0-360
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat    = length(latsels)
nmonths = 12

fillthresh = 1f30   # WOCE file uses the netCDF default fill (9.96921e36), no _FillValue attrib

# ---------------------------------------------------------------------------
# read grid

ds = NCDataset(string(dirin,fnamein),"r")

longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])
nz        = length(depth)

is, dlon = nearest_index(longitude, lonsel)
println("requested lon = ", lonsel-360, " W, nearest lon = ", longitude[is]-360, " W, dlon = ", dlon)

lon_used = Float64(longitude[is]) - 360.0
lat_used = latsels

# single-month profile (masking the fill value) at a native WOCE latitude index j
function woce_month_profile(ds, is, j, m, nz)
    Tm = Float64.(ds["mean_temp"][is,j,:,m])
    Sm = Float64.(ds["mean_salt"][is,j,:,m])
    Tm[Tm .> fillthresh] .= NaN
    Sm[Sm .> fillthresh] .= NaN
    return Tm, Sm
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

for k in 1:nlat
    jlo, jhi, frac = bracket(latitude, latsels[k])
    println("requested lat = ", latsels[k], ", bracket = [", latitude[jlo], ", ", latitude[jhi], "], frac = ", round(frac,digits=2))
end

# ---------------------------------------------------------------------------
# extract T,S per month at the exact requested latitudes (linear interp in lat)

Tmonth = fill(NaN, nz, nlat, nmonths)
Smonth = fill(NaN, nz, nlat, nmonths)

for k in 1:nlat
    jlo, jhi, frac = bracket(latitude, latsels[k])
    for m in 1:nmonths
        Tlo, Slo = woce_month_profile(ds, is, jlo, m, nz)
        if jlo == jhi
            Tmonth[:,k,m] = Tlo
            Smonth[:,k,m] = Slo
        else
            Thi, Shi = woce_month_profile(ds, is, jhi, m, nz)
            Tmonth[:,k,m] = (1-frac).*Tlo .+ frac.*Thi
            Smonth[:,k,m] = (1-frac).*Slo .+ frac.*Shi
        end
    end
end

close(ds)

# ---------------------------------------------------------------------------
# compute N2 (TEOS-10) per month -- no clamping of negative/noisy values here

depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2
N2month = fill(NaN, nz-1, nlat, nmonths)

for k in 1:nlat, m in 1:nmonths
    Ts = Tmonth[:,k,m]
    Ss = Smonth[:,k,m]

    igood = findall(!isnan, Ts .+ Ss)
    if length(igood) < 2
        continue
    end
    Tsg = Ts[igood]; Ssg = Ss[igood]; dpk = depth[igood]
    nzk = length(dpk)

    p  = gsw_p_from_z.(-dpk, lat_used[k])
    SA = gsw_sa_from_sp.(Ssg, p, lon_used, lat_used[k])
    CT = gsw_ct_from_t.(SA, Tsg, p)

    N2k = zeros(nzk-1); Pmidk = zeros(nzk-1)
    Lats = fill(lat_used[k], nzk)
    gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

    N2month[1:nzk-1,k,m] = N2k
end

# ---------------------------------------------------------------------------
# annual mean of the monthly N2 profiles (mean over month, skipping NaN;
# negative N2 values are included as-is, not clamped)

N2_annual = fill(NaN, nz-1, nlat)
for k in 1:nlat, z in 1:nz-1
    vals = filter(!isnan, N2month[z,k,:])
    if !isempty(vals)
        N2_annual[z,k] = sum(vals)/length(vals)
    end
end

# ---------------------------------------------------------------------------
# save T,S monthly profiles

isfile(string(dirout,fnameTS)) && rm(string(dirout,fnameTS))
dsout = NCDataset(string(dirout,fnameTS),"c")

defDim(dsout,"depth",nz)
defDim(dsout,"latitude",nlat)
defDim(dsout,"month",nmonths)

v = defVar(dsout,"depth",Float64,("depth",)); v[:] = depth
v.attrib["units"] = "m"; v.attrib["long_name"] = "Depth"; v.attrib["positive"] = "down"

v = defVar(dsout,"latitude",Float64,("latitude",)); v[:] = lat_used
v.attrib["units"] = "degrees_north"; v.attrib["long_name"] = "Latitude"

v = defVar(dsout,"month",Int32,("month",)); v[:] = 1:nmonths

v = defVar(dsout,"temp",Float64,("depth","latitude","month"), fillvalue=NaN); v[:,:,:] = Tmonth
v.attrib["units"] = "degrees_C"; v.attrib["long_name"] = "In-situ temperature (monthly mean)"

v = defVar(dsout,"salt",Float64,("depth","latitude","month"), fillvalue=NaN); v[:,:,:] = Smonth
v.attrib["units"] = "PSU"; v.attrib["long_name"] = "Salinity (monthly mean)"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "WOCE (Gregg Jacobs / WOD) in-situ T,S monthly-mean profiles along 40 W"

close(dsout)
println(fnameTS," saved ........ ")

# ---------------------------------------------------------------------------
# save monthly + annual-mean-of-monthly N2

isfile(string(dirout,fnameN2)) && rm(string(dirout,fnameN2))
dsout = NCDataset(string(dirout,fnameN2),"c")

defDim(dsout,"depth_mid",nz-1)
defDim(dsout,"latitude",nlat)
defDim(dsout,"month",nmonths)

v = defVar(dsout,"depth_mid",Float64,("depth_mid",)); v[:] = depth_mid
v.attrib["units"] = "m"; v.attrib["long_name"] = "Depth at N2 mid-points"; v.attrib["positive"] = "down"

v = defVar(dsout,"latitude",Float64,("latitude",)); v[:] = lat_used
v.attrib["units"] = "degrees_north"; v.attrib["long_name"] = "Latitude"

v = defVar(dsout,"month",Int32,("month",)); v[:] = 1:nmonths

v = defVar(dsout,"N2_monthly",Float64,("depth_mid","latitude","month"), fillvalue=NaN); v[:,:,:] = N2month
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Monthly-mean N2 (raw, negative values not clamped)"

v = defVar(dsout,"N2_annual_mean",Float64,("depth_mid","latitude"), fillvalue=NaN); v[:,:] = N2_annual
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Annual mean of the 12 monthly N2 profiles (raw, negative values not clamped)"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "WOCE N2 (TEOS-10, in-situ T -> gsw_ct_from_t): monthly + annual mean of monthly N2, along 40 W"

close(dsout)
println(fnameN2," saved ........ ")
