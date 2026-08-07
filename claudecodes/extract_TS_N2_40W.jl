# extract_TS_N2_40W.jl
# MCB/Claude, USM, 2026-8-2
# Extract vertical T,S profiles from Mercator/GLORYS12V1 annual mean
# along 40 W at latitudes 0, 2.5, 5:5:60, and compute N2 (TEOS-10).

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater

# ---------------------------------------------------------------------------
# input/output

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
fnamein = "gl12_mean_1993_2016_allmonths.nc"

dirout = dirin
fnameTS = "TS_profiles_40W.nc"
fnameN2 = "N2_profiles_40W.nc"

lonsel  = -40.0
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat    = length(latsels)

# ---------------------------------------------------------------------------
# read grid and locate longitude / latitude indices

ds = NCDataset(string(dirin,fnamein),"r")

longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])
nz        = length(depth)

is, dlon = nearest_index(longitude, lonsel)
println("requested lon = ", lonsel, ", nearest lon = ", longitude[is], ", dlon = ", dlon)

js = zeros(Int, nlat)
for k in 1:nlat
    js[k], dlat = nearest_index(latitude, latsels[k])
    println("requested lat = ", latsels[k], ", nearest lat = ", latitude[js[k]], ", dlat = ", dlat)
end

lon_used = Float64(longitude[is])
lat_used = Float64.(latitude[js])

# ---------------------------------------------------------------------------
# extract T,S profiles (time = 1, i.e. the all-months / annual mean)

Tprof = Array{Union{Missing,Float64}}(missing, nz, nlat)
Sprof = Array{Union{Missing,Float64}}(missing, nz, nlat)

for k in 1:nlat
    Tprof[:,k] = ds["thetao"][is,js[k],:,1]
    Sprof[:,k] = ds["so"][is,js[k],:,1]
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

v = defVar(dsout,"thetao",Float64,("depth","latitude"), fillvalue=NaN); v[:,:] = coalesce.(Tprof, NaN)
v.attrib["units"] = "degrees_C"; v.attrib["long_name"] = "Temperature"

v = defVar(dsout,"so",Float64,("depth","latitude"), fillvalue=NaN); v[:,:] = coalesce.(Sprof, NaN)
v.attrib["units"] = "1e-3"; v.attrib["long_name"] = "Salinity"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "T,S profiles along 40 W at selected latitudes"

close(dsout)
println(fnameTS," saved ........ ")

# ---------------------------------------------------------------------------
# compute buoyancy frequency N2 using TEOS-10 (GibbsSeaWater.jl)
# N2 is computed at mid-depths between the depth levels (nz-1 values)

N2prof   = fill(NaN, nz-1, nlat)
Pmidprof = fill(NaN, nz-1, nlat)

for k in 1:nlat
    Ts = Tprof[:,k]
    Ss = Sprof[:,k]
    dpk = depth

    # trim missing values at/below the sea floor
    igood = findall(!ismissing, Ts .+ Ss)
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
    # Mercator/GLORYS "thetao" is potential temperature (referenced to the
    # sea surface), not in-situ temperature, so convert via gsw_ct_from_pt
    # (no pressure argument) rather than gsw_ct_from_t.
    CT  = gsw_ct_from_pt.(SA, Ts)

    N2k   = zeros(nzk-1)
    Pmidk = zeros(nzk-1)
    Lats  = fill(lat_used[k], nzk)
    gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

    N2prof[1:nzk-1,k]   = N2k
    Pmidprof[1:nzk-1,k] = Pmidk
end

# mid-point depths corresponding to N2 (same grid at every latitude)
depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2

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
dsout.attrib["description"] = "N2 (TEOS-10, GibbsSeaWater.jl) along 40 W at selected latitudes"

close(dsout)
println(fnameN2," saved ........ ")
