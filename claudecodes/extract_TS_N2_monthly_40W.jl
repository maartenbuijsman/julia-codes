# extract_TS_N2_monthly_40W.jl
# MCB/Claude, USM, 2026-8-4
#
# Extract Mercator/GLORYS12V1 monthly-mean T,S profiles (the 12 individual
# mercatorglorys12v1_gl12_mean_1993_2016_MM.nc files, as opposed to the
# all-months annual mean) along 40 W at the same latitudes used before, compute
# N2 (TEOS-10) for each month, then average the 12 *monthly N2 profiles* to get
# the annual mean -- averaging N2 directly is preferred over averaging T,S
# first and then computing N2, since N2 depends nonlinearly on the vertical
# T,S gradients (e.g. seasonal mixed-layer depth changes would get blurred out
# by averaging T,S before differentiating).
#
# thetao is potential temperature (as with the annual-mean file) -> gsw_ct_from_pt.

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater
using Printf

# ---------------------------------------------------------------------------
# input/output

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
dirout = dirin

fnameTS = "MERC_TS_monthly_40W.nc"
fnameN2 = "MERC_N2_monthly_40W.nc"

lonsel  = -40.0
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat    = length(latsels)
nmonths = 12

# ---------------------------------------------------------------------------
# grid (same for all 12 monthly files)

ds1 = NCDataset(string(dirin,@sprintf("mercatorglorys12v1_gl12_mean_1993_2016_%02i.nc",1)),"r")
longitude = ds1["longitude"][:]
latitude  = ds1["latitude"][:]
depth     = Float64.(ds1["depth"][:])
nz        = length(depth)
close(ds1)

is, dlon = nearest_index(longitude, lonsel)
println("requested lon = ", lonsel, ", nearest lon = ", longitude[is], ", dlon = ", dlon)

js = zeros(Int, nlat)
for k in 1:nlat
    js[k], dlat = nearest_index(latitude, latsels[k])
    println("requested lat = ", latsels[k], ", nearest lat = ", latitude[js[k]], ", dlat = ", dlat)
end

lon_used = Float64(longitude[is])
lat_used = Float64.(latitude[js])
depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2

# ---------------------------------------------------------------------------
# extract T,S and compute N2 for each month

Tmonth  = fill(NaN, nz,   nlat, nmonths)
Smonth  = fill(NaN, nz,   nlat, nmonths)
N2month = fill(NaN, nz-1, nlat, nmonths)

for m in 1:nmonths
    fname = @sprintf("mercatorglorys12v1_gl12_mean_1993_2016_%02i.nc", m)
    ds = NCDataset(string(dirin,fname),"r")

    for k in 1:nlat
        Ts = ds["thetao"][is,js[k],:,1]
        Ss = ds["so"][is,js[k],:,1]

        igood = findall(!ismissing, Ts .+ Ss)
        Tsg = Float64.(Ts[igood]); Ssg = Float64.(Ss[igood]); dpk = depth[igood]
        nzk = length(dpk)

        Tmonth[igood,k,m] = Tsg
        Smonth[igood,k,m] = Ssg

        p  = gsw_p_from_z.(-dpk, lat_used[k])
        SA = gsw_sa_from_sp.(Ssg, p, lon_used, lat_used[k])
        CT = gsw_ct_from_pt.(SA, Tsg)

        N2k = zeros(nzk-1); Pmidk = zeros(nzk-1)
        Lats = fill(lat_used[k], nzk)
        gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

        N2month[1:nzk-1,k,m] = N2k
    end

    close(ds)
    println("month ", m, " done")
end

# ---------------------------------------------------------------------------
# annual mean of the monthly N2 profiles (mean over month, skipping any NaN)

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

v = defVar(dsout,"thetao",Float64,("depth","latitude","month"), fillvalue=NaN); v[:,:,:] = Tmonth
v.attrib["units"] = "degrees_C"; v.attrib["long_name"] = "Potential temperature (monthly mean)"

v = defVar(dsout,"so",Float64,("depth","latitude","month"), fillvalue=NaN); v[:,:,:] = Smonth
v.attrib["units"] = "1e-3"; v.attrib["long_name"] = "Salinity (monthly mean)"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,"mercatorglorys12v1_gl12_mean_1993_2016_MM.nc")
dsout.attrib["description"] = "Mercator/GLORYS12V1 monthly-mean T,S profiles along 40 W"

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
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Monthly-mean N2"

v = defVar(dsout,"N2_annual_mean",Float64,("depth_mid","latitude"), fillvalue=NaN); v[:,:] = N2_annual
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Annual mean of the 12 monthly N2 profiles"

dsout.attrib["longitude"] = lon_used
dsout.attrib["source"]    = string(dirin,"mercatorglorys12v1_gl12_mean_1993_2016_MM.nc")
dsout.attrib["description"] = "Mercator N2 (TEOS-10, potential T -> gsw_ct_from_pt): monthly + annual mean of monthly N2, along 40 W"

close(dsout)
println(fnameN2," saved ........ ")
