# extract_TS_N2_Atlantic_zonalmean_40W.jl
# MCB/Claude, USM, 2026-8-4
#
# Alternative deep-water source for the N2 merge: instead of relying on the
# single Mercator column at exactly 40 W (which runs out of valid data as
# shallow as ~2083 m at some latitudes, per merge_N2_WOCE_Mercator_40W.jl),
# compute N2 (TEOS-10, annual-mean T,S -> gl12_mean_1993_2016_allmonths.nc)
# at every Mercator column in an open-Atlantic longitude band, keep only
# columns whose sea floor is deeper than 2000 m (using the deepest valid/
# unmasked T,S level as the bathymetry proxy), and zonally average the
# resulting N2 profiles at each of the requested latitudes.
#
# Longitude band: 60 W to 20 W, a +/-20 deg window centered on 40 W. This
# stays within the open Atlantic at every requested latitude (0-60 N) without
# needing a separate marginal-sea mask: it's entirely west of Gibraltar
# (excludes the Mediterranean) and east of the Caribbean/Gulf of Mexico.
# thetao is potential temperature -> gsw_ct_from_pt (as elsewhere for Mercator).

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater
using Statistics

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
fnamein = "gl12_mean_1993_2016_allmonths.nc"
dirout = dirin
fnameout = "MERC_N2_zonalmean_Atl_40W.nc"

lonband = (-60.0, -20.0)
mindepth_bathy = 2000.0   # only keep columns whose sea floor is deeper than this
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat = length(latsels)

ds = NCDataset(string(dirin,fnamein),"r")
longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])
nz        = length(depth)

ilo, _ = nearest_index(longitude, lonband[1])
ihi, _ = nearest_index(longitude, lonband[2])
println("longitude band: ", longitude[ilo], " to ", longitude[ihi], " (", ihi-ilo+1, " columns)")

js = zeros(Int, nlat)
for k in 1:nlat
    js[k], dlat = nearest_index(latitude, latsels[k])
end

depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2

N2_zonalmean = fill(NaN, nz-1, nlat)
ncolumns     = zeros(Int, nlat)

for k in 1:nlat
    Tblock = ds["thetao"][ilo:ihi,js[k],:,1]   # (lon, depth)
    Sblock = ds["so"][ilo:ihi,js[k],:,1]
    ncol = size(Tblock,1)

    N2cols = fill(NaN, nz-1, ncol)
    kept = 0

    for c in 1:ncol
        Ts = Tblock[c,:]; Ss = Sblock[c,:]
        igood = findall(!ismissing, Ts .+ Ss)
        isempty(igood) && continue
        bathy = depth[igood[end]]
        bathy <= mindepth_bathy && continue

        Tsg = Float64.(Ts[igood]); Ssg = Float64.(Ss[igood]); dpk = depth[igood]
        nzk = length(dpk)

        p  = gsw_p_from_z.(-dpk, latsels[k])
        SA = gsw_sa_from_sp.(Ssg, p, longitude[ilo+c-1], latsels[k])
        CT = gsw_ct_from_pt.(SA, Tsg)

        N2k = zeros(nzk-1); Pmidk = zeros(nzk-1)
        Lats = fill(latsels[k], nzk)
        gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

        N2cols[1:nzk-1,c] = N2k
        kept += 1
    end

    ncolumns[k] = kept
    for z in 1:nz-1
        vals = filter(!isnan, N2cols[z,:])
        if !isempty(vals)
            N2_zonalmean[z,k] = mean(vals)
        end
    end
    println("lat = ", latsels[k], ": ", kept, " of ", ncol, " columns had sea floor > ", mindepth_bathy, " m")
end

close(ds)

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
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Zonal-mean N2 over open-Atlantic columns with sea floor > 2000 m"

v = defVar(dsout,"ncolumns",Int32,("latitude",)); v[:] = ncolumns
v.attrib["long_name"] = "Number of qualifying (sea floor > 2000 m) columns averaged"

dsout.attrib["longitude_band"] = string(lonband[1]," to ",lonband[2])
dsout.attrib["source"]    = string(dirin,fnamein)
dsout.attrib["description"] = "Mercator N2 (TEOS-10, potential T -> gsw_ct_from_pt), zonally averaged over open-Atlantic (60W-20W) columns with sea floor > 2000 m, at each latitude"

close(dsout)
println(fnameout," saved ........ ")
