# extract_TS_N2_Atlantic_zonalmean_offshelf_monthly.jl
# MCB/Claude, USM, 2026-8-5
#
# Three-stage domain determination + N2 computation, per the discussed
# recipe:
#   Range 1: true coastline extent at each latitude (from
#            find_land_crossings.jl's west/east coast crossings, walking out
#            from 40 W), with the 15N override (60W to 17W) applied since the
#            raw walk there threads through the Caribbean/Central America.
#   Range 2: within Range 1, walk INWARD from each coast until the sea floor
#            first exceeds 500 m (the real shelf break, anchored to the
#            coast rather than walking out from a single center point --
#            avoids the earlier problem of an isolated seamount/ridge point
#            falsely truncating the search).
#   N2: for every column in Range 2, compute N2 (TEOS-10) from EACH of the 12
#       individual monthly Mercator files (not from annual-mean T,S), average
#       those 12 monthly N2 profiles to an annual mean PER COLUMN, then
#       zonally average the per-column annual-mean N2 profiles across all
#       qualifying columns at each latitude (per-depth-level NaN-aware, i.e.
#       a column only contributes at depths shallower than its own sea floor).

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using NCDatasets
using GibbsSeaWater
using Statistics
using Printf

dirin  = "/home/mbui/ModelOutput/IW/mercator/"
fnameannual = "gl12_mean_1993_2016_allmonths.nc"
dirout = dirin
fnameout = "MERC_N2_zonalmean_Atl_offshelf_monthly.nc"

lonsearch = (-95.0, 25.0)   # wide search window for Range 1
loncenter = -40.0
shelf_threshold = 500.0
latsels = Float64.(vcat(0, 2.5, 5:5:60))
nlat = length(latsels)
nmonths = 12

# 15N override: raw west-coast walk threads through the Caribbean/Central
# America rather than stopping at the edge of the open Atlantic
range1_overrides = Dict(15.0 => (-60.0, -17.0))

# ---------------------------------------------------------------------------
# Range 1 (coastline) and Range 2 (shelf-break inward from Range 1) per lat

ds = NCDataset(string(dirin,fnameannual),"r")
longitude = ds["longitude"][:]
latitude  = ds["latitude"][:]
depth     = Float64.(ds["depth"][:])
nz        = length(depth)

ilo, _ = nearest_index(longitude, lonsearch[1])
ihi, _ = nearest_index(longitude, lonsearch[2])
ic0, _ = nearest_index(longitude, loncenter)
lonvec = longitude[ilo:ihi]

range1_west = fill(NaN, nlat)
range1_east = fill(NaN, nlat)
range2_west = fill(NaN, nlat)
range2_east = fill(NaN, nlat)

for k in 1:nlat
    js, _ = nearest_index(latitude, latsels[k])
    Tb = ds["thetao"][ilo:ihi,js,:,1]
    n = size(Tb,1)
    bottom = fill(NaN, n)
    island = fill(true, n)
    for c in 1:n
        igood = findall(!ismissing, Tb[c,:])
        if !isempty(igood)
            island[c] = false
            bottom[c] = depth[igood[end]]
        end
    end

    if haskey(range1_overrides, latsels[k])
        range1_west[k], range1_east[k] = range1_overrides[latsels[k]]
        iw1 = nearest_index(lonvec, range1_west[k])[1]
        ie1 = nearest_index(lonvec, range1_east[k])[1]
    else
        ic = ic0 - ilo + 1
        iw1 = ic
        while iw1 > 1 && !island[iw1-1]
            iw1 -= 1
        end
        ie1 = ic
        while ie1 < n && !island[ie1+1]
            ie1 += 1
        end
        range1_west[k] = lonvec[iw1]
        range1_east[k] = lonvec[ie1]
    end

    # Range 2: walk inward from each Range-1 coast until sea floor > 500 m
    iw2 = iw1
    while iw2 < n && (island[iw2] || bottom[iw2] <= shelf_threshold)
        iw2 += 1
    end
    ie2 = ie1
    while ie2 > 1 && (island[ie2] || bottom[ie2] <= shelf_threshold)
        ie2 -= 1
    end
    range2_west[k] = lonvec[iw2]
    range2_east[k] = lonvec[ie2]

    println("lat=",latsels[k]," Range1=[",range1_west[k],",",range1_east[k],
            "]  Range2=[",range2_west[k],",",range2_east[k],"]")
end

close(ds)

# ---------------------------------------------------------------------------
# per-column monthly N2 -> annual mean per column -> zonal mean, for each lat

depth_mid = (depth[1:end-1] .+ depth[2:end]) ./ 2
N2_zonalmean = fill(NaN, nz-1, nlat)
ncolumns     = zeros(Int, nlat)

for k in 1:nlat
    js, _ = nearest_index(latitude, latsels[k])
    i2w, _ = nearest_index(longitude, range2_west[k])
    i2e, _ = nearest_index(longitude, range2_east[k])
    ncol = i2e - i2w + 1

    N2_monthly_col = fill(NaN, nz-1, ncol, nmonths)   # per column, per month

    for m in 1:nmonths
        fname = @sprintf("mercatorglorys12v1_gl12_mean_1993_2016_%02i.nc", m)
        dsm = NCDataset(string(dirin,fname),"r")
        Tblock = dsm["thetao"][i2w:i2e,js,:,1]
        Sblock = dsm["so"][i2w:i2e,js,:,1]
        close(dsm)

        for c in 1:ncol
            Ts = Tblock[c,:]; Ss = Sblock[c,:]
            igood = findall(!ismissing, Ts .+ Ss)
            length(igood) < 2 && continue

            Tsg = Float64.(Ts[igood]); Ssg = Float64.(Ss[igood]); dpk = depth[igood]
            nzk = length(dpk)

            p  = gsw_p_from_z.(-dpk, latsels[k])
            SA = gsw_sa_from_sp.(Ssg, p, longitude[i2w+c-1], latsels[k])
            CT = gsw_ct_from_pt.(SA, Tsg)

            N2k = zeros(nzk-1); Pmidk = zeros(nzk-1)
            Lats = fill(latsels[k], nzk)
            gsw_nsquared(SA, CT, p, Lats, nzk, N2k, Pmidk)

            N2_monthly_col[1:nzk-1,c,m] = N2k
        end
    end

    # annual mean per column (average N2 across the 12 months, NaN-aware)
    N2_annual_col = fill(NaN, nz-1, ncol)
    for c in 1:ncol, z in 1:nz-1
        vals = filter(!isnan, N2_monthly_col[z,c,:])
        !isempty(vals) && (N2_annual_col[z,c] = mean(vals))
    end

    # zonal mean across columns (NaN-aware per depth level)
    kept = count(c -> any(!isnan, N2_annual_col[:,c]), 1:ncol)
    ncolumns[k] = kept
    for z in 1:nz-1
        vals = filter(!isnan, N2_annual_col[z,:])
        !isempty(vals) && (N2_zonalmean[z,k] = mean(vals))
    end

    println("lat = ", latsels[k], ": ", kept, " of ", ncol, " Range-2 columns had any valid data")
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
v.attrib["units"] = "s-2"; v.attrib["long_name"] = "Zonal mean of per-column annual-mean-of-monthly N2, over Range 2 (off-shelf, >500m) columns"

v = defVar(dsout,"ncolumns",Int32,("latitude",)); v[:] = ncolumns
v.attrib["long_name"] = "Number of Range-2 columns averaged"

v = defVar(dsout,"range1_west",Float64,("latitude",)); v[:] = range1_west
v = defVar(dsout,"range1_east",Float64,("latitude",)); v[:] = range1_east
v = defVar(dsout,"range2_west",Float64,("latitude",)); v[:] = range2_west
v = defVar(dsout,"range2_east",Float64,("latitude",)); v[:] = range2_east

dsout.attrib["source"] = string(dirin,"mercatorglorys12v1_gl12_mean_1993_2016_MM.nc (monthly), ",dirin,fnameannual," (Range1/2 masks)")
dsout.attrib["description"] = "Mercator N2 (TEOS-10, potential T -> gsw_ct_from_pt): per-column monthly N2 -> annual mean per column -> zonal mean over Range2 (coastline-anchored, off-shelf >500m) columns"

close(dsout)
println(fnameout," saved ........ ")
