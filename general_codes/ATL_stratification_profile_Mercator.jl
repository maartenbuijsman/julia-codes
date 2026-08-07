# ATL_stratification_profile_Mercator.jl
# MCB, USM, 2026-08-06
# Load MERCATOR N2 profile and then WKB scale it
# based on AMZ_stratification_profile.jl

#pathname = "C://Users//w944461//Documents//JULIA//functions/";
pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using CairoMakie
using NCDatasets
using GibbsSeaWater
using Interpolations
using Trapz
using JLD2
using Printf 

Threads.nthreads()

# load MERCATOR dataset
filename = "/home/mbui/ModelOutput/IW/mercator/MERC_N2_zonalmean_Atl_offshelf_monthly.nc"
ds = NCDataset(filename,"r");

zmid  = -1 .* ds["depth_mid"][:];
N2b   = ds["N2_zonalmean"][:,2];
lat   = ds["latitude"][2];
close(ds)

# check if neg N2?
N2b = coalesce.(N2b, NaN) # missing -> NaN
findall(N2b .< 0) 


# simple plot
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax,N2b,zmid)
#ylims!(ax2, -100, 0)
fig


# range is < mindepth
mindepth = -4000;
Iz = findall(item -> item > mindepth, zmid) 

# create final vectors
zz  = [0; zmid[Iz]; mindepth]
N2c = [N2b[Iz[1]]; N2b[Iz]; N2b[Iz[end]]]; #use nearest neighbor to fill in surf/bottom values
#N2c = [0; N2b[Iz]; N2b[Iz[end]]] 

zeroval = 1e-12
Iz = findall(item -> item ==0, N2c) 
N2c[Iz] .= zeroval 

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,N2c,zz)
ylims!(ax1, -200, 0)
fig

# make sure all values are from bottom to surface
# to comply with Oceananigans

zf = zz;
N2 = N2c;
Nzf = length(zf);

flipped = zf[1] < zf[end]  # if false, input is surface to bottom

# !flipped means if flipped is true, then !flipped is not true
if !flipped
    zf = reverse(zf)
    N2 = reverse(N2)
end

zc = zf[1:end-1]/2 + zf[2:end]/2;

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,N2,zf)
ylims!(ax1, -200, 0)
fig



# ==============================================================
# WKB scale z 
# pick a constant WKB scaled dz and convert it back
# this will be used for getting a scaled dz

# depth-mean N 
H = abs(zf[1]);
Nave = trapz(zf,sqrt.(N2))/H

# WKB scaled z
# integrate from surface to bottom
zwkb = zeros(size(zf));
for i in Nzf-1:-1:1
    # reverse integrate, hence omit -
    zwkb[i] = trapz(zf[Nzf:-1:i],sqrt.(N2[Nzf:-1:i]))/Nave
    #println(zf[i])
end

dzkwb = diff(zwkb)
# scatter(zwkb,zf)
scatter(dzkwb,zc)

# now interpolate equidistant dzwkb
nzWKB = 110;
dzwkb2 = H/nzWKB
zwkb2 = collect(range(-H, 0, length=nzWKB+1))
#zwkb2 = collect(-H:dzwkb2:0)
#zwkb2 = vcat(collect(-H:dzwkb2:-40), collect(-35:5:0))

# extract new z values
#itz = interpolate((zwkb[end:-1:1],), zf[end:-1:1], Gridded(Linear()))
#zfw = itz.(zwkb2[2:end-1])

interp_linextr = linear_interpolation(zwkb, zf, extrapolation_bc=Line())
zfd = interp_linextr.(zwkb2)

zcd = zfd[1:end-1]/2 + zfd[2:end]/2;
dzd = diff(zfd)

fig9 = Figure()
ax9 = Axis(fig9[1,1])
scatter!(ax9,dzd,zcd) # plot org dz
ylims!(ax9, -500, 0)
xlims!(ax9, 0, 40)
fig9

# fix the dz near the surface
dzminfix = 5;

# index of smallest dz
dzmin, Imin = findmin(dzd)  

# find deeper index, which dz is larger than dzminfix
Iminfix = findlast(>(dzminfix), dzd[1:Imin])

# now add this dzminfix to top
len = 1 + Int(ceil(abs(zfd[Iminfix])/dzminfix))
zfdadd = collect(range(zfd[Iminfix], 0, len))

#dzmin, Imin = findmin(dzd)
#zfdadd = collect( range(zfd[Imin],0,length=Int(ceil(abs(zfd[Imin])/dzmin))) )
#zfw = vcat(zfd[1:Imin],zfdadd[2:end])

zfw = vcat(zfd[1:Iminfix],zfdadd[2:end])
zcw = zfw[1:end-1]/2 + zfw[2:end]/2;
dzw = diff(zfw)
nzw = length(dzw)

sum(dzw)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,zwkb, zf)
scatter!(ax1,zwkb2,zfd,color=:red)
#ylims!(ax1, -500, 0)
#xlims!(ax1, -2000, 10)
fig

fig9 = Figure()
ax9 = Axis(fig9[1,1])
scatter!(ax9,dzw,zcw) # plot limited dz
#ylims!(ax9, -500, 0)
#xlims!(ax9, 0, 40)
fig9

# interpolate N2 to the new zfw
# then compute eigenfunctions
# these do not need rescaling :-) because of interpolation
intzc = linear_interpolation(zf, N2, extrapolation_bc=Line())
N2w = intzc.(zfw)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,N2, zf)
scatter!(ax1,N2w,zfw,color=:red)
ylims!(ax1, -200, 0)
#xlims!(ax1, -2000, 10)
fig

println("number of WKB faces is ", length(zfw))
println("min/max thickness is ", @sprintf("%.2f", minimum(dzw)), "/", @sprintf("%.2f", maximum(dzw)))


# =============================================================================
# save the stratification 
# and N2w and zfw
# then load in Oceananigans

#dirout = "/data3/mbui/ModelOutput/IW/forcingfiles/"
fnameAZ = "N2_amz1.jld2"

#jldsave(string(dirout,fnameAZ); N2w, zfw, lonsel, latsel);
#println(string(fnameAZ)," data saved ........ ")



