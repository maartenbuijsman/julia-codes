# ATL_stratification_profile.jl
# MCB, USM, 2025-9-8
# based on AMZ_stratification_profile.jl
# extracting a WOCE profile at different latitudes

#pathname = "C://Users//w944461//Documents//JULIA//functions/";
pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

using CairoMakie
using NCDatasets
using GibbsSeaWater
using Interpolations
using Trapz
using JLD2

Threads.nthreads()

# load Gregg Jacobs' data set
filename = "/data3/mbui/ModelOutput/IW/stratification/Maarten_NATL_mean_TS.nc"
ds = NCDataset(filename,"r");

depth = ds["depth"][:];
latitude = ds["latitude"][:];
longitude = ds["longitude"][:];
mean_temp = ds["mean_temp"][:,:,:,:]; # lon 41 × lat 31 × depth 99 × month 12
mean_salt = ds["mean_salt"][:,:,:,:];
std_salt = ds["std_salt"][:,:,:,:];
std_temp = ds["std_temp"][:,:,:,:];

# replace 
mean_temp[mean_temp .> 1e10] .= NaN
mean_salt[mean_salt .> 1e10] .= NaN
SST = mean_temp[:,:,1,1]
SSS = mean_salt[:,:,1,1]

fig1 = Figure(size = (600, 800))
axa = Axis(fig1[1, 1],title="SST")
hma = heatmap!(axa, longitude, latitude, SST, colormap = (:Spectral)); Colorbar(fig1[1,2], hma); 

axb = Axis(fig1[2, 1],title="SSS")
hmb = heatmap!(axb, longitude, latitude, SSS, colormap = (:Spectral)); Colorbar(fig1[1,2], hmb); 
fig1    


# plot profiles
LATS = [10 20 30 40 50]

lonsel = 325; latsel = LATS[1]    # deep cast 4000 m
#lonsel = 300; latsel = LATS[1]    # deep cast 4000 m
d,is = findmin(abs.(longitude.-lonsel));
d,js = findmin(abs.(latitude.-latsel));

# transect
# skipmissing
Tt = mean_temp[is,:,:,:]
St = mean_salt[is,:,:,:]

Tta = dropdims(mean(Tt,dims=3),dims=3)
Sta = dropdims(mean(St,dims=3),dims=3)

Tta2 = zeros(length(latitude),length(depth));
Sta2 = zeros(length(latitude),length(depth));
for i=1:length(latitude)
    for j=1:length(depth)
       Tta2[i,j] = mean(skipmissing(Tt[i,j,:]))         
       Sta2[i,j] = mean(skipmissing(St[i,j,:]))                
    end
end

fig1 = Figure(size = (600, 800))
axa = Axis(fig1[1, 1],title="SST")
hma = heatmap!(axa, latitude, -depth, Tta2, colormap = (:Spectral)); Colorbar(fig1[1,2], hma); 

axb = Axis(fig1[2, 1],title="SSS")
hmb = heatmap!(axb, latitude, -depth, Sta2, colormap = (:Spectral)); Colorbar(fig1[2,2], hmb); 
fig1    


Ta = mean(mean_temp[is,js,:,:],dims = 4)
Sa = mean(mean_salt[is,js,:,:],dims = 4)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Ta,-depth)

ax2 = Axis(fig[1,2])
lines!(ax2,Sa,-depth)
fig


double_depth1 = ds["double_depth1"][:];
TSmean = ds["mean"];

Nz = length(depth);

LON, LAT = meshgrid(longitude,latitude);

#=
fig = Figure()
ax = Axis(fig[1,1])
scatter!(LON[:],LAT[:])
scatter!(ax,LON[is,js],LAT[is,js], marker = '*', markersize = 50)
fig
=#

lonsel = 360-42; latsel = 7    # deep cast 4000 m
#lonsel = 360-45; latsel = 3   # shallow cast 2000 m
d,is = findmin(abs.(longitude.-lonsel));
d,js = findmin(abs.(latitude.-latsel));

lonsel = LON[is,js]
latsel = LAT[is,js]

TSsel = TSmean[:,is,js];
Ts = TSsel[1:Nz];
Ss = TSsel[Nz+1:end];
depths = ds["depth"][:]; 

# check for bad values and remove them
threshold = 1e4;
ibad = [i for i in eachindex(Ts) if Ts[i] > threshold]
if ~isempty(ibad)
    deleteat!(Ts, ibad)
    deleteat!(Ss, ibad)
    deleteat!(depths, ibad)
end

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Ts,-depths)

ax2 = Axis(fig[1,2])
lines!(ax2,Ss,-depths)
fig

Nz2 = length(depths);

# compute N2 using TEOS10
p  = gsw_p_from_z.(-depths,latsel)
SA = gsw_sa_from_sp.(Ss, p, lonsel, latsel)
CT = gsw_ct_from_t.(SA,Ts,p)
sig2 = gsw_sigma2.(SA,CT)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,CT,-p)

ax2 = Axis(fig[1,2])
lines!(ax2,SA,-p)
fig

N2   = zeros(Nz2-1) # Array for output of Brunt-Vaisalla values @ Pmid) 
Pmid = zeros(Nz2-1) # Array for outputs of pressure for N2
Lats = fill(latsel, Nz2) 
gsw_nsquared(SA, CT, p, Lats, Nz2, N2, Pmid)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,sig2,-p)
ylims!(ax1, -100, 0)

ax2 = Axis(fig[1,2])
lines!(ax2,N2,-Pmid)
ylims!(ax2, -100, 0)
fig

# find bad data points
N2b = N2
ibad = [i for i in eachindex(N2b) if N2b[i] < 0]
ii = ibad[end]
N2b[1:ii] .= 0.0

lines!(ax2,N2b,-Pmid,color=:red)
fig

# new vectors for EVP
itp = interpolate((p,), -depths, Gridded(Linear()))
zmid = itp.(Pmid)

# range is < mindepth
mindepth = -4000;
Iz = findall(item -> item > mindepth, zmid) 

# create final vectors
zz  = [0; zmid[Iz]; mindepth]
N2c = [0; N2b[Iz]; N2b[Iz[end]]] 

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
dzwkb2 = 40;
zwkb2 = collect(-H:dzwkb2:0)
#zwkb2 = vcat(collect(-H:dzwkb2:-40), collect(-35:5:0))

# extract new z values
#itz = interpolate((zwkb[end:-1:1],), zf[end:-1:1], Gridded(Linear()))
#zfw = itz.(zwkb2[2:end-1])

interp_linextr = linear_interpolation(zwkb, zf, extrapolation_bc=Line())
zfd = interp_linextr.(zwkb2)

zcd = zfd[1:end-1]/2 + zfd[2:end]/2;
dzd = diff(zfd)

# fix the dz near the surface
dzmin, Imin = findmin(dzd)
zfdadd = collect(range(zfd[Imin],0,length=Int(ceil(abs(zfd[Imin])/dzmin))))

zfw = vcat(zfd[1:Imin],zfdadd[2:end])
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

fig = Figure()
ax1 = Axis(fig[1,1])
scatter!(dzw,zcw)
ylims!(ax1, -500, 0)
xlims!(ax1, 0, 40)
fig

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

# =============================================================================
# save the stratification 
# and N2w and zfw
# then load in Oceananigans

dirout = "/data3/mbui/ModelOutput/IW/forcingfiles/"
fnameAZ = "N2_amz1.jld2"

jldsave(string(dirout,fnameAZ); N2w, zfw, lonsel, latsel);
println(string(fnameAZ)," data saved ........ ")



