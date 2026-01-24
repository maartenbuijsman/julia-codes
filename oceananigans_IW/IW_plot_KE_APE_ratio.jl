#= IW_plot_KE_APE_ratio.jl
Maarten Buijsman, USM DMS, 2026-1-23
load the KE and APE data from IW_total_energetics.jl
to plot KE+APE in tidal and supertidal band as a function of latitude
also look into the ratio KE/APE = (om2+f2)/(om2-f2) for D2 and HH
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
    dirEIG = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    dirsim = "/data3/mbui/ModelOutput/IW/";
    dirfig = "/data3/mbui/ModelOutput/figs/";
    dirout = "/data3/mbui/ModelOutput/diagout/";
    dirEIG = "/data3/mbui/ModelOutput/IW/forcingfiles/";
end

include(string(pathname,"include_functions.jl"))

# print figures
figflag = 1
const T2 = 12+25.2/60
om2 = 2*pi/(T2*3600)

#         38 39  40 41 42 43 44 48    45 49 46 47    
LATS   = [0  2.5 5  10 15 20 25 28.80 30 35 40 50 ]
runnms = [38 39  40 41 42 43 44 48    45 49 46 47];

#runnms = [38]


LAT = Float64[];
KED2a = Float64[]; APED2a = Float64[];
KEha = Float64[];  APEha = Float64[];

for ii=1:length(runnms)
    runnm = runnms[ii]
    println("runnm is ",runnm,"; lat =",LATS[ii]) 

    # file name ===========================================

    # file ID
    mainnm = 1

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")

    #LAT[ii] = LATS[runnm-37];
    LAT = push!(LAT,LATS[ii]);

    # load simulations ===========================================

    # load xc, Fxt, Fx, Fxh, Fxl, Fx2, KEt, APEt, KE, APE, KEh, APEh, KEl, APEl, KE2, KEut, KEu, KEuh, KEul);
    # KEt is total, unfiltered
    # KE  is D2+HH
    # KEh is HH
    # KEl is low-passed subtidal, compare with KEll  

    fnamein = string(dirout,"energetics_",fname_short2,".jld2")

    #= Open the JLD2 file for displaying variables
    gridfile = jldopen(fnamein, "r")
    println(keys(gridfile))  # List the keys (variables) in the file
    close(gridfile)
    =#

    # this is a quick load 
    @load fnamein xc KEt APEt KEh APEh KEl APEl

    # KED2 = KEl
    KED2  = KEl
    APED2 = APEl

    # average coefficients fall inside 75-500 km range (exc. forcing and sponge))
    xlims = [75,500]*1e3;
    Plims = [-12 1];
    Ix = findall(item -> item >= xlims[1] && item<= xlims[2], xc);

    val1 = mean(KED2[Ix,],dims=1)
    val2 = mean(APED2[Ix,],dims=1)    
    KED2a  = push!(KED2a,val1[1]); 
    APED2a = push!(APED2a,val2[1]); 

    val1 = mean(KEh[Ix,],dims=1)
    val2 = mean(APEh[Ix,],dims=1)    
    KEha  = push!(KEha,val1[1]); 
    APEha = push!(APEha,val2[1]); 


end # runnms

# ratio KE HH/D2
fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,KED2a/1e3,LAT, linestyle=:solid, color = :red, linewidth = 3)
lines!(ax1,APED2a/1e3,LAT, linestyle=:solid, color = :blue, linewidth = 3)

lines!(ax1,KEha/1e3,LAT, linestyle=:dash, color = :red, linewidth = 3)
lines!(ax1,APEha/1e3,LAT, linestyle=:dash, color = :blue, linewidth = 3)

ax2 = Axis(fig[1,2])
lines!(ax2,KEha ./KED2a,LAT, linestyle=:solid, color = :red, linewidth = 3)
lines!(ax2,APEha ./APED2a,LAT, linestyle=:solid, color = :blue, linewidth = 3)
#lines!(ax1,(KEha .+ APEha) ./(KED2a .+ APED2a),LAT)

fig

# APE calculation is trick because the ratio's are
# not right ....
fcor = coriolis.(LAT)


fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,KED2a ./APED2a,LAT)
lines!(ax1,(om2^2 .+ fcor.^2)./(om2^2 .- fcor.^2),LAT)
fig

# not a constant value ......
#(KED2a./(APED2a)) ./ ((om2^2 .+ fcor.^2)./(om2^2 .- fcor.^2)) 

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,(om2^2 .+ fcor.^2)./(om2^2 .- fcor.^2),KED2a./APED2a)
lines!(ax1,KED2a./APED2a,KED2a./APED2a)
fig


stop()


################################################################

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2

figflag = 0
WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
    dirEIG = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    dirsim = "/data3/mbui/ModelOutput/IW/";
    dirfig = "/data3/mbui/ModelOutput/figs/";
    dirout = "/data3/mbui/ModelOutput/diagout/";
    dirEIG = "/data3/mbui/ModelOutput/IW/forcingfiles/";
end

include(string(pathname,"include_functions.jl"))

# load simulations ===========================================

# function of latitude
lats = [0, 2.5, 5, 10, 20, 25, 30, 40];

KEom = zeros(1000,length(lats));
KEuom = zeros(1000,length(lats));

#ii=1
lep=0; period=0; freqs=[];
for ii in 1:length(lats)

    lat = lats[ii]
    fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); 
    println("loading ",fnames)
    fname_short2 = fnames[1:33]

    filename = string(dirsim,fnames)

    T2 = 12+25.2/60

    ds = NCDataset(filename,"r");

    tsec = ds["time"][:];
    tday = tsec/24/3600;
    dt = tday[2]-tday[1]

    xf   = ds["x_faa"][:]; 
    xc   = ds["x_caa"][:]; 
    zc   = ds["z_aac"][:]; 

    dx   = ds["Δx_caa"][:];
    dz   = ds["Δz_aac"][:];

    H  = sum(dz);   # depth

    Nz = length(zc);
    Nx = length(xc);
    Nt = length(tday);

    # u, v, w velocities
    # NOTE: in future select a certain x range away from boundaries

    # a loop is faster than permuting?
    uf = zeros(Nt,Nx+1,Nz);
    vf = zeros(Nt,Nx,Nz);
    wf = zeros(Nt,Nx,Nz+1);

    for i in 1:Nt
        #println(i)
        uf[i,:,:] = ds["u"][:, :, i];
        vf[i,:,:] = ds["v"][:, :, i];
        wf[i,:,:] = ds["w"][:, :, i];
    end
    vc = vf; # all values at centers along x

    # close the nc file
    close(ds)

    # compute at cell centers
    # v is already at x,W centers
    uc = uf[:,1:end-1,:]/2 + uf[:,2:end,:]/2; 
    wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 

    # compute some ffts of surface velocity ======================================================

    Sp_Region_right = 20000
    Ix = findall(item -> item<=xc[end]+dx[1]/2-Sp_Region_right, xc)

    EXCL = 0;  # can be zero for fft
    t1,t2 = 4, tday[end]-EXCL*T2/24
    numcycles = floor((t2-t1)/(T2/24))
    t2 = t1+numcycles*(T2/24)
    Iday = findall(item -> item >= t1 && item<= t2, tday)

    tukeycf=0.2; numwin=1; linfit=true; prewhit=false;

    i=1;
    period, freq, pp = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit); #get the dimensions
    lep = length(period);
    poweru = zeros(lep,Nx);
    powerv = zeros(lep,Nx);
    for i in 1:Nx
        period, freqs, poweru[:,i] = fft_spectra(tday[Iday], uc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);
        period, freqs, powerv[:,i] = fft_spectra(tday[Iday], vc[Iday,i,end]; tukeycf, numwin, linfit, prewhit);    
    end

    # average along x
    KEom[1:lep,ii] = mean(poweru[:,Ix] .+ powerv[:,Ix],dims=2);
    KEuom[1:lep,ii] = mean(poweru[:,Ix],dims=2);   
   
end

KEom  = copy(KEom[1:lep,:]);
KEuom = copy(KEuom[1:lep,:]);


# isolate D2 and HH frequencies and compute KE
Id2 = findall(item -> item>=10/24 && item<=14/24, period)
Ihh = findall(item -> item<=9/24, period)

# do the sum (need to multiply x df)
df = freqs[2]-freqs[1]

KEd2 = sum(KEom[Id2,:],dims=1)
KEhh = sum(KEom[Ihh,:],dims=1)

KEud2 = sum(KEuom[Id2,:],dims=1)
KEuhh = sum(KEuom[Ihh,:],dims=1)

fig1 = Figure()
axa = Axis(fig1[1, 1], title="a) latitude vs KE",
xlabel="KE [m2/s2]",ylabel="latitude [degrees]");  
scatterlines!(axa,vec(KEd2),lats,label="D2",color=:blue,linestyle=:solid,linewidth=2)
scatterlines!(axa,vec(KEhh),lats,label="HH",color=:red,linestyle=:solid,linewidth=2)
scatterlines!(axa,vec(KEud2),lats,label="D2 u",marker=:cross,color=:dodgerblue,linestyle=:dash,linewidth=2)
scatterlines!(axa,vec(KEuhh),lats,label="HH u",marker=:cross,color=:orangered,linestyle=:dash,linewidth=2)
axislegend(axa, position = :rb)

axb = Axis(fig1[1, 2], title="b) latitude vs KE D2/HH ratio",
xlabel=string("KE ratio HH/D2"))
scatterlines!(axb,vec(KEhh)./vec(KEd2),lats,label="HH/D2",color=:red,linestyle=:solid,linewidth=2)
scatterlines!(axb,vec(KEuhh)./vec(KEud2),lats,label="HH/D2 u",marker=:cross,color=:orangered,linestyle=:dash,linewidth=2)
axislegend(axb, position = :lb)
fig1

#save(string(dirfig,"KE_ratio_usur.png"), fig1)

# spectra
xlim = [0 11];
ylim = [1e-8 1e0];

fig1 = Figure()
axa = Axis(fig1[1, 1],#yticks = (0:2:10),
title=string("log10 KE [m2/s2] ",tistr),
ylabel="Power [m2/s2]",xlabel="frequency [cpd]",yscale = log10);  

for i in 1:length(lats)
    lines!(axa,freqs,(KEom[:,i]),label=@sprintf("%04.1f",lats[i]))
end
axislegend(axa, position = :lb)
xlims!(axa, xlim[1], xlim[2])
ylims!(axa, ylim[1], ylim[2])
fig1

# heatmap of spectral power
ylim = [0 11];
clims = (-0.05,0.05)

#tistr = " mode 1 + 2"
tistr = " mode 1"

fig1 = Figure()
axa = Axis(fig1[1, 1],yticks = (0:2:10),title=string("log10 KE [m2/s2] ",tistr),xlabel="x [km]",ylabel="frequency [cpd]");  
ylims!(axa, ylim[1], ylim[2])
hm = heatmap!(axa, xc/1e3, freq, log10.(transpose(KEom)),colormap = Reverse(:Spectral)); 
Colorbar(fig1[1,2], hm); 
hm.colorrange = (-6, 0)
fig1   

# Save the figure as a PNG file
if figflag==1; save(string(dirfig,"fft_usur_",fname_short2,".png"), fig1)
end
##=#

