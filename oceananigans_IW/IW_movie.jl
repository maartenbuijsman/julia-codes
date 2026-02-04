#= IW_movie.jl
Maarten Buijsman, USM DMS, 2025-9-5
Load model runs make a movie
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using Printf
using ColorSchemes

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirmovie = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\movies\\";  
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    dirsim  = "/data3/mbui/ModelOutput/IW/"
    dirmovie = "/data3/mbui/ModelOutput/movies/"
end

include(string(pathname,"include_functions.jl"))


# load and store files ===========================================
clims  = (-0.3,0.3)

#= function of latitude
#fnames = "AMZ3_40.0_hvis_12d_U1_0.40_U2_0.00.nc"; movienm = "mode 1"  # mode 2

# nonhydrostatic at 200 m
#fnames = "AMZ4_00.0_hvis_12d_U1_0.40_U2_0.00.nc"; movienm = "mode 1"  # mode 1 
fnames = "AMZ4_00.0_hvis_12d_U1_0.40_U2_0.30_v2.nc"; movienm = "mode12"  # mode 1 + 2
movienm2 = fnames[1:33]

filename = string(dirsim,fnames)
=#

oldnm   = 1  # before changing to numbered runs; https://docs.google.com/spreadsheets/d/1Qdaa95_I1ESBgkNMpJ9l8Vjzy4fuHMl2n6oIUELLi_A/edit?usp=sharing

#      38 39 40 41 42 43 44 45 46 47 48    49
LATS = [0 2.5 5 10 15 20 25 30 40 50 28.80 35]


# file name ===========================================

if oldnm==1
    # function of latitude
    lat = 0

   #fnames = @sprintf("AMZv_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); movienm = "mode 1"
   #fnames = @sprintf("AMZ3_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); movienm = "mode 1"  # hydro
   #fnames = @sprintf("AMZ4_%04.1f_hvis_12d_U1_0.40_U2_0.00.nc",lat); movienm = "mode 1"     # nonhydro

   fnames = "AMZ4_00.0_hvis_12d_U1_0.40_U2_0.30_v3.nc"; movienm = "mode 1+2"  # nonhydro and compare with
   # fnames = "AMZ4km_00.0_hvis_12d_U1_0.40_U2_0.30.nc"; movienm = "mode 1+2"  # hydro

    fname_short2 = fnames[1:33]
    filename = string(dirsim,fnames)

    LAT = LATS[1];
else
    # file ID
    mainnm = 1
    runnm  = 46

    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")
    movienm = "mode 1";

    LAT = LATS[runnm-37];
    #println("lat is ",LAT) 

end


# open nc file =============================================

ds = NCDataset(filename,"r");

tsec = ds["time"];
tday = tsec/24/3600;
dt = tday[2]-tday[1]

xf   = ds["x_faa"]; 
xc   = ds["x_caa"]; 
zc   = ds["z_aac"]; 

dx   = ds["Δx_caa"];
dz   = ds["Δz_aac"];

H  = sum(dz);   # depth

#const Nb = 0.005;     # buoyancy freq
const T2 = 12+25.2/60

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# u, v, w velocities
# NOTE: in future select a certain x range away from boundaries

# 
uf = ds["u"]; #x_faa × z_aac × time
wf = ds["w"]; #x_caa × z_aaf × time

# compute at cell centers
# v is already at x,W centers
uc = uf[1:end-1,:,:]/2 + uf[2:end,:,:]/2; 
wc = wf[:,1:end-1,:]/2 + wf[:,2:end,:]/2; 

#=
fig1 = Figure()
axa = Axis(fig1[1, 1])
hm = heatmap!(axa, xc/1e3, zc, uc[:,:,end], colormap = Reverse(:Spectral), 
              colorrange = (-0.05,0.05)); Colorbar(fig1[1,2], hm)
fig1  
=#

# interpolate in between frames
# aa  |    |  bb
#fadd = 3
fadd = 0      # no interpolation
ucnew = zeros(Nx,Nz,(Nt-1)*(fadd+1)+1)
k=0
for i in 1:Nt-1
#for i in 1:3
    k=k+1
    aa = copy(uc[:,:,i])
    bb = copy(uc[:,:,i+1])   
    ucnew[:,:,k] = copy(aa)
            println(k," i= ",i)

    for j in 1:fadd
        k=k+1
        println(k," ",j)
        ucnew[:,:,k] = aa + (bb-aa)/(fadd+1)
    end
end
k=k+1
ucnew[:,:,k] = copy(uc[:,:,end]);

# 1. Initialize Figure and Axis
fig = Figure(size = (1000,250))
ax = Axis(fig[1, 1], xlabel="x [km]", ylabel="z [m]")

# 2. Create an Observable for your heatmap data
# This allows you to efficiently update the plot without redrawing everything
initial_data = ucnew[:,:,1] # Example initial data
heatmap_data = Observable(initial_data)

# 3. Plot the heatmap using the Observable data :balance :RdBu_11
#hm=heatmap!(ax, xc/1e3, zc, heatmap_data, colormap = Reverse(:Spectral), colorrange = clims) # Customize colormap as needed
hm=heatmap!(ax, xc/1e3, zc, heatmap_data, colormap = Reverse(:RdBu_11), colorrange = clims) # Customize colormap as needed
Colorbar(fig[1, 2], hm)

# 4. Define the animation parameters
frames = 2:(size(ucnew)[3]) # Number of frames in the animation
framerate = 60 # Frames per second

# 5. Record the animation
record(fig, string(dirmovie,fname_short2,".mp4"), frames; 
framerate = framerate) do frame_num
    # Update the data for each frame
    heatmap_data[] = ucnew[:,:,frame_num]

    # You can also change other plot attributes here, like the title
    str = @sprintf("%5.2f days",frame_num*dt/(fadd+1));
    ax.title[] = string("sim. ",movienm,"; u velocity [m/s]; ",str)
    println("frame ",frame_num)
end

fig


