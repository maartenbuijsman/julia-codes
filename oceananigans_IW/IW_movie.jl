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

pathname = "/home/mbui/Documents/julia-codes/functions/"
include(string(pathname,"include_functions.jl"))

# load variables ===========================================
pathin  = "/data3/mbui/ModelOutput/IW/"

fnames = "AMZ1_lat0_2d_mode1_U10.05.nc"
#fnames = "AMZ1_lat0_8d_mode1_2_U10.05.nc"

filename = string(pathin,fnames)

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


fig1 = Figure()
axa = Axis(fig1[1, 1])
hm = heatmap!(axa, xc/1e3, zc, uc[:,:,end], colormap = :Spectral, colorrange = (-0.05,0.05)); Colorbar(fig1[1,2], hm)
fig1  


# 1. Initialize Figure and Axis
fig = Figure()
ax = Axis(fig[1, 1], title = "Amazon mode 1")

# 2. Create an Observable for your heatmap data
# This allows you to efficiently update the plot without redrawing everything
initial_data = uc[:,:,1] # Example initial data
heatmap_data = Observable(initial_data)

# 3. Plot the heatmap using the Observable data
heatmap!(ax, xc/1e3, zc, heatmap_data, colormap = :Spectral, colorrange = (-0.05,0.05)) # Customize colormap as needed
Colorbar(fig[1, 2], hm)

# 4. Define the animation parameters
frames = 2:(size(uc)[3]) # Number of frames in the animation
framerate = 4 # Frames per second

# 5. Record the animation
record(fig, "/data3/mbui/ModelOutput/movies/AMZ_mode1.mp4", frames; 
framerate = framerate) do frame_num
    # Update the data for each frame
    # For example, simulate some change in the heatmap
    #new_data = heatmap_data[] .+ randn(10, 10) * 0.1 # Add some noise
    #heatmap_data[] = clamp.(new_data, 0, 1) # Clamp values to a range (e.g., 0 to 1)
    heatmap_data[] = uc[:,:,frame_num]
    # You can also change other plot attributes here, like the title
    ax.title[] = "Frame $(frame_num)"
    println("frame ",frame_num)
end

fig


