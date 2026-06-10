#= IW_pressure_soliton.jl
Maarten Buijsman, USM DMS, 2026-1-24
zoom in on soliton; plot ssh
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2
using ColorSchemes
using LaTeXStrings
using Interpolations


WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirfig = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\figs\\";  
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";  
    dirEIG = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirfig = string(pth0,"figs/");
    dirout = string(pth0,"diagout/");
    dirforce = string(pth0,"IW/forcingfiles/");
    dirEIG = string(pth0,"IW/forcingfiles/");
end

include(string(pathname,"include_functions.jl"))

# print figures
figflag = 0
const T2 = 12+25.2/60
const rho0=1020; 
const grav=9.81; 

# only select data after the spinup time 
# => i.e., time it takes before waves reach the eastern boundary
const tspin = 4; #days

# run names --------------------------------
# D2, mode 1 + 2 interactions
LATS    = [0, 0, 0, 0];
mainnms = [6, 8, 7, 8]; #4km-nh, 4k-h, 200m-nh, 200m-k
runnms  = [1, 1, 1, 2];

titstrs = ["4 km nonhyd.","4 km hyd.","200 m nonhyd.","200 m hyd."]

## plotting function  -----------------------------

# do the analysis in a function
ax = function run_figs(mainnm,runnm,LAT,titstr,figcount)
    #IS = 1; mainnm = mainnms[IS]; runnm = runnms[IS]; LAT = LATS[IS]

    # filename
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 
    fname_short2 = fnames
    filename = string(dirsim,fnames,".nc")

    println(fname_short2,"; lat=",LAT," -------------------") 


    # load simulations ===========================================
    ds = NCDataset(filename,"r");
    #println(ds)
    println(keys(ds))

    # select Indices after spinup
    tday0 = ds["time"][:]/24/3600;
    Isel = findall(>=(tspin),tday0);

    tsec = ds["time"][Isel];
    tday = tsec/24/3600;
    dt = tday[2]-tday[1]

    #lines(Isel,tday[Isel])
    xf   = ds["x_faa"][:]; 
    xc   = ds["x_caa"][:]; 
    zc   = ds["z_aac"][:]; 

    dx   = ds["Δx_caa"][:];
    dz   = ds["Δz_aac"][:];
    Ldom = sum(dx);

    H  = sum(dz);   # depth

    Nz = length(zc);
    Nx = length(xc);
    Nt = length(tday);

    # u, v, w velocities
    # NOTE: in future select a certain x range away from boundaries
    # this is much faster than loading all data in Isel
    It = Isel[462];

    @time begin
        println("reading nc file ",filename)
        uf  = ds["u"][:,:,It];
        bc  = ds["b"][:,:,It];

        #=uf  = permutedims(ds["u"][:,:,Isel],    (3,1,2));
        vc  = permutedims(ds["v"][:,:,Isel],    (3,1,2));
#        wf  = permutedims(ds["w"][:,:,Isel],    (3,1,2));
        bc  = permutedims(ds["b"][:,:,Isel],    (3,1,2));
        #pHY = permutedims(ds["pHY"][:,:,Isel],  (3,1,2));
        #pNH = permutedims(ds["pNHS"][:,:,Isel], (3,1,2));
        =#
    end

    # close the nc file
    close(ds)

    # compute at cell centers
    # v is already at x,W centers
    uc = uf[1:end-1,:]/2 + uf[2:end,:]/2; 
#    wc = wf[:,:,1:end-1]/2 + wf[:,:,2:end]/2; 

    # clear variables from memory
    uf=nothing; wf=nothing; GC.gc()

    # load N2 profile -----------------------------------------------------------
    # load profile created by AMZ_stratification_profile.jl
    fnamegrid = "N2_amz1.jld2";
    path_fname = string(dirforce,fnamegrid);

    # variables loaded
    # "N2w", "zfw", "lonsel", "latsel"

    # Open the JLD2 file
    gridfile = jldopen(path_fname, "r")
    println(keys(gridfile))  # List the keys (variables) in the file
    close(gridfile)

    @load path_fname N2w zfw

    # map to cell centers
    N2c = N2w[1:end-1]/2 + N2w[2:end]/2;

    # ref and pert densities ----------------------------------------------------
    # compute reference density profile
    # b = sum N2 * dz = sum -g/rho0*drho/dz * dz
    # b = -g/rho0*rho_pert
    # rho_pert = -b*rho0/g 

    # bottom up!
    breff = cumtrapz(zfw, N2w);   # length Nz+1, on zfw grid

    # in the mod sims buoyancy is interpolated to cell centers
    intzc   = interpolate((zfw,), breff, Gridded(Linear()));
    rhorefc = -intzc.(zc) * rho0/grav;     # rho0 is not added!
    rr      = reshape(rhorefc, 1, :);
    rhop    = -bc * rho0/grav .+ rr;  

    rhoc = rhop .+ rho0;

    #=thresh = 1e-5;
    rrr = reshape(rhorefc,1,1,:);  
    APEz,Zz = APEKFeq2(rhop[:,:,:], rhorefc, zc, grav, thresh);
    =#

    ## figures

    # plot a heatmap of velocity and include density contours
    cmap = Reverse(:RdBu_9);
    clims  = (-0.3,0.3)
    xlim = (305,385); zlim = (-1500,0); 

    fnum = string(mainnm,".",runnm)

    ax = Axis(fig[figcount, 1],title=string(titstr," (",fnum,"); velocity [m/s]"), ylabel="z [m]",xticks = 310:10:380)#, xticks = 367:2:387)
    hm=heatmap!(ax, xc/1e3, zc, uc, colormap = cmap, colorrange = clims) # Customize colormap as needed

    my_levels = [1015, 1015.5, 1018.5, 1019, 1019.5, 1019.95]
    contour!(ax, xc/1e3, zc, rhoc; 
        levels = my_levels,  # Prescribes exact values for lines
        color = :black,      # Sets all lines to black
        labels = true,       # Optional: shows level values
        linewidth = 1
    )

    Colorbar(fig[1, 2], hm)
    xlims!(ax, xlim[1], xlim[2])
    ylims!(ax, zlim[1], zlim[2])
    return ax

#stop()
end # function run_figs(runnm,LAT)

# runnms loop ---------------

fig = Figure(size = (750,1000))

elapsed = @elapsed begin
    figcount=0
    for (mainnm,runnm,LAT,titstr) in zip(mainnms,runnms,LATS,titstrs)
        figcount = figcount + 1;
        ax = run_figs(mainnm,runnm,LAT,titstr,figcount)
    end 
end
println("finished in $(round(elapsed, digits=1)) s")

ax.xlabel = "x [km]"
display(fig)

save(string(dirfig,"four_snapshots.png"), fig)
