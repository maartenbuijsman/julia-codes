#= IW_single_snapshot.jl
Maarten Buijsman, USM DMS, 2026-08-20
Plot a snapshot per run (loop over runnms, 1 figure each):
heatmap of u velocity with superposed density (rhoc) contours.
Long transects are split into 1-4 stacked x-subplots (chunks).
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
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

# figure control
figflag  = 1    # 1 = create & display the figure(s)
saveflag = 1    # 1 = save figure(s) to dirfig as PNG
const T2 = 12+25.2/60
const rho0=1020;
const grav=9.81;

# only select data after the spinup time
const tspin = 4; #days

# run-ID selection: only mainnm + runnms need to be prescribed here; LAT and
# the N2 stratification profile are looked up from run_master.jl, so run-ID
# and latitude can never drift out of sync. runnms need not be a full block --
# any subset of run-IDs already present in RUN_TABLE works.

# D2 NH flux forcing
mainnm  = 10
#runnms  = collect(1:14) # constant N2 WOCE AMZ
runnms  = collect(27:39) # varying  N2 MERCATOR
#runnms  = collect(40:52) # constant N2 MERCATOR 2.5N
#runnms  = collect(53:65) # constant N2 MERCATOR 50N

# 200 m
mainnm  = 11
#runnms  = collect(27:32) # varying  N2 MERCATOR
runnms  = collect(66:78) # varying  N2 MERCATOR 50 kW/m

runs = get_runs(mainnm, runnms)   # errors immediately if a runnm isn't in RUN_TABLE
LATS = [r.lat for r in runs]

# snapshot / plotting options ----------------------------------------------
snap_which   = :last                       # :last => last output; or Integer index into post-spinup times
nsub         = 4                           # number of stacked x-subplots (1-4): short transects 1-2, long 4
clims        = (-0.3, 0.3)                 # u colour range [m/s]
zlim         = (-1500, 0)                  # depth range [m]
cmap         = Reverse(:RdBu_9)
level_depths = [100, 300, 500, 750, 1000, 1250]  # m; density contours drawn at ref-density of these depths


## plotting function -------------------------------------------------------
function run_snapshot(runnm, LAT)
    # IS = 1; runnm = runnms[IS]; LAT = LATS[IS]

    fnames   = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
    filename = string(dirsim,fnames,".nc")
    println(fnames,"; lat=",LAT," -------------------")

    # look up this run's metadata (lat/Flux/DX/N2 source) from the master table
    row = get_runs(mainnm, [runnm])[1]

    # load simulation ------------------------------------------------------
    ds = NCDataset(filename,"r");
    println(keys(ds))

    tday0 = ds["time"][:]/24/3600;
    Isel  = findall(>=(tspin),tday0);

    xc = ds["x_caa"][:];
    zc = ds["z_aac"][:];
    dx = ds["Δx_caa"][:];
    dz = ds["Δz_aac"][:];
    Ldom = sum(dx);

    # choose snapshot time (index into the full time dimension)
    It    = snap_which === :last ? Isel[end] : Isel[snap_which]
    tsnap = tday0[It];

    @time begin
        println("reading nc file ",filename," at day ",round(tsnap,digits=2))
        uf = ds["u"][:,:,It];
        bc = ds["b"][:,:,It];
    end
    close(ds)

    # u at cell centres
    uc = uf[1:end-1,:]/2 + uf[2:end,:]/2;
    uf = nothing; GC.gc()

    # reference + perturbation density -------------------------------------
    # N2 profile file matched to how the run was forced (looked up from run_master.jl)
    fnamegrid  = n2_filename(row)
    path_fname = string(dirforce,fnamegrid);
    isfile(path_fname) || error("N2 forcing file not found: ", path_fname)
    println("N2source = $(row.N2source), fnamegrid = $fnamegrid")
    @load path_fname N2w zfw

    # b = -g/rho0 * rho_pert ;  reference density from cumulative N2
    breff   = cumtrapz(zfw, N2w);                       # length Nz+1, on zfw grid
    intzc   = interpolate((zfw,), breff, Gridded(Linear()));
    rhorefc = -intzc.(zc) * rho0/grav;                  # reference density anomaly at cell centres
    rr      = reshape(rhorefc, 1, :);
    rhop    = -bc * rho0/grav .+ rr;
    rhoc    = rhop .+ rho0;

    # contour levels = reference density at the fixed depths (isopycnals ~ those depths)
    my_levels = sort([ -intzc(-d) * rho0/grav + rho0 for d in level_depths ]);

    # figure: nsub stacked x-chunks ---------------------------------------
    fnum   = string(mainnm,".",runnm)
    Ldomkm = Ldom/1e3
    edges  = range(0, Ldomkm, length=nsub+1)   # x-chunk boundaries [km]

    fig = Figure(size = (1100, 200*nsub + 120))
    local hm
    for s in 1:nsub
        ax = Axis(fig[s, 1], ylabel = "z [m]")
        hm = heatmap!(ax, xc/1e3, zc, uc, colormap = cmap, colorrange = clims)
        contour!(ax, xc/1e3, zc, rhoc;
            levels = my_levels, color = :black, labels = true, linewidth = 1)
        xlims!(ax, edges[s], edges[s+1])
        ylims!(ax, zlim[1], zlim[2])
        if s == 1
            ax.title = string(LAT,"°N (",fnum,"); u [m/s] + ρ contours; day ",round(tsnap,digits=1))
        end
        if s == nsub
            ax.xlabel = "x [km]"
        else
            ax.xticklabelsvisible = false
        end
    end
    Colorbar(fig[1:nsub, 2], hm, label = "u [m/s]")
    figflag == 1 && display(fig)

    if saveflag == 1
        fout = string(dirfig,"snapshot_u_rho_",fnames,".png")
        save(fout, fig)
        println("saved ",fout)
    end
    return fig
end # function run_snapshot


# runnms loop --------------------------------------------------------------
elapsed = @elapsed begin
    for (runnm, LAT) in zip(runnms, LATS)
        looptime = @elapsed run_snapshot(runnm, LAT)
        println("finished ", runnm," in $(round(looptime, digits=1)) s")
    end
end
println("finished in $(round(elapsed, digits=1)) s")