#= IW_A0nl_extract.jl
Maarten Buijsman, USM DMS, 2026-8-28
Extract the MEASURED A0nl (max tidal-band vertical isopycnal displacement) from
the model output, one x-column just east of the Gaussian source (x=xA0), for
each run. Split out of IW_nondim_params.jl's calcA0mod=true branch: that path
opens the sim netCDF, Butterworth-filters a whole water column, and runs
APEKFeq2 -- by far the slowest part of that script (1000+ s for a 13-run
block on the 200 m grid, vs the analytic A0nlana this replaces as the default
which needs no netCDF at all). Since A0nl doesn't change once a run has been
simulated, there's no reason to pay that cost on every IW_nondim_params.jl
run -- extract it here once per run, save it, and have IW_nondim_params.jl
just load it.

Saves one a0nl_AMZexptXX.YY.jld2 per run (same dirout as nondim_*.jld2 and
beatdist_*.jld2), loaded back into IW_nondim_params.jl.
=#

println("number of threads is ",Threads.nthreads())

using Pkg, NCDatasets, Printf, Statistics, JLD2, Interpolations, Trapz

WIN = 0;

if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\";
    dirsim = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\";
    dirout = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\diagout\\";
    dirforce = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\IW\\forcingfiles\\";
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0 = "/home/mbui/ModelOutput/"
    dirsim = string(pth0,"IW/");
    dirout = string(pth0,"diagout/");
    dirforce = string(pth0,"IW/forcingfiles/");
    dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";
end

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

savefl = 1  # save the per-run a0nl_*.jld2

const T2 = 12+25.2/60
const rho0=1020;
const grav=9.81;

# Gaussian source patch -- must match IW_flux_LAT_2000km_bash_cuda.jl and
# IW_nondim_params.jl (kept in sync manually, same as those files)
const gausW_center     = 80_000   # m
const gausW_width      = 16_000   # m
const A0_offset_sigma  = 2        # A0nl extraction point: this many sigma east of the source center
const xA0              = gausW_center + A0_offset_sigma*gausW_width   # 112 km

# run-ID selection: same block convention as IW_nondim_params.jl /
# IW_KEt_beat_distance.jl -- swap the active runnms line to pick a different block
mainnm  = 11
runnms  = collect(66:78)   # varying  N2 MERCATOR             F=12.5 kW/m
#runnms  = collect(27:39) # varying  N2 MERCATOR              F=25   kW/m
#runnms  = collect(40:52) # constant N2 MERCATOR 2.5N        F=25   kW/m
#runnms  = collect(53:65) # constant N2 MERCATOR 50N         F=25   kW/m
#runnms  = collect(66:78) # constant N2 MERCATOR             F=50   kW/m

runs = get_runs(mainnm, runnms)
LATS = [r.lat for r in runs]

function extract_A0nl(runnm, LAT, savefl)
    fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
    filename = string(dirsim,fnames,".nc")
    println(fnames,"; lat=",LAT," -------------------")

    row = get_runs(mainnm, [runnm])[1]

    # load N2 profile -- needed for the reference density profile below
    fnamegrid = n2_filename(row)
    path_fname = string(dirforce,fnamegrid);
    @load path_fname N2w zfw
    zc = (zfw[1:end-1] .+ zfw[2:end]) ./ 2;   # cell centers, matches the model grid

    ds = NCDataset(filename,"r");

    # only select data after the spinup time
    tspin = 10; #days  # for 2000 km domain
    tday0 = ds["time"][:]/24/3600;
    Isel  = findall(>=(tspin),tday0);
    tsec  = ds["time"][Isel];
    tday  = tsec/24/3600;
    dt    = tday[2]-tday[1]

    xc = ds["x_caa"][:];

    # reference density profile ---------------------------------------------
    # b = sum N2 * dz = sum -g/rho0*drho/dz * dz
    # b = -g/rho0*rho_pert  ->  rho_pert = -b*rho0/g
    breff   = cumtrapz(zfw, N2w);                              # bottom up!
    intzc   = interpolate((zfw,), breff, Gridded(Linear()));
    rhorefc = -intzc.(zc) * rho0/grav;                         # rho0 is not added!
    rrr_shape = reshape(rhorefc,1,1,:);

    # time window for averaging (same convention as IW_total_energetics_tile.jl)
    EXCL = 2; t1 = tday[1]+EXCL*T2/24; t2 = tday[end]-EXCL*T2/24;
    numcycles = floor((t2-t1)/(T2/24))
    t2   = t1+numcycles*(T2/24)
    Iday = findall(item -> item >= t1 && item<= t2, tday)

    # filter settings
    Nf    = 8;
    Tcut1 = 18/24           #D2+HH
    Tcut2 = (T2+T2/2)/2/24  #day; HH M2-M4

    # A0nl: max tidal-band isopycnal displacement at a single x-column just east
    # of the Gaussian source (x=xA0), instead of a domain-wide max
    ixA = argmin(abs.(xc .- xA0))
    println("A0nl column: x=",@sprintf("%.1f",xc[ixA]/1e3)," km (source center=",gausW_center/1e3,
        " km, +",A0_offset_sigma,"σ)")

    Nz = length(zc);
    bc_col = permutedims(ds["b"][ixA:ixA, :, Isel], (3,1,2));   # (Nt, 1, Nz)

    passflg = "high";
    bh_col = similar(bc_col)
    for iz = 1:Nz
        bh_col[:,1,iz] = lowhighpass_butter(bc_col[:,1,iz], Tcut2, dt, Nf, passflg)
    end

    passflg = "low";
    bs_col = similar(bc_col)
    for iz = 1:Nz
        bs_col[:,1,iz] = lowhighpass_butter(bc_col[:,1,iz], Tcut1, dt, Nf, passflg)
    end

    bt_col = (bc_col .- bh_col) .- bs_col   # tidal band = total - highpass - subtidal
    rt_col = -bt_col * rho0/grav
    bc_col = nothing; bh_col = nothing; bs_col = nothing

    thresh = 1e-5;
    APEz_col, Zzt_col = APEKFeq2(rt_col[Iday,:,:] .+ rrr_shape, rhorefc, zc, grav, thresh)
    A0nl = maximum(Zzt_col)/2 + abs(minimum(Zzt_col))/2

    close(ds)

    println(fnames,"; A0nl=",@sprintf("%.1f",A0nl)," m")

    if savefl == 1
        fnameout = string("a0nl_",fnames,".jld2")
        jldsave(string(dirout,fnameout); LAT, A0nl);
        println(fnameout," data saved ........ ")
    end

    return A0nl
end

elapsed = @elapsed begin
    for (runnm, LAT) in zip(runnms, LATS)
        looptime = @elapsed extract_A0nl(runnm, LAT, savefl)
        println("finished ", runnm," in $(round(looptime, digits=1)) s")
    end
end
println("finished in $(round(elapsed, digits=1)) s")
