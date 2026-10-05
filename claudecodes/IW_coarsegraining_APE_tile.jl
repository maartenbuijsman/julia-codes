#= IW_coarsegraining_APE_tile.jl
Maarten Buijsman, USM DMS, 2026-10-3
Load model runs and perform coarsegraining diagnostics: cross-scale KE transfer
Π_K (identical to oceananigans_IW/IW_coarsegraining_tile.jl) PLUS the cross-scale
APE transfer Π_A following Wenegrat, Chor & Barkan (2026, JFM subm., eq. 2.11)
Tiled in x to avoid memory overflow for large domains (e.g. 200 m, 2000 km)

    Π_A = -τ(u_i,ρ) ∂_i Υ^l,   Υ^l = g (z - z*(ρ̄))/ρ0,   τ(f,g) = fg‾ - f̄ ḡ

The reference profile ρ*(z) = rhorefc is the fixed background (BackgroundField),
the same one used by APEKFeq2 in IW_total_energetics_tile.jl, so the reference-
profile term R = 0. With a temporal filter ρ*(z) drops out of τ, and
τ(u_i,ρ) = -(ρ0/g) τ(u_i,b), so in 2-D (x-z)

    Π_A = τ(u,b) ∂ζ^l/∂x + τ(w,b) ∂ζ^l/∂z,   ζ^l = z - z*(ρ̄),  ρ̄ = ρ*(z) - ρ0 b̄/g

ζ^l is the exact (nonlinear) displacement of the filtered density, as in
APEKFeq2 but WITHOUT its thresh/out-of-range skips (those set ζ=0 for tiny
perturbations, which creates spurious steps in ∂ζ/∂x); z*(ρ) is linearly
extrapolated beyond the top/bottom cell centers. Positive Π_A = transfer of APE
from large (low-frequency) to small (high-frequency) scales, same sign
convention as Π_K. For comparison the linear form Π_A^lin, with ζ_lin = -b̄/N²(z),
is also computed.

Filter: same temporal Butterworth low pass (Tcut between D2 and D4) as for Π_K.
The derivation only requires that the filter commutes with derivatives.

Output: diagout/EtranAPE_AMZexptXX.YY.jld2 (Π_K terms with the same names as
Etran_*.jld2, plus ΠAxxa, ΠAzxa, ΠAlinxa [W/kg m] and ΠAxztot, ΠAlinxztot [W/kg])
The existing Etran_*.jld2 files are NOT overwritten.

# run in terminal
julia -t auto /home/mbui/Documents/julia-codes/claudecodes/IW_coarsegraining_APE_tile.jl 11 27
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2
using Interpolations

pathname = "/home/mbui/Documents/julia-codes/functions/"
pth0 = "/home/mbui/ModelOutput/"
dirsim = string(pth0,"IW/");
dirfig = string(pth0,"figs/");
dirout = string(pth0,"diagout/");
dirforce = string(pth0,"IW/forcingfiles/");
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/";

include(string(pathname,"include_functions.jl"))
include(string(dirparams,"run_master.jl"))  # RUN_TABLE, get_runs(), n2_filename(), elim_flim()

# print and save flags
figflag = 1
saveflag = 1

const T2   = 12+25.2/60
const rho0 = 1020;
const grav = 9.81;

# series and runs: ARGS = mainnm runnm1 runnm2 ... (default 11.27, D2 tide only, 0°N)
mainnm  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 11
runnms  = length(ARGS) >= 2 ? parse.(Int, ARGS[2:end]) : [27]

runs = get_runs(mainnm, runnms)   # errors immediately if a runnm isn't in RUN_TABLE
LATS = [r.lat for r in runs]

# inputs
xlim = 2000; # km
ntile = 10   # number of x-tiles; increase if memory is still tight

# z*(ρ) of the filtered density, no skips; linear extrapolation beyond the end cells
function zeta_nl(rhobar, itp_zs, zc)
    Nt, Nx, Nz = size(rhobar)
    zeta = similar(rhobar)
    Threads.@threads for ix in 1:Nx
        for iz in 1:Nz, it in 1:Nt
            zeta[it,ix,iz] = zc[iz] - itp_zs(-rhobar[it,ix,iz])
        end
    end
    return zeta
end

# d/dx at x-centers: central in tile interior, one-sided at tile ends (as dvdx, dwdx)
function ddx_c(f, xc_t)
    d = zeros(size(f))
    dx2 = reshape(xc_t[3:end] - xc_t[1:end-2],1,:,1)
    d[:,2:end-1,:] = (f[:,3:end,:] - f[:,1:end-2,:])./dx2
    d[:,1,:]   = (f[:,2,:] - f[:,1,:])/(xc_t[2]-xc_t[1])
    d[:,end,:] = (f[:,end,:] - f[:,end-1,:])/(xc_t[end]-xc_t[end-1])
    return d
end

# d/dz at z-centers: central in interior, one-sided at bottom/surface (as dudz)
function ddz_c(f, dzz2, dzb, dzs)
    d = zeros(size(f))
    d[:,:,2:end-1] = (f[:,:,3:end] - f[:,:,1:end-2])./dzz2
    d[:,:,1]   = (f[:,:,2] - f[:,:,1])/dzb
    d[:,:,end] = (f[:,:,end] - f[:,:,end-1])/dzs
    return d
end

# coarsegraining function (tiled in x) -----------------------------
function run_coarsegraining(runnm, LAT, row)

# file ID ----------------------
fnames = @sprintf("AMZexpt%02i.%02i",mainnm,runnm)
fname_short2 = fnames
filename = string(dirsim,fnames,".nc")
titlenm2 = string(LAT,"°N")
println(fname_short2,"; lat=",LAT," -------------------")
flush(stdout)   # explicit flush: stdout doesn't auto-flush during long blocking NetCDF reads when redirected to a file

# === Load grid and time metadata — keep ds open for tile reads ===
ds = NCDataset(filename,"r");

tspin = 10; # days (2000 km domain)
tday0 = ds["time"][:]/24/3600;
Isel  = findall(>=(tspin),tday0);
tsec  = ds["time"][Isel];
tday  = tsec/24/3600;
dt    = tday[2]-tday[1]

xc = ds["x_caa"][:];
zc = ds["z_aac"][:];
dx = ds["Δx_caa"][:];
dz = ds["Δz_aac"][:];

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# Time averaging window (same for all tiles, compute once)
if LAT <= 35
    EXCL = 2; t1 = tday[1]+EXCL*T2/24; t2 = tday[end]-EXCL*T2/24;
    numcycles = floor((t2-t1)/(T2/24))
    t2   = t1+numcycles*(T2/24)
else # 40, 45
    EXCL = 2; numcycles = 11;
    t2 = tday[end]-EXCL*T2/24
    t1 = t2 - numcycles*T2/24 # around 13 days
end
Iday = findall(item -> item >= t1 && item<= t2, tday)

# Filter parameters
dth = dt*24; N = 8;
Tcut = (T2+2*T2/4)/2
fstring = "low"

# z-gradient helpers (same for all tiles)
dz2  = zc[3:end] - zc[1:end-2];
dzz2 = reshape(dz2,1,1,:);
dzr  = reshape(dz,1,1,:);
dzb  = zc[2] - zc[1];      # bottom one-sided
dzs  = zc[end] - zc[end-1]; # surface one-sided

# reference density (as in IW_total_energetics_tile.jl) ---------------------
# b = sum N2 * dz = sum -g/rho0*drho/dz * dz;  rho_pert = -b*rho0/g
@load string(dirforce, n2_filename(row)) N2w zfw
breff   = cumtrapz(zfw, N2w);                              # bottom up!
intzc   = interpolate((zfw,), breff, Gridded(Linear()));
rhorefc = -intzc.(zc) * rho0/grav;                         # rho0 is not added!
rrr_shape = reshape(rhorefc,1,1,:);
# inverse reference profile z*(ρ); linear (not flat) extrapolation, no skips
itp_zs  = extrapolate(interpolate((-rhorefc,), zc, Gridded(Linear())), Line())
N2c     = N2w[1:end-1]/2 + N2w[2:end]/2;
iN2c    = reshape([n > 1e-10 ? 1/n : 0.0 for n in N2c],1,1,:)   # linear ζ only where N² > 0

# === Pre-allocate full time-mean output arrays ===
Πxa_full   = zeros(Nx,Nz);
Πza_full   = zeros(Nx,Nz);
Πnha_full  = zeros(Nx,Nz);
ΠAxa_full  = zeros(Nx,Nz);
ΠAza_full  = zeros(Nx,Nz);
ΠAlin_full = zeros(Nx,Nz);

# === Tile parameters ===
# nhalo=1: one x-center point on each side, needed for central diff of dvdx, dwdx, dζdx.
# The Butterworth filter is purely temporal so needs NO halo.
nhalo   = 1;
nx_base = Nx ÷ ntile;

# === Tile loop ===
for i_tile in 1:ntile
    println("  tile ",i_tile," / ",ntile)
    flush(stdout)

    # interior x-center indices (global, 1-based)
    ix_a = (i_tile-1)*nx_base + 1;
    ix_b = (i_tile == ntile) ? Nx : i_tile*nx_base;

    # x-center indices with halo
    ix_ah = max(1, ix_a-nhalo);
    ix_bh = min(Nx, ix_b+nhalo);
    nx_h  = ix_bh - ix_ah + 1;

    # x-face indices for u: one extra face to the right of the rightmost center
    ixu_ah = ix_ah;
    ixu_bh = ix_bh + 1;    # always ≤ Nx+1 since ix_bh ≤ Nx

    # Load tile: all selected time steps, full z range
    @time begin
        uf_t = permutedims(ds["u"][ixu_ah:ixu_bh, :, Isel], (3,1,2)); # Nt × (nx_h+1) × Nz
        vf_t = permutedims(ds["v"][ix_ah:ix_bh,   :, Isel], (3,1,2)); # Nt × nx_h     × Nz
        wf_t = permutedims(ds["w"][ix_ah:ix_bh,   :, Isel], (3,1,2)); # Nt × nx_h     × (Nz+1)
        bc_t = permutedims(ds["b"][ix_ah:ix_bh,   :, Isel], (3,1,2)); # Nt × nx_h     × Nz (perturbation b)
    end

    # Cell-center velocities for tile
    uc_t = uf_t[:,1:end-1,:]/2 + uf_t[:,2:end,:]/2;  # Nt × nx_h × Nz
    wc_t = wf_t[:,:,1:end-1]/2 + wf_t[:,:,2:end]/2;  # Nt × nx_h × Nz

    # === Filter (time-domain: each (ix,iz) column independent, no halo needed) ===
    ufl_t  = similar(uf_t);
    vfl_t  = similar(vf_t);
    wfl_t  = similar(wf_t);
    ucl_t  = similar(uc_t);
    wcl_t  = similar(wc_t);
    uucl_t = similar(uc_t);
    uvcl_t = similar(uc_t);
    uwcl_t = similar(uc_t);
    vvcl_t = similar(uc_t);
    vwcl_t = similar(uc_t);
    wwcl_t = similar(uc_t);
    bcl_t  = zeros(size(uc_t));
    ubcl_t = zeros(size(uc_t));
    wbcl_t = zeros(size(uc_t));

    nx_uf = size(uf_t,2);   # = nx_h+1

    Threads.@threads for ix in 1:nx_uf
        for iz in 1:Nz
            ufl_t[:,ix,iz] = lowhighpass_butter(uf_t[:,ix,iz],Tcut,dth,N,fstring)
        end
    end

    Threads.@threads for ix in 1:nx_h
        for iz in 1:Nz
            vfl_t[:,ix,iz]  = lowhighpass_butter(vf_t[:,ix,iz],Tcut,dth,N,fstring);
            ucl_t[:,ix,iz]  = lowhighpass_butter(uc_t[:,ix,iz],Tcut,dth,N,fstring);
            wcl_t[:,ix,iz]  = lowhighpass_butter(wc_t[:,ix,iz],Tcut,dth,N,fstring);
            uucl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*uc_t[:,ix,iz],Tcut,dth,N,fstring);
            uvcl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*vf_t[:,ix,iz],Tcut,dth,N,fstring);
            uwcl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*wc_t[:,ix,iz],Tcut,dth,N,fstring);
            vvcl_t[:,ix,iz] = lowhighpass_butter(vf_t[:,ix,iz].*vf_t[:,ix,iz],Tcut,dth,N,fstring);
            vwcl_t[:,ix,iz] = lowhighpass_butter(vf_t[:,ix,iz].*wc_t[:,ix,iz],Tcut,dth,N,fstring);
            wwcl_t[:,ix,iz] = lowhighpass_butter(wc_t[:,ix,iz].*wc_t[:,ix,iz],Tcut,dth,N,fstring);
            # APE: b̄, (ub)‾, (wb)‾
            bcl_t[:,ix,iz]  = lowhighpass_butter(bc_t[:,ix,iz],Tcut,dth,N,fstring);
            ubcl_t[:,ix,iz] = lowhighpass_butter(uc_t[:,ix,iz].*bc_t[:,ix,iz],Tcut,dth,N,fstring);
            wbcl_t[:,ix,iz] = lowhighpass_butter(wc_t[:,ix,iz].*bc_t[:,ix,iz],Tcut,dth,N,fstring);
        end
        wfl_t[:,ix,Nz+1] = lowhighpass_butter(wf_t[:,ix,Nz+1],Tcut,dth,N,fstring);
    end

    uf_t=nothing; vf_t=nothing; wf_t=nothing;
    uc_t=nothing; wc_t=nothing; bc_t=nothing;
    GC.gc()

    # === Gradients ===

    # dudx: forward diff over x-faces → x-centers (nx_h values)
    dxr_t  = reshape(dx[ix_ah:ix_bh],1,:,1);
    dudx_t = diff(ufl_t,dims=2)./dxr_t;           # Nt × nx_h × Nz

    # dwdz: forward diff over z-faces → z-centers
    dwdz_t = diff(wfl_t,dims=3)./dzr;             # Nt × nx_h × Nz

    # dudz, dvdz: central diff in z; one-sided at z-boundaries
    dudz_t = ddz_c(ucl_t, dzz2, dzb, dzs);
    dvdz_t = ddz_c(vfl_t, dzz2, dzb, dzs);

    # dvdx, dwdx: central diff in x; tile-end one-sided diffs are only kept
    # where the tile boundary coincides with the global domain boundary
    xc_t   = xc[ix_ah:ix_bh];
    dvdx_t = ddx_c(vfl_t, xc_t);
    dwdx_t = ddx_c(wcl_t, xc_t);

    # === Π_K terms ===
    Πx_t  = (ucl_t.*ucl_t .- uucl_t).*dudx_t .+
            (vfl_t.*ucl_t .- uvcl_t).*dvdx_t;
    Πz_t  = (ucl_t.*wcl_t .- uwcl_t).*dudz_t .+
            (vfl_t.*wcl_t .- vwcl_t).*dvdz_t;
    Πnh_t = (ucl_t.*wcl_t .- uwcl_t).*dwdx_t .+
            (wcl_t.*wcl_t .- wwcl_t).*dwdz_t;

    dudx_t=nothing; dwdz_t=nothing; dudz_t=nothing; dvdz_t=nothing;
    dvdx_t=nothing; dwdx_t=nothing;
    uucl_t=nothing; uvcl_t=nothing; uwcl_t=nothing;
    vvcl_t=nothing; vwcl_t=nothing; wwcl_t=nothing;
    ufl_t=nothing; vfl_t=nothing; wfl_t=nothing;

    # === Π_A terms (nonlinear and linear) ===
    τub_t = ubcl_t .- ucl_t.*bcl_t;    # τ(u,b)
    τwb_t = wbcl_t .- wcl_t.*bcl_t;    # τ(w,b)
    ubcl_t=nothing; wbcl_t=nothing; ucl_t=nothing; wcl_t=nothing;

    ζl_t  = zeta_nl(rrr_shape .- (rho0/grav)*bcl_t, itp_zs, zc);   # ζ^l = z - z*(ρ̄)
    ΠAx_t = τub_t.*ddx_c(ζl_t, xc_t);
    ΠAz_t = τwb_t.*ddz_c(ζl_t, dzz2, dzb, dzs);
    ζl_t  = nothing;

    ζlin_t  = -bcl_t.*iN2c;                                         # ζ_lin = -b̄/N²
    ΠAlin_t = τub_t.*ddx_c(ζlin_t, xc_t) .+ τwb_t.*ddz_c(ζlin_t, dzz2, dzb, dzs);
    ζlin_t=nothing; τub_t=nothing; τwb_t=nothing; bcl_t=nothing;
    GC.gc()

    # === Time-average over Iday ===
    tmean(A) = dropdims(mean(A[Iday,:,:],dims=1),dims=1);   # nx_h × Nz
    Πxa_t  = tmean(Πx_t);  Πza_t  = tmean(Πz_t);  Πnha_t  = tmean(Πnh_t);
    ΠAxa_t = tmean(ΠAx_t); ΠAza_t = tmean(ΠAz_t); ΠAlina_t = tmean(ΠAlin_t);
    Πx_t=nothing; Πz_t=nothing; Πnh_t=nothing; ΠAx_t=nothing; ΠAz_t=nothing; ΠAlin_t=nothing; GC.gc()

    # === Store interior (strip halo) into full arrays ===
    jx_a = ix_a - ix_ah + 1;   # local tile index of interior start
    jx_b = ix_b - ix_ah + 1;   # local tile index of interior end

    Πxa_full[ix_a:ix_b,:]   = Πxa_t[jx_a:jx_b,:];
    Πza_full[ix_a:ix_b,:]   = Πza_t[jx_a:jx_b,:];
    Πnha_full[ix_a:ix_b,:]  = Πnha_t[jx_a:jx_b,:];
    ΠAxa_full[ix_a:ix_b,:]  = ΠAxa_t[jx_a:jx_b,:];
    ΠAza_full[ix_a:ix_b,:]  = ΠAza_t[jx_a:jx_b,:];
    ΠAlin_full[ix_a:ix_b,:] = ΠAlina_t[jx_a:jx_b,:];

end  # tile loop

close(ds)

# depth integrals f(x) [W/kg m] and depth-mean profiles f(z) [W/kg]
dzz     = reshape(dz,1,:);
zint(A) = dropdims(sum(A.*dzz,dims=2),dims=2);
xmean(A) = dropdims(mean(A,dims=1),dims=1);
Πxxa  = zint(Πxa_full);  Πzxa  = zint(Πza_full);  Πnhxa  = zint(Πnha_full);
Πxza  = xmean(Πxa_full); Πzza  = xmean(Πza_full); Πnhza  = xmean(Πnha_full);
ΠAxxa = zint(ΠAxa_full); ΠAzxa = zint(ΠAza_full); ΠAlinxa = zint(ΠAlin_full);
ΠAxza = xmean(ΠAxa_full); ΠAzza = xmean(ΠAza_full); ΠAlinza = xmean(ΠAlin_full);
Πxztot     = Πxa_full .+ Πza_full .+ Πnha_full;
ΠAxztot    = ΠAxa_full .+ ΠAza_full;
ΠAlinxztot = ΠAlin_full;

# cumulative ρ0∫Π dx at 1800 km [kW/m]
Ix = findall(100e3 .<= xc .<= 1800e3)
cK = rho0*sum((Πxxa .+ Πzxa .+ Πnhxa)[Ix].*dx[Ix])/1e3
cA = rho0*sum((ΠAxxa .+ ΠAzxa)[Ix].*dx[Ix])/1e3
cL = rho0*sum(ΠAlinxa[Ix].*dx[Ix])/1e3
@printf("%s: rho0*int Pi dx (100-1800 km) [kW/m]: Pi_K %.2f  Pi_A %.2f  Pi_A^lin %.2f  Pi_K+Pi_A %.2f\n",
        fname_short2, cK, cA, cL, cK+cA)

# figure: Π_K, Π_A, Π_A^lin f(x) and f(z)
if figflag==1
    fig = Figure(size = (1000, 400));
    ax1 = Axis(fig[1, 1], xlabel = "x [km]", ylabel = "ρ₀∫Π dz [W/m²]", title=string(fname_short2,"; ",titlenm2))
    lines!(ax1, xc/1e3, rho0*(Πxxa.+Πzxa.+Πnhxa), color=:black, label="Π_K")
    lines!(ax1, xc/1e3, rho0*(ΠAxxa.+ΠAzxa), color=:red, label="Π_A")
    lines!(ax1, xc/1e3, rho0*ΠAlinxa, color=:blue, linestyle=:dash, label="Π_A lin")
    axislegend(ax1, position = :rt; framevisible = false)
    ax2 = Axis(fig[1, 2], xlabel = "ρ₀Π [W/m³]", ylabel = "z [m]", title="x-mean")
    lines!(ax2, rho0*(Πxza.+Πzza.+Πnhza), zc, color=:black, label="Π_K")
    lines!(ax2, rho0*ΠAxza, zc, color=:orange, label="Π_A x")
    lines!(ax2, rho0*ΠAzza, zc, color=:green, label="Π_A z")
    lines!(ax2, rho0*(ΠAxza.+ΠAzza), zc, color=:red, label="Π_A")
    lines!(ax2, rho0*ΠAlinza, zc, color=:blue, linestyle=:dash, label="Π_A lin")
    axislegend(ax2, position = :rb; framevisible = false)
    colsize!(fig.layout, 2, Relative(0.3))
    save(string(dirfig,"PIK_PIA_",fname_short2,".png"), fig)
end

fnameout = string("EtranAPE_",fname_short2,".jld2")
if saveflag==1
    jldsave(string(dirout,fnameout); LAT, xc, zc, Πnhxa, Πzxa, Πxxa, Πnhza, Πzza, Πxza, Πxztot,
            ΠAxxa, ΠAzxa, ΠAlinxa, ΠAxza, ΠAzza, ΠAlinza, ΠAxztot, ΠAlinxztot);
    println(string(fnameout)," data saved ........ ")
end

end  # function run_coarsegraining


# runnms loop ---------------
elapsed = @elapsed begin
    for (runnm, LAT, row) in zip(runnms, LATS, runs)
        looptime = @elapsed begin
            run_coarsegraining(runnm, LAT, row)
        end
        println("finished ", runnm," in $(round(looptime, digits=1)) s")
        flush(stdout)
    end
end
println("finished in $(round(elapsed, digits=1)) s")
flush(stdout)
