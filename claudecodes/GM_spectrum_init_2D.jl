#= GM_spectrum_init_2D.jl
Maarten Buijsman, USM DMS, 2026-9-1 (generated with Claude Code)

Build a 2D (x,z) Garrett-Munk internal-wave velocity field u(x,z), v(x,z) for
initializing a nonhydrostatic simulation, using this project's own vertical
eigenmode solver (sturm_liouville_noneqDZ_norm) driven by a real, horizontally-
uniform N(z) profile -- NOT the idealized exponential-N/WKB approximation used
in the classic GM76/GM81 reference codes (e.g. https://github.com/joernc/GM81).

METHOD
------
For a set of frequencies ω sampled log-spaced from f to N_max = max(N(z)):
  1. Call sturm_liouville_noneqDZ_norm(zfw, N2w, f, ω, nonhyd) to get, for
     EVERY vertical mode simultaneously, the exact (non-hydrostatic) horizontal
     wavenumber k_j(ω), wavelength L_j(ω) = 2π/k_j, and the normalized
     horizontal-velocity eigenfunction Ueig2_j(z) (depth-mean-square = 1).
  2. Keep only modes resolvable on the target grid: 6*DX <= L_j(ω) <= L (at
     least 6 grid cells per wavelength, and no longer than the domain itself
     -- longer waves don't complete a cycle in this Bounded-x domain and are
     not real propagating waves here, just a large-scale tilt/residual).
  3. Weight each surviving (ω,j) pair by Munk's empirical GM76 spectral density
     E(ω,j) = B(ω)*H(j)*E (same B, H, E as in the GM81 reference code), and
     convert to a physical depth-averaged kinetic-energy density via the
     canonical GM76 reference stratification (N0, b) -- i.e. each (ω,j)
     component is given the STANDARD GM76 reference energy level, and the
     ACTUAL vertical structure (already correctly informed by the real N(z)
     via the eigensolver) distributes that energy over depth. This avoids
     double-applying WKB depth-scaling on top of an already-exact eigenmode
     shape.
  4. Since the domain is 2D (x,z only, no y), the true 3D Garrett-Munk
     integral over horizontal propagation azimuth is collapsed onto pure
     +/-x propagation: each (ω,j) component gets a uniformly-random direction
     (sign of k) and a uniformly-random phase in [0,2π). This is a standard
     random-phase superposition: with amplitude A(ω,j) = sqrt(2*K(ω,j)*Δω),
     summing many independent random-phase cosines reproduces the target
     power spectral density in expectation.
  5. u(x,z) = Σ A(ω,j) Ueig2_j(z) cos(s*k_j*x + φ)
     v(x,z) = Σ (f/ω) A(ω,j) Ueig2_j(z) sin(s*k_j*x + φ)
     (v/u polarization relation for a linear inertia-gravity wave, from
     -iω v + f u = 0 in the rotating linearized momentum equations; holds
     for either propagation direction since it depends only on ω, not on
     the sign of k.)

ASSUMPTIONS TO REVISIT IF NEEDED
---------------------------------
- Reference GM76 constants (N0, b, E, js) are the canonical literature values,
  not tuned to this specific ocean profile.
- "Random direction" = random sign of k (+x or -x), not a full 2D azimuth.
- Energy level uses the canonical GM76 reference ocean, not this profile's
  own N; only the mode SHAPE (not the total energy per mode) reflects the
  real, loaded N(z).
=#

println("number of threads is ", Threads.nthreads())

using Printf
using JLD2
using CairoMakie
using Statistics
using Random

pathname   = "/home/mbui/Documents/julia-codes/functions/"
pth0       = "/home/mbui/ModelOutput/"
dirforce   = string(pth0, "IW/forcingfiles/")
dirout     = string(pth0, "diagout/")
dirfig     = string(pth0, "figs/")

include(string(pathname, "include_functions.jl"))  # coriolis(), sturm_liouville_noneqDZ_norm()

# Flags -----------------------------------------------------------------
savefl  = 1  # save u,v(x,z) + metadata to jld2
figflag = 1  # save heatmap figure

# Domain / grid parameters (all easily changeable) ----------------------
LAT   = 25.0      # deg N -- picks the N2_ZonalMeanAtl_lat<LAT>.jld2 profile
                   # (the REAL N(z) for this latitude is always used, even
                   # near/at the equator -- only f is floored, see below)
LAT_f_floor = 2.5  # deg N -- minimum *effective* latitude used for the
                   # Coriolis frequency f. GM76's B(ω) collapses to zero at
                   # f=0 (it's a mid-latitude parameterization built around
                   # the near-inertial 1/sqrt(ω-f) peak, which loses meaning
                   # as f->0), so f is clamped to max(|f(LAT)|, |f(LAT_f_floor)|).
                   # Away from the equator this has zero effect. This is a
                   # pragmatic approximation, not a physically rigorous
                   # equatorial spectrum -- see chat discussion.
L     = 2_000_000.0   # domain length [m], 2000 km
DX    = 4000.0        # target grid spacing [m] -- "10-series" (4 km) for starters;
                       # switch to 200.0 for the "11-series" 200 m grid, etc.
Nx    = Integer(L / DX)
nonhyd = 1             # 1 = non-hydrostatic Sturm-Liouville problem (matches
                        # the production simulation script's convention)

# sponge/relaxation zone widths [m] -- match IW_flux_LAT_2000km_bash_cuda.jl's
# Sp_Region_left/right so the GM field is tapered to zero exactly where that
# script's Rayleigh-damping sponge already relaxes u,v,w,b toward zero. Keep
# these in sync if the production script's sponge widths ever change.
Sp_Region_left  = 40_000.0     # left sponge width [m] (0 to Sp_Region_left)
Sp_Region_right = 200_000.0    # right sponge width [m] (L-Sp_Region_right to L)

# spectral sampling -------------------------------------------------------
Nfreq   = 60           # number of log-spaced frequency bins from f to N_max
seed    = 1            # RNG seed, for reproducibility
minwavelenfac = 6       # "at least 6*dx cells per wavelength" resolution cutoff

# canonical GM76 reference constants (Munk 1981 notation, see gm.py) -----
const Egm = 6.3e-5      # GM energy parameter (dimensionless)
const js  = 3.0         # mode-number scale j*
const N0  = 5.24e-3     # reference buoyancy frequency [rad/s] (3 cph)
const bgm = 1300.0      # stratification e-folding scale [m]
const jsum = (π*js/tanh(π*js) - 1)/(2*js^2)  # sum_{j=1}^∞ (j²+js²)^-1

Random.seed!(seed)

# Munk's B(ω), H(j), E(ω,j) ----------------------------------------------
Bomg(om, f) = 2/π * f/om / sqrt(om^2 - f^2)
Hmode(j)    = 1.0 / (j^2 + js^2) / jsum
Eomgj(om, j, f) = Bomg(om, f) * Hmode(j) * Egm

# depth-averaged reference KE density [m²/s²  per rad/s] for mode (ω,j),
# using the canonical GM76 reference ocean (N -> N0, i.e. no extra WKB
# depth-scaling -- see header note)
Komgj(om, j, f) = bgm^2 * N0^2 * (om^2 + f^2) / om^2 * Eomgj(om, j, f)

# load N2(z) profile -------------------------------------------------------
fnamegrid  = @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", LAT)
path_fname = string(dirforce, fnamegrid)
@load path_fname N2w zfw

H  = abs(zfw[end] - zfw[1])
Nz = length(zfw) - 1
zc = zfw[1:end-1]/2 .+ zfw[2:end]/2

f_true = abs(coriolis(LAT))
f_floor = abs(coriolis(LAT_f_floor))
f_cor  = max(f_true, f_floor)
if f_cor > f_true
    println("NOTE: f at LAT=", LAT, " (", f_true, " rad/s) floored to f at LAT_f_floor=",
            LAT_f_floor, " (", f_cor, " rad/s) -- N(z) itself is still the real LAT=", LAT, " profile.")
end
N_max  = sqrt(maximum(N2w))
println("LAT=", LAT, "; f=", f_cor, " rad/s; N_max=", N_max, " rad/s; depth H=", round(H, digits=1), " m")
println("Nx=", Nx, "; DX=", DX, " m; min resolvable wavelength = ", minwavelenfac*DX/1e3, " km")

# frequency bins: log-spaced, staying strictly inside (f, N_max) ----------
eps_f = 1.001
eps_N = 0.999
omg_edges = exp.(range(log(f_cor*eps_f), log(N_max*eps_N), length=Nfreq+1))
omg_mid   = sqrt.(omg_edges[1:end-1] .* omg_edges[2:end])   # geometric bin centers
domg      = diff(omg_edges)

# build the list of (ω, j, k, amplitude, Ueig2 column) components --------
comp_k    = Float64[]
comp_A    = Float64[]
comp_om   = Float64[]
comp_Ueig = Vector{Vector{Float64}}()

zfw_f = Float64.(zfw); N2w_f = Float64.(N2w)  # ensure Float64 for the solver

for n in 1:Nfreq
    om = omg_mid[n]
    k, Lw, C, Cg, Ce, Weig, Ueig, Ueig2 = sturm_liouville_noneqDZ_norm(zfw_f, N2w_f, f_cor, om, nonhyd)
    nmodes = length(k)
    for j in 1:nmodes
        Lw[j] < minwavelenfac*DX && continue   # 6*dx resolution cutoff (unresolved on the grid)
        Lw[j] > L && continue                   # domain-length cutoff (doesn't complete a
                                                  # cycle -- not a propagating wave in this
                                                  # Bounded-x domain, just a large-scale tilt)
        K = Komgj(om, j, f_cor)
        A = sqrt(2 * K * domg[n])
        push!(comp_k, k[j])
        push!(comp_A, A)
        push!(comp_om, om)
        push!(comp_Ueig, Ueig2[:, j])
    end
end

ncomp = length(comp_k)
println("total (ω,mode) components used: ", ncomp)

# random phase and propagation direction per component --------------------
phase = 2π .* rand(ncomp)
dirsign = rand([-1.0, 1.0], ncomp)

# build u(x,z), v(x,z) ------------------------------------------------------
xc = ((0:Nx-1) .+ 0.5) .* DX

u = zeros(Nz, Nx)
v = zeros(Nz, Nx)

# single-threaded accumulation (ncomp * Nz * Nx multiply-adds -- a few
# hundred million flops at most for this problem size, no need to
# parallelize and risk a race on the shared u,v arrays)
for i in 1:ncomp
    kx  = dirsign[i] .* comp_k[i] .* xc
    th  = kx .+ phase[i]
    cph = cos.(th); sph = sin.(th)
    fac_v = f_cor / comp_om[i]
    u .+= comp_A[i] .* comp_Ueig[i] .* transpose(cph)
    v .+= (comp_A[i]*fac_v) .* comp_Ueig[i] .* transpose(sph)
end

# taper to zero across the sponge zones ------------------------------------
# same quadratic ramp shape as the production script's left_mask/right_mask
# (Rayleigh-damping weight), just used here as a 1->0 window instead of a
# damping-rate multiplier: 1 in the interior, ramping smoothly to 0 exactly
# at x=0 and x=L. Avoids populating GM energy that the sponge would just
# absorb anyway (~500s-2.8hr timescales) in the first stretch of the run.
mask2nd(X) = X > 0 ? X^2 : 0.0
left_mask(x)  = mask2nd((Sp_Region_left - x) / Sp_Region_left)
right_mask(x) = mask2nd((x - L + Sp_Region_right) / Sp_Region_right)
taper = 1.0 .- left_mask.(xc) .- right_mask.(xc)   # (Nx,), 1 in interior -> 0 at edges
u .*= transpose(taper)
v .*= transpose(taper)

# diagnostics ---------------------------------------------------------------
# E[depth-avg(u²+v²)] over the random phases/directions, from the u=cos,
# v=(f/ω)sin construction: Σ (1/2) A_i² (1 + (f/ω_i)²) -- this is the exact
# ensemble-mean target for THIS realization's component list.
#
# IMPORTANT: the vertical grid zfw is non-uniform (WKB-stretched, much finer
# near the surface -- dz varies by >20x top to bottom here), so a plain
# unweighted mean() over grid points overweights the energetic surface layer
# and is NOT a physically meaningful depth average. Must weight by dz.
dz_diag = diff(zfw)                    # zfw is bottom-to-surface (increasing) in this project's files
depth_avg_uv2 = vec(sum((u.^2 .+ v.^2) .* dz_diag, dims=1)) ./ H   # dz-weighted, per x
rms_actual = sqrt(mean(depth_avg_uv2))   # x-average is fine unweighted (x-grid IS uniform)
ensemble_mean_sq = sum(0.5 .* comp_A.^2 .* (1 .+ (f_cor ./ comp_om).^2))
rms_expected = sqrt(ensemble_mean_sq)
println("achieved rms sqrt(u²+v²): ", round(rms_actual, digits=4), " m/s")
println("ensemble-expected rms sqrt(u²+v²): ", round(rms_expected, digits=4), " m/s")
println("(a single random realization with only ", ncomp, " components can deviate ",
        "meaningfully from this ensemble mean -- increase Nfreq, or average multiple ",
        "seeds, if you need tight convergence to the target GM energy level)")

# figures --------------------------------------------------------------------
fig = Figure(size=(1000, 800))

axu = Axis(fig[1,1], title="GM-spectrum u(x,z), LAT=$(LAT)°N, DX=$(DX/1e3) km",
           xlabel="x [km]", ylabel="z [m]")
clim = maximum(abs, u) * 0.8
hmu = heatmap!(axu, xc/1e3, zc, transpose(u), colormap=Reverse(:RdBu), colorrange=(-clim, clim))
Colorbar(fig[1,2], hmu, label="u [m/s]")

axv = Axis(fig[2,1], title="GM-spectrum v(x,z)", xlabel="x [km]", ylabel="z [m]")
clim2 = maximum(abs, v) * 0.8
hmv = heatmap!(axv, xc/1e3, zc, transpose(v), colormap=Reverse(:RdBu), colorrange=(-clim2, clim2))
Colorbar(fig[2,2], hmv, label="v [m/s]")

fig
if figflag == 1
    save(string(dirfig, "GM_init_uv_lat", @sprintf("%04.1f", LAT), "_DX", Integer(DX), ".png"), fig)
end

# save fields -----------------------------------------------------------------
if savefl == 1
    fnameout = @sprintf("GM_init_uv_lat%04.1f_DX%d.jld2", LAT, Integer(DX))
    jldsave(string(dirout, fnameout); xc, zc, u, v, LAT, DX, f_cor, N_max, ncomp)
    println(fnameout, " saved ........")
end

println("done.")
