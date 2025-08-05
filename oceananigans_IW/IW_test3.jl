#= IW_test3.jl
Maarten Buijsman, USM DMS, 2025-7-29
Set up IW test case for Oceananigans
use open boundary forcing w and u
also include a sponge layer on the right boundary

todo: 1) include parameters in functions, 
      2) add a mode 2
      3) realistic N, 
=#

println("number of threads is ",Threads.nthreads())

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie
using Statistics

###########------ SIMULATION PARAMETERS ------#############

# file ID
#fid = "_lat0_boundaryforce_sponge_closure"  #true U = 0.1  amplitude
#fid = "_lat0_boundaryforce_closure"  #true U = 0.1  amplitude
#fid = "_lat0_boundaryforce_closure_IW1" 
fid = "_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup" 

# grid parameters
pm = (lat=0, Nz=50, Nx=100, H=1000, L=500_000)

coriolis = FPlane(latitude = pm.lat)    # Coriolis


# internal wave parameters
# Imod: number of modes; U0n: modal amps; N: stratification; T:tidal period
pm = merge(pm,(Imod=2, U0n=[0.1, 0.05], N=0.005, T=(12+25.2/60)*3600, f=coriolis.f))

println("Δx = ",pm.L/pm.Nx/1e3," km")
println("Δz = ",pm.H/pm.Nz," m")

#= create parameter tuple == examples
parameters are used as local variables in functions => faster
pm = (; i, f, s)
pm2 = (; k=1, g=2, j="top")
pm4 = merge(pm,pm2,(ii = 1, ff = 3.14))
aa, bb = 3,-5.3
pm5 = merge(pm4,(; aa, bb))
=#

# sponge regions
#const Sp_Region_right = 500                               # size of sponge region on RHS
#const Sp_Region_left = 500
const Sp_Region_right = 20000                              # size of sponge region on RHS
const Sp_Region_left = 20000
const Sp_extra = 0                                         # not really needed

coriolis = FPlane(latitude = pm.lat);    # Coriolis

grid = RectilinearGrid(size=(pm.Nx, pm.Nz), 
                       x=(0,pm.L), 
                       z=(-pm.H, 0), 
                       topology=(Bounded, Flat, Bounded))

###########-------- Parameters ----------------#############

# internal wave derived parameters

# frequency
ω=2*π/pm.T
pm = merge(pm,(; ω))

# wave number and amplitude
kn=zeros(pm.Imod)
an=zeros(pm.Imod)
for n in 1:pm.Imod 
    kn[n] = n*π/pm.H*sqrt( (pm.ω^2-pm.f^2)/(pm.N^2-pm.ω^2))

    # reverse engineer an (from Gerkema IW syllabus; to be consistent with w eq)
    an[n] = pm.U0n[n]/(n*π/(kn[n]*pm.H))
end
pm = merge(pm,(; kn, an))

# check values
nn = 1:pm.Imod
println("U0n =", an.*(nn*π./(kn*pm.H)))
println("Ln =", 2*π./pm.kn/1e3, " km")
println("cn =",  pm.ω./pm.kn," m/s")
println("velocity u at z=0: ",-pm.an.*nn*π./(pm.kn*pm.H))

#ueig(z,p) = cos(n*π*z/H)
#weig(z,p) = sin(n*π*z/H)

###########------ FORCING ------#############

# background 
B_func(x, z, t, p) = p.N^2 * z
B = BackgroundField(B_func, parameters=pm)

#B_func(0, -1000, 0, pm)

# sponge regions
@inline heaviside(X)    = ifelse(X <0, 0.0, 1.0)
@inline mask2nd(X)      = heaviside(X)* X^2
@inline right_mask(x,p) = mask2nd((x-p.L+Sp_Region_right+Sp_extra)/(Sp_Region_right+Sp_extra))
@inline left_mask(x,p)  = mask2nd(((Sp_Region_left+Sp_extra)-x)/(Sp_Region_left+Sp_extra))

#= plot function 
heavisidef(X)  = ifelse(X <0, 0.0, 1.0)
mask2ndf(X)    = heavisidef(X)* X^2
right_maskf(x) = mask2ndf((x-L+Sp_Region_right+Sp_extra)/(Sp_Region_right+Sp_extra))
left_maskf(x)  = mask2ndf(((Sp_Region_left+Sp_extra)-x)/(Sp_Region_left+Sp_extra))

xx = xnodes(grid, Center())
#xx = range(0,L,1000)
asw1 = right_maskf.(xx)
asw2 = left_maskf.(xx)

fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax,xx/1e3,asw1,color=:red)
scatter!(ax,xx/1e3,asw2,color=:blue)
fig
=#

# nudging layer ∂x/∂t = F(x) + K(x_target - x) 
# K has units [1/time]
@inline u_sponge(x, z, t, u, p) = - 0.001 * right_mask(x, p) * u 
@inline v_sponge(x, z, t, v, p) = - 0.001 * right_mask(x, p) * v 
@inline w_sponge(x, z, t, w, p) = - 0.001 * right_mask(x, p) * w 
@inline b_sponge(x, z, t, b, p) = - 0.001 * right_mask(x, p) * b 
#@inline b_sponge(x, z, t, b) =   0.001 * right_mask(x) * (N^2 * z - b) + 0.001 * left_mask(x) * (N^2 * z - b)

# additional body forcing can be added
@inline force_u(x, z, t, u, p) = u_sponge(x, z, t, u, p) 
@inline force_v(x, z, t, v, p) = v_sponge(x, z, t, v, p) 
@inline force_w(x, z, t, w, p) = w_sponge(x, z, t, w, p) 
@inline force_b(x, z, t, b, p) = b_sponge(x, z, t, b, p) 

u_forcing = Forcing(force_u, field_dependencies = :u, parameters = pm)
v_forcing = Forcing(force_v, field_dependencies = :v, parameters = pm)
w_forcing = Forcing(force_w, field_dependencies = :w, parameters = pm)
b_forcing = Forcing(force_b, field_dependencies = :b, parameters = pm)

# boundary forcing
# u and w functions per mode, from Gerkema syllabus
Dx = xspacings(grid, Center())[1]
fun(z,t,n,p)    = -p.an[n] * n*π/(p.kn[n]*p.H) * cos(n*π*z/p.H) * sin(               -p.ω*t)
fwn(z,t,n,Dx,p) =  p.an[n]                     * sin(n*π*z/p.H) * cos(p.kn[n]*(-Dx/2)-p.ω*t)

# rampup function to start u and w from zero
const Tr = pm.T/2
framp(t,Tr) = 1-exp(-1/Tr*t)
#framp.(range(0,2*pm.T,24),Tr)

#= check
fun(1000,pm.T/4,1,pm)
fwn(-500,pm.T/4,1,Dx,pm) 
=#

# u at face, w at center offset by -Δx/2
@inline umod(z,t,p) = (fun(z,t,1,p)    + fun(z,t,2,p))    * framp(t,Tr)
@inline wmod(z,t,p) = (fwn(z,t,1,Dx,p) + fwn(z,t,2,Dx,p)) * framp(t,Tr)
#@inline vmod(z,t) = f/ω*an*n*π/(kn*H)*ueig(z)*cos(-kn*Dx/2-ω*t)

u_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(umod, parameters = pm))
w_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(wmod, parameters = pm))
#v_bcs = FieldBoundaryConditions(west = OpenBoundaryCondition(vmod))

#= WENO causes diffusion near the boundaries
model = NonhydrostaticModel(; grid, coriolis,
# try order = 9                advection = WENO(order=9),
                advection = Centered(order=4),
                buoyancy = BuoyancyTracer(),
                timestepper = :RungeKutta3,
                tracers = :b,
                background_fields = (; b=B),
                boundary_conditions=(u=u_bcs, w=w_bcs))
#                forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing),
=#

# this model does not cause diffusion near the botom amd surface boundaries
model = NonhydrostaticModel(; grid, coriolis,
                advection = Centered(order=4),
                closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
                tracers = :b,
                buoyancy = BuoyancyTracer(),
                background_fields = (; b=B),
                forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing),
                boundary_conditions=(u=u_bcs, w=w_bcs))                 


#model.forcing.u
#methods(force_u)

# generate screen output
wall_clock = Ref(time_ns())
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))
    wall_clock[] = time_ns()
    @info msg
    return nothing
end

# simulation time stepping
#Δt = 30seconds
Δt = 2minutes
start_time = 0days
stop_time = 8days
simulation = Simulation(model; Δt, stop_time)

add_callback!(simulation, progress, name=:progress, IterationInterval(400))

# write output
fields = Dict("u" => model.velocities.u, 
              "v" => model.velocities.v, 
              "w" => model.velocities.w, 
              "b" => model.tracers.b)

pthnm = "/data3/mbui/ModelOutput/IW/"
fname = "IW_fields_U0n"
filename=string(pthnm,fname,pm.U0n[1],fid,".nc")

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filename, 
    schedule=TimeInterval(30minutes),
    overwrite_existing = true)

#conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minutes)

# integrate the mofo
model.clock.iteration = 0
model.clock.time = 0
run!(simulation)


##################  read the NC fields   ##################### 
# plot the velocity as a function of time

pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\"
include(string(pathname,"include_functions.jl"))

fnames = "IW_fields_U0n0.1_lat0_bndfrc_advc4_spng_8d_dt2m_2mds_rampup.nc"
filename = string("C:\\Users\\w944461\\Documents\\work\\data\\julia\\",fnames)

ds = NCDataset(filename,"r");

tsec = ds["time"];
tday = tsec/24/3600;

xf   = ds["x_faa"]; 
xc   = ds["x_caa"]; 
zc   = ds["z_aac"]; 

Nz = length(zc);
Nx = length(xc);
Nt = length(tday);

# u(x_faa, z_aac, time)
ufs = ds["u"][:,1,:];
ufs = ds["u"][:,end,:];   #surface velocity

fig1 = Figure()
ax1a = fig1[1, 1] 
ax1b = fig1[2, 1] 
heatmap(ax1a, xf/1e3, tday, ufs)
heatmap(ax1b, xf/1e3, tday, ufs)
fig1

# compute some energy terms
uf = ds["u"];
uc = uf[1:end-1,:,:]/2 + uf[2:end,:,:]/2; #map to centers

KE = dropdims(mean(uc.^2, dims=3), dims=3);

fig2 = Figure()
ax2 = fig2[1, 1] 
heatmap(ax2, xc/1e3, zc , KE)
fig2

# now do some fft on surface velocities along the transect

# surf vel
ucs = uc[:,end,:]

fig3 = Figure()
ax3 = fig3[1, 1] 
heatmap(ax3, xc/1e3, tday, ucs)
fig3

tukeycf=0.0; numwin=1; linfit=true; prewhit=false;
Nfreq = Nt÷numwin÷2;
pwr = Matrix{Float64}(undef, Nx, Nfreq); 
period=[]; freq=[];
for i=1:Nx
    period, freq, pwr[i,:] = fft_spectra(tday, ucs[i,:]; tukeycf, numwin, linfit, prewhit);    
end

fig4 = Figure()
ax4 = Axis(fig4[1, 1])  # <-- create Axis, not GridPosition
heatmap!(xc ./ 1e3, freq, log10.(pwr))
ylims!(ax4, 0, 5)
fig4

# close the nc file
close(ds)
