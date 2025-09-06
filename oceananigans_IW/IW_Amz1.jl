#= IW_Amz1.jl
Maarten Buijsman, USM DMS, 2025-8-28
use realistic N(z) and eigenfunctions generated in testing_sturmL.jl
use open boundary forcing w and u
also include a sponge layer on the right boundary
based on IW_test3.jl

todo: 1) include parameters in functions, 
      2) add a mode 2
      3) realistic N, 
=#

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2

println("number of threads is ",Threads.nthreads())

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

pathout  = "/data3/mbui/ModelOutput/IW/"

###########------ OUTPUT FILE NAME ------#############
# file ID

#fid      = "AMZ1_lat0_2d_mode1_U1" 

#= mode 1 + 2 weak velocity
numM = [1 2];    
Usur1, Usur2 = 0.05, 0.025
fid      = "AMZ1_lat0_8d_mode1_2_U1" 
=#

# mode 1 only, strong velocity
numM = [1];    
Usur1, Usur2 = 0.25, 0.0
fid      = "AMZ1_lat0_8d_mode1_U1" 

###########------ LOAD N and grid params ------#############

dirin     = "/data3/mbui/ModelOutput/IW/forcingfiles/";
fnamegrid = "N2_amz1.jld2";
path_fname = string(dirin,fnamegrid);

# variables loaded
# "N2w", "zfw", "lonsel", "latsel"

# Open the JLD2 file
gridfile = jldopen(path_fname, "r")
println(keys(gridfile))  # List the keys (variables) in the file
close(gridfile)

@load path_fname N2w zfw

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,N2w, zfw)
ylims!(ax1, -500, 0)
#xlims!(ax1, -2000, 10)
fig

###########------ SIMULATION PARAMETERS ------#############

# simulation time stepping
#Δt = 30seconds
Δt = 1minutes
start_time = 0days
stop_time  = 8days

#numM = 1;       
Nz = length(zfw)-1;
DX = 4000;
L  = 500_000;
Nx = Integer(L/DX);
H  = abs(round(minimum(zfw)));
TM2 = (12+25.2/60)*3600 # M2 tidal period

# sponge parameters
#const Sp_Region_right = 500                               # size of sponge region on RHS
#const Sp_Region_left = 500
const Sp_Region_right = 20000                              # size of sponge region on RHS
const Sp_Region_left = 20000
const Sp_extra = 0                                         # not really needed

# grid parameters
pm = (lat=0, Nz=Nz, Nx=Nx, H=H, L=L, numM=numM);

# surface velocities; T:tidal period
pm = merge(pm,(Usur=[Usur1, Usur2], T=TM2));

# Estimate velocities from surface KE

# surface velocities from surface KE
# surface KE = ave(1/2*rho*u^2) = 1/2*rho*ave(U^2*cos^2 om*t) 
# = 1/2*rho*U^2*ave(cos^2 om*t) = 1/2*rho*U^2*1/2 => KE = 1/4*rho*U^2 
# U = sqrt(4*KE/rho)
rhos = 1034
KEs1 = 15.0 #max 15 J/m3 mode 1
KEs2 = 10.0  #max 10 J/m3 mode 2
Usur1tst = sqrt(4*KEs1/rhos);
Usur2tst = sqrt(4*KEs2/rhos);
@printf("Based on KE, U1 = %.2f, U2 = %.2f\n",Usur1tst,Usur2tst)

println("Δx = ",pm.L/pm.Nx/1e3," km")
println("Δz = ",pm.H/pm.Nz," m on average")

grid = RectilinearGrid(size=(pm.Nx, pm.Nz), 
                       x=(0,pm.L), 
                       z=zfw,                               # provide z-faces
                       topology=(Bounded, Flat, Bounded))

###########-------- Parameters ----------------#############

# internal wave derived parameters

# tide and Coriolis frequencies
ω=2*π/pm.T
fcor = FPlane(latitude = pm.lat);    # Coriolis
pm = merge(pm,(f=fcor.f, ω=ω))

# eigen value problem 
nonhyd = 1;
kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = 
    sturm_liouville_noneqDZ_norm(zfw, N2w, pm.f, pm.ω, nonhyd);

# compute z at cel lcenters
zcw = zfw[1:end-1]/2 + zfw[2:end]/2;

pm = merge(pm,(zcw=zcw, zfw=zfw, kn=kn[1:2], Ueig=Ueig[:,1:2], Weig=Weig[:,1:2], N2w=N2w));


fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,Weig[:,1], zfw)
lines!(ax1,Weig[:,2], zfw)

ax2 = Axis(fig[1,2])
lines!(ax2,Ueig[:,1], zcw)
lines!(ax2,Ueig[:,2], zcw)

#ylims!(ax1, -500, 0)
#xlims!(ax1, -2000, 10)
fig

###########------ FORCING ------#############

# create functions fun u and fwm for w 
# for rightward propagating wave following Gerkema IW syllabus

using Interpolations

# u velocity field at the west boundary (x=0)
function fun(x,z,t,p)
    u = 0.0
    # loop over n modes
    for i in p.numM
        Ueig = p.Ueig[:,i] * p.Usur[i]/p.Ueig[end,i]   # scale to match Usurface velocity
        intzc = linear_interpolation(p.zcw, Ueig, extrapolation_bc=Line());
        Ueigi = intzc.(z);
        u = u .-Ueigi * sin(p.kn[i]*x - p.ω*t)
    end
    return u
end

# w velocity field at the west boundary (x=-Dx/2)
function fwn(x,z,t,p)
    w = 0.0
    # loop over n modes
    for i in p.numM    
        Weig = p.Weig[:,i] * p.Usur[i]/p.Ueig[end,i]   # scale to match Usurface velocity
        intzc = linear_interpolation(p.zfw, Weig, extrapolation_bc=Line());
        Weigi = intzc.(z);
        w = w .+ Weigi * cos(p.kn[i]*x - p.ω*t)  
    end
    return w
end

#=
fun(0,zfw[end],3/4*2π/pm.ω,1,pm)
fun(Ln[1]*3/4,zcw[end],0*2π/pm.ω,1,pm)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,fun(0,zfw,3/4*2π/pm.ω,1,pm),zfw)
lines!(ax1,fun(0,zcw,3/4*2π/pm.ω,1,pm),zcw)
fig

fwn(Ln[1]*4/4,zfw[end-10],0*2π/pm.ω,1,pm)

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,fwn(0,zfw,0*2π/pm.ω,1,pm),zfw)
lines!(ax1,fwn(0,zcw,0*2π/pm.ω,1,pm),zcw)
fig
=#


# maybe include strain of U due to W

###########------ FORCING ------#############

# background 
# buoyancy = -g/rho0*rho_pert
function B_func(x,z,t,p)
    # computes buoyancy field and interpolates values at z
    bb = cumtrapz(p.zfw, p.N2w);
    intzc = linear_interpolation(p.zfw, bb, extrapolation_bc=Line());
    bbi  = intzc.(z);
    return bbi
end

#B_func(x, z, t, p) = p.N^2 * z
#B_func(x, z, t, p) = 0.001 * z

B = BackgroundField(B_func, parameters=pm);

fig = Figure()
ax1 = Axis(fig[1,1])
lines!(ax1,B_func(0, zfw, 0, pm),zfw)
#lines!(ax1,bb*-1000/10,zfw)
fig

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

# rampup function to start u and w from zero
const Tr = pm.T/2
framp(t,Tr) = 1-exp(-1/Tr*t)

# u at face, w at center offset by -Δx/2
Dx = -0.5*xspacings(grid, Center())[1]
@inline umod(z,t,p) = fun(0,z,t,p)  * framp(t,Tr)
@inline wmod(z,t,p) = fwn(Dx,z,t,p) * framp(t,Tr)

#@inline umod(z,t,p) = 0.0 * framp(t,Tr)
#@inline wmod(z,t,p) = 0.0 * framp(t,Tr)

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
model = NonhydrostaticModel(; grid, coriolis=fcor,
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
simulation = Simulation(model; Δt, stop_time)

add_callback!(simulation, progress, name=:progress, IterationInterval(400))

# write output
fields = Dict("u" => model.velocities.u, 
              "v" => model.velocities.v, 
              "w" => model.velocities.w, 
              "b" => model.tracers.b)


filenameout=string(pathout,fid,pm.Usur[1],".nc")


simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filenameout, 
    schedule=TimeInterval(30minutes),
    overwrite_existing = true)

#conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=1minutes)

# integrate the mofo
model.clock.iteration = 0
model.clock.time = 0
run!(simulation)


