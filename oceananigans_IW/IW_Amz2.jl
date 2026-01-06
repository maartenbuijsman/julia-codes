#= IW_Amz2.jl
Maarten Buijsman, USM DMS, 2025-12-16
use realistic N(z) and eigenfunctions generated in testing_sturmL.jl
adopt closed L and R BCOs with sponge layers 
IW forcing is with body Gaussian force
based on IW_test3.jl
=#

using Pkg
using Oceananigans
using Oceananigans.Units
using NCDatasets
using Printf
using CairoMakie
using Statistics
using JLD2
using Printf

println("number of threads is ",Threads.nthreads())

pathname = "/home/mbui/Documents/julia-codes/functions/";
include(string(pathname,"include_functions.jl"));

pathout  = "/data3/mbui/ModelOutput/IW/"

###########------ OUTPUT FILE NAME ------#############

# file ID
mainnm = 1
runnm  = 29

fid = @sprintf("AMZexpt%02i.%02i",mainnm,runnm) 

println("running ",fid)

# WMTD seminar double the velocity ===========================
# mode 1+2, stronger velocity
#numM = [1 2];    
#Usur1, Usur2 = 0.4, 0.3

numM = [1];    
Usur1, Usur2 = 0.4, 0.0

#numM = [2];    
#Usur1, Usur2 = 0.0, 0.3

# dx grid size ----------------------
DX = 4000;
#DX = 200;

# select latitude ------------------------
lat = 0.0
#lat = 2.5
#lat = 5
#lat = 10
#lat = 20
#lat = 25
#lat = 30
#lat = 40
#lat = 50

# forscaling the Gaussian forcing
lats = [0, 2.5, 5, 10, 20, 25, 30, 40, 50];
fracs = [0.151, 0.151, 0.150, 0.148, 0.141, 0.135, 0.129, 0.112, 0.091];

lines(lats,fracs)

# simulation time stepping
#Δt = 30seconds
max_Δt = 10minutes
Δt     = 1minutes
#Δt     = 15seconds  #weno 200 m

start_time = 0days
#stop_time  = 2days
stop_time  = 12days

println("stop_time: ",stop_time,"; lat: ",lat,"; select mode: ",numM)

###########------ LOAD N and grid params ------#############

# load profile created by AMZ_stratification_profile.jl
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


#numM = 1;       
Nz = length(zfw)-1;
L  = 500_000;
Nx = Integer(L/DX);
H  = abs(round(minimum(zfw)));
TM2 = (12+25.2/60)*3600 # M2 tidal period
dx  = L/Nx

# sponge parameters
#const fnudl = 0.001  # 01.27
const fnudl = 0.002  # 01.28 stronger damping on left side
const fnudr = 0.0001 # 01.25
# const fnud = 0.00025
#const Sp_Region_right = 20_000                              # size of sponge region on RHS
const Sp_Region_right = 200_000                              # size of sponge region on RHS
const Sp_Region_left = 40_000
const Sp_extra = 0                                     # not really needed

# for Gaussian body force
const gausW_width = 16_000             # 3*4000 m
#const gausW_center = 25_000           # <=01.12; x position of Gaussian of forced wave
#const gausW_center = 25_000+Sp_extra  # 01.13 + 01.14
const gausW_center = 40_000            # x position of Gaussian of forced wave 01.15

#= plot sponge and forcing 
heavisidef(X)  = ifelse(X <0, 0.0, 1.0)
mask2ndf(X)    = heavisidef(X)* X^2
right_maskf(x) = mask2ndf((x-L+Sp_Region_right+Sp_extra)/(Sp_Region_right+Sp_extra))
left_maskf(x)  = mask2ndf(((Sp_Region_left+Sp_extra)-x)/(Sp_Region_left+Sp_extra))
gaus(x)        = exp( -(x -  gausW_center)^2 / (2 *  gausW_width^2))

xx = xnodes(grid, Center())
#xx = range(0,L,1000)
asw1 = right_maskf.(xx)
asw2 = left_maskf.(xx)
asw3 = gaus.(xx)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax,xx/1e3,asw1,color=:red)
lines!(ax,xx/1e3,asw2,color=:blue)
lines!(ax,xx/1e3,asw3,color=:green)
fig
=#

#stop()

# grid parameters
pm = (lat=lat, Nz=Nz, Nx=Nx, H=H, L=L, numM=numM, gausW_center=gausW_center, 
      gausW_width=gausW_width);

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

# eigen value problem - that depends on Coriolis!!
# store everytime
nonhyd = 1;
kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 = 
    sturm_liouville_noneqDZ_norm(zfw, N2w, pm.f, pm.ω, nonhyd);

fnameEIG = @sprintf("EIG_amz_%04.1f.jld2",lat) 

f   = copy(fcor.f);
om2 = copy(ω);
jldsave(string(dirin,fnameEIG); f, om2, zfw, N2w, nonhyd, kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2);
println(string(fnameEIG)," Ueig data saved ........ ")
Lstr = @sprintf("%5.1f",Ln[1]/1e3)
println("Mode 1 wavelength is ",Lstr," km")
println("fraction gauss_width/L1 is ",@sprintf("%5.3f",gausW_width/Ln[1]))

stop()

#=
fnameEIG = "EIG_amz1.jld2";
path_fname2 = string(dirin,fnameEIG);
@load path_fname2 kn Ueig Weig

# Open the JLD2 file
gridfile = jldopen(path_fname, "r")
println(keys(gridfile))  # List the keys (variables) in the file
close(gridfile)
=#

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

# dudt field centered at gausW_center
function fdun(x,z,t,p)
    du = 0.0
    # loop over n modes
    for i in p.numM
        phi = [0 π] # mode 1 and mode 2 are out of phase
        Ueig = p.Ueig[:,i] * p.Usur[i]/p.Ueig[end,i]   # scale to match Usurface velocity
        intzc = linear_interpolation(p.zcw, Ueig, extrapolation_bc=Line());
        Ueigi = intzc.(z);
        #u = u .-Ueigi * sin(p.kn[i]*x - p.ω*t - phi[i])
        # derivative; d/dx sin x = cos x 
        du = du .+Ueigi * p.ω * cos(p.kn[i]*(x-p.gausW_center) - p.ω*t - phi[i])
    end
    return du
end

# dwdt field centered at gausW_center
function fdwn(x,z,t,p)
    dw = 0.0
    # loop over n modes
    for i in p.numM   
        phi = [0 π] # mode 1 and mode 2 are out of phase
        Weig = p.Weig[:,i] * p.Usur[i]/p.Ueig[end,i]   # scale to match Usurface velocity
        intzc = linear_interpolation(p.zfw, Weig, extrapolation_bc=Line());
        Weigi = intzc.(z);
        #w = w .+ Weigi * cos(p.kn[i]*x - p.ω*t - phi[i])  
        # derivative; d/dx cos x = - sin x
        dw = dw .+ Weigi * p.ω * sin(p.kn[i]*(x-p.gausW_center) - p.ω*t - phi[i])  
    end
    return dw
end

# v velocity field at the west boundary (x=-Dx/2)
function fdvn(x,z,t,p)
    dv = 0.0
    # loop over n modes
    for i in p.numM   
        phi = [0 π] # mode 1 and mode 2 are out of phase
        Ueig = p.Ueig[:,i] * p.Usur[i]/p.Ueig[end,i]   # scale to match Usurface velocity
        intzc = linear_interpolation(p.zcw, Ueig, extrapolation_bc=Line());
        Ueigi = intzc.(z);
        #v = v .+ p.f/p.ω .* Ueigi * cos(p.kn[i]*x - p.ω*t - phi[i])
        # derivative; d/dx cos x = - sin x
        dv = dv .+ p.f .* Ueigi * sin(p.kn[i]*(x-p.gausW_center) - p.ω*t - phi[i])
    end
    return dv
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

###########------ FORCING ------#############

# background 
# buoyancy b = -g/rho0*rho_pert
# b = N2*z   = -g/rho0 * drho
# N2         = -g/rho0 * drho/dz
function B_func(x,z,t,p)
    # computes buoyancy field and interpolates values at z
    bb = cumtrapz(p.zfw, p.N2w);
    intzc = linear_interpolation(p.zfw, bb, extrapolation_bc=Line());
    bbi  = intzc.(z);
    return bbi
end

B = BackgroundField(B_func, parameters=pm);

#fig = Figure()
#ax1 = Axis(fig[1,1])
#lines!(ax1,B_func(0, zfw, 0, pm),zfw)
#lines!(ax1,bb*-1000/10,zfw)
#fig

# sponge regions
@inline heaviside(X)    = ifelse(X <0, 0.0, 1.0)
@inline mask2nd(X)      = heaviside(X)* X^2
@inline right_mask(x,p) = mask2nd((x-p.L+Sp_Region_right+Sp_extra)/(Sp_Region_right+Sp_extra))
@inline left_mask(x,p)  = mask2nd(((Sp_Region_left+Sp_extra)-x)/(Sp_Region_left+Sp_extra))

# nudging layer ∂x/∂t = F(x) + K(x_target - x) 
# K has units [1/time]
@inline u_sponge(x, z, t, u, p) = - (fnudl * left_mask(x,p) + fnudr * right_mask(x, p)) * u 
@inline v_sponge(x, z, t, v, p) = - (fnudl * left_mask(x,p) + fnudr * right_mask(x, p)) * v 
@inline w_sponge(x, z, t, w, p) = - (fnudl * left_mask(x,p) + fnudr * right_mask(x, p)) * w 
@inline b_sponge(x, z, t, b, p) = - (fnudl * left_mask(x,p) + fnudr * right_mask(x, p)) * b 
#@inline b_sponge(x, z, t, b) =   0.001 * right_mask(x) * (N^2 * z - b) + 0.001 * left_mask(x) * (N^2 * z - b)

# body forcing internal waves 
# ramp-up function to start du/dt, etc from zero
@inline framp(t, p) = 1 - exp(-1/(p.T/2)*t)
@inline gaus(x, p) = exp( -(x - p.gausW_center)^2 / (2 * p.gausW_width^2))
@inline Fu_wave(x, z, t, p) = fdun(x, z, t, p) * framp(t, p) * gaus(x, p)
@inline Fv_wave(x, z, t, p) = fdvn(x, z, t, p) * framp(t, p) * gaus(x, p)
@inline Fw_wave(x, z, t, p) = fdwn(x, z, t, p) * framp(t, p) * gaus(x, p)

@inline force_u(x, z, t, u, p) = u_sponge(x, z, t, u, p) + Fu_wave(x, z, t, p)
@inline force_v(x, z, t, v, p) = v_sponge(x, z, t, v, p) #+ Fv_wave(x, z, t, p) 
@inline force_w(x, z, t, w, p) = w_sponge(x, z, t, w, p) #+ Fw_wave(x, z, t, p)
@inline force_b(x, z, t, b, p) = b_sponge(x, z, t, b, p) 

u_forcing = Forcing(force_u, field_dependencies = :u, parameters = pm)
v_forcing = Forcing(force_v, field_dependencies = :v, parameters = pm)
w_forcing = Forcing(force_w, field_dependencies = :w, parameters = pm)
b_forcing = Forcing(force_b, field_dependencies = :b, parameters = pm)

#= WENO works very well; smooth field, but run is twice as long
model = NonhydrostaticModel(; grid, coriolis=fcor,
                advection = WENO(),
                tracers = :b,
                buoyancy = BuoyancyTracer(),
                background_fields = (; b=B),
                forcing = (u=u_forcing,v=v_forcing, w=w_forcing, b=b_forcing),
                boundary_conditions=(u=u_bcs, w=w_bcs)) 
=#

# this model does not cause diffusion near the botom amd surface boundaries
# order?
#                closure = ScalarDiffusivity(ν=1e-6, κ=1e-6),
#                closure = ScalarDiffusivity(ν=1e-4, κ=1e-4),
model = NonhydrostaticModel(; grid, coriolis=fcor,
                advection = Centered(order=4),
                closure = ScalarDiffusivity(ν=1e-2, κ=1e-2),
                tracers = :b,
                buoyancy = BuoyancyTracer(),
                background_fields = (; b=B),
                forcing = (u=u_forcing, v=v_forcing, w=w_forcing, b=b_forcing))  #01.26
#                forcing = (; u=u_forcing, v=v_forcing))   #01.21                              
#                boundary_conditions=(u=u_bcs, w=w_bcs))                 
#                boundary_conditions=(u=u_bcs, v=v_bcs, w=w_bcs))                 


# simulation time stepping
simulation = Simulation(model; Δt, stop_time)


#= generate screen output
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

add_callback!(simulation, progress, name=:progress, IterationInterval(400))
=#

progress(sim) = @printf("Iter: % 6d, sim time: % s , wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
                        iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
                        sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(50))


# write output
fields = Dict("u" => model.velocities.u, 
              "v" => model.velocities.v, 
              "w" => model.velocities.w, 
              "b" => model.tracers.b)

filenameout=string(pathout,fid,".nc")

simulation.output_writers[:field_writer] =
    NetCDFWriter(model, fields, filename=filenameout, 
    schedule=TimeInterval(30minutes),
    overwrite_existing = true)

conjure_time_step_wizard!(simulation, cfl=1.0, max_Δt=max_Δt)

# integrate the mofo
model.clock.iteration = 0
model.clock.time = 0
run!(simulation)


