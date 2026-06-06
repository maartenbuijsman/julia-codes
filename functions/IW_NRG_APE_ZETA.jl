"""
    APEKFeq2(rhop, rhorefc, zc, grav, thresh)

Computes instantaneous 2D APE and vertical displacent based
Kang And Fringer (2010), their equation 2. It is more accurate than the 
linear APE 1/2*b2/N2 (equation 4), in particular for supertidal motions.

# Arguments
- `rhop`:   dens. pert.               (kg/m3),  [time,x,z]
- `rhorefc`:ref. dens. pert.          (kg/m3),  [z]
- `zc`:     vertical coord. bottom-up (m),      [z]

# Returns
- `APE`: APE (J/m3),                                 [time,x,z] 
- `ZETA`: vertical displacement, positive is up (m), [time,x,z] 
"""

# function to compute APE as in Kang And Fringer (2010) --------------------
# This is Claudformed code based on the expensive loop APE and its preamble below

function APEKFeq2(rhop, rhorefc, zc, grav, thresh)
    # suggested value thresh = 1e-5;
    # if rhop-rhorefc > thresh, then APE  = 0    
    Nt, Nx, Nz = size(rhop)
    APEx        = zeros(Nt, Nx, Nz)
    zeta        = zeros(Nt, Nx, Nz) 

    itp_zs = extrapolate(interpolate((-rhorefc,), zc, Gridded(Linear())), Flat())
    F_ref  = cumtrapz(zc, rhorefc)
    rho_lo = rhorefc[end]   # lightest (surface)
    rho_hi = rhorefc[1]     # densest  (bottom)

    # exact cumulative integral of piecewise-linear rhorefc from zc[1] to z
    function exact_F(z)
        j = searchsortedfirst(zc, z)
        if j == 1
            return F_ref[1] + rhorefc[1] * (z - zc[1])
        elseif j > length(zc)
            return F_ref[end] + rhorefc[end] * (z - zc[end])
        else
            dz = zc[j] - zc[j-1]
            dt = z - zc[j-1]
            return F_ref[j-1] + rhorefc[j-1]*dt + (rhorefc[j]-rhorefc[j-1])/(2*dz) * dt^2
        end
    end

    Threads.@threads for it in 1:Nt
        for ix in 1:Nx
            for i in 1:Nz
                rho_i = rhop[it, ix, i]
                (rho_i < rho_lo || rho_i > rho_hi) && continue
                abs(rhorefc[i] - rho_i) < thresh    && continue

                # zeta = zrho-zstar
                zrho  = zc[i]
                zstar = itp_zs(-rho_i)
                z1    = min(zrho, zstar)
                z2    = max(zrho, zstar)
                fac   = zrho > zstar ? 1.0 : -1.0

                APEx[it, ix, i] = fac * grav * (rho_i*(z2-z1) - (exact_F(z2) - exact_F(z1)))
                # vertical displacement
                zeta[it, ix, i] = zrho - zstar
            end
        end
    end
    return APEx, zeta
end

## ---------------------------------------------------------

#= The above code is based on the simple, but slow code below
idx, d = nearest_index(xc, 1480e3)

# find vertical displacement xi
#it = 700 #xi down
#it = 718 #xi up
it = 740 #xi up
rhops = rhop[it,idx,:]  #xi is up

# map rhops values out of rhorefc range to local rhorefc
# this is due to machine errors
# near surface
Isel = rhorefc[end] .- rhops  .> 0 
rhops[Isel] = rhorefc[Isel]

# and near bottom
Isel = rhorefc[1] .- rhops  .< 0 
rhops[Isel] = rhorefc[Isel]

# exclude small differences to avoid weird extrapolations
Isel = abs.(rhorefc .- rhops)  .> 1/1e5 
Ilp = collect(1:Nz)
Ilp = Ilp[Isel]

# interpolate zstar along rhoref @ rhops
# minus sign is to accomodate positive increase 
itp   = interpolate((-rhorefc,), zc, Gridded(Linear()))
intrc = extrapolate(itp, Line())

zs = copy(zc)
zs[Isel] = intrc.(-rhops[Isel]) 

# vertical displacement (+ is upward)
xi = zc - zs

# loop over depth and non-zero xi
APE = zeros(size(zc))
for i in Ilp
    zstar = zs[i]  # location of rho on rhoref; z-xi
    zrho  = zc[i]  # peturbation density rho  
    if zrho < zstar      # xi<0; down
        Is = findall(zrho .< zc .< zstar)
        zz = [zrho; zc[Is]; zstar]        
        rr = rhops[i] .- [rhorefc[i] ;rhorefc[Is]; rhops[i]] 
        fac = -1
    elseif zrho > zstar  # xi>0; up
        Is = findall(zstar .< zc .< zrho)
        zz = [zstar; zc[Is]; zrho]        
        rr = rhops[i] .- [rhops[i] ;rhorefc[Is]; rhorefc[i]] 
        fac = 1        
    end
    APE[i] = fac * grav * trapz(zz,rr)
end
=#

# APE_4D = APEKFeq2(rhop, rhorefc, zc, grav, thresh);

#= compare the performance of the various APEs
fig = Figure(size = (600, 800))
ax1 = Axis(fig[1,1])
lines!(ax1,APE3z[it,idx,:], zc,   color = :red,   label = "APE3z")
lines!(ax1,APE, zc,               color = :black, label = "APE")
lines!(ax1,APE3nlz[it,idx,:], zc, color = :green, label = "APE3nlz")
lines!(ax1,APE_4D[it,idx,:], zc, color = :orange, label = "APE_4D", linestyle = :dash)
axislegend(ax1, position = :rb)

ax2 = Axis(fig[1,2])
lines!(ax2,APE .- APE3z[it,idx,:], zc,   color = :red,   label = "APE3z")
lines!(ax2,APE .- APE3nlz[it,idx,:], zc, color = :green, label = "APE3nlz")
lines!(ax2,APE .- APE_4D[it,idx,:], zc, color = :orange, label = "APE_4D")
axislegend(ax2, position = :rb)
#limits!(ax1, nothing, nothing,-300, 0)
fig
=#