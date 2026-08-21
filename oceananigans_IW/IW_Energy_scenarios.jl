#= IW_Energy_scenarios.jl
Maarten Buijsman, USM DMS, 2026-8-10  (generated with Claude Code)

Design tool (eigen-solver only, NO model run): for a mode-1 internal tide,
compare three source-normalization scenarios across latitude —
  (1) fixed surface velocity   usurf
  (2) fixed depth-integrated KE
  (3) fixed energy flux         F = (KE+APE)*Cg
for different stratifications:
  "amz1"           : constant WOCE AMZ N2 (single fixed profile, N2_amz1.jld2)
  "zonalmeanfixed" : Mercator N2 at fixed latfix (same profile all lats)
  "zonalmean"      : latitude-VARYING Mercator N2 (N2_ZonalMeanAtl_lat<lat>.jld2)
so you can see the effect of the latitude-varying Mercator N2.

Relations (mode 1, linear, depth-integrated, time-mean; horizontal KE):
  u_wave = a*U(z)*cos(kx-wt),  a = modal amplitude,  usurf = a*U(0)
  KE  = 1/4 * rho0 * (1+f^2/w^2) * a^2 * ∫U(z)^2 dz          (∫Ueig2^2 dz = H)
  KE/APE = (w^2+f^2)/(w^2-f^2)  ->  APE = KE*(1-f^2/w^2)/(1+f^2/w^2)
  E = KE+APE ,  F = E*Cg
=#

using Pkg, NCDatasets, Printf, CairoMakie, Statistics, JLD2, Interpolations

WIN = 0
if WIN==1
    pathname = "C:\\Users\\w944461\\Documents\\JULIA\\functions\\"
    pth0     = "C:\\Users\\w944461\\Documents\\work\\data\\julia\\Oceananigans\\"
    dirforce = string(pth0,"IW\\forcingfiles\\")
    dirfig   = string(pth0,"figs\\")
else
    pathname = "/home/mbui/Documents/julia-codes/functions/"
    pth0     = "/home/mbui/ModelOutput/"
    dirforce = string(pth0,"IW/forcingfiles/")
    dirfig   = string(pth0,"figs/")
end
include(string(pathname,"include_functions.jl"))

const T2   = 12 + 25.2/60
const rho0 = 1020.0
const grav = 9.81

# ---------------- USER SETTINGS -------------------------------------------
figflag = 1                                        # 1 = save overview figure

# latitudes to evaluate (for "zonalmean" must be in the Mercator set:
#   0, 2.5, 5, 10, 15, ..., 60)
LATS = vcat(collect(0:2.5:5), collect(10:5:60))

# which stratifications to compare (any subset)
N2sources = ["amz1", "zonalmeanfixed", "zonalmean"]
latfix    = 2.5                                    # for "zonalmeanfixed"

# scenario targets
usurf_target = 0.40     # m/s     scenario 1 (fixed surface velocity)
KE_target    = 12.5e3   # J/m2    scenario 2 (fixed depth-integrated KE)
F_target     = 50e3   # W/m     scenario 3 (fixed energy flux E*Cg)

Nm     = 1              # mode number
nonhyd = 1
# --------------------------------------------------------------------------

ω = 2π / (T2*3600)

fnamegrid_of(src, lat) =
    src == "amz1"           ? "N2_amz1.jld2" :
    src == "zonalmean"      ? @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", lat) :
    src == "zonalmeanfixed" ? @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", latfix) :
    error("N2source must be amz1 | zonalmean | zonalmeanfixed, got: ", src)

scen_names = ("fix usurf=$(usurf_target) m/s",
              "fix KE=$(KE_target/1e3) kJ/m2",
              "fix F=$(F_target/1e3) kW/m")

nlat = length(LATS); nscen = 3
# results[src] = NamedTuple of (nscen × nlat) arrays; NaN where file missing
results = Dict{String,Any}()

for src in N2sources
    KE  = fill(NaN, nscen, nlat); APE = fill(NaN, nscen, nlat)
    E   = fill(NaN, nscen, nlat); F   = fill(NaN, nscen, nlat)
    us  = fill(NaN, nscen, nlat); amp = fill(NaN, nscen, nlat)
    fw2v = fill(NaN, nlat); Cgv = fill(NaN, nlat); U0v = fill(NaN, nlat); IU2v = fill(NaN, nlat)

    for (j,lat) in enumerate(LATS)
        path = string(dirforce, fnamegrid_of(src, lat))
        if !isfile(path)
            @warn "missing N2 file, skipping" src lat path
            continue
        end
        @load path N2w zfw
        fcor = coriolis(Float64(lat))
        kn, Ln, Cn, Cgn, Cen, Weig, Ueig, Ueig2 =
            sturm_liouville_noneqDZ_norm(zfw, N2w, fcor, ω, nonhyd)

        U0    = Ueig2[end,Nm]
        IntU2 = sum(Ueig2[:,Nm].^2 .* diff(zfw))     # ≈ H
        fw2   = (fcor/ω)^2
        Cg    = Cgn[Nm]
        pol   = 1 + fw2                               # 1 + f^2/w^2
        rat   = (1-fw2)/(1+fw2)                       # APE/KE
        fw2v[j]=fw2; Cgv[j]=Cg; U0v[j]=U0; IU2v[j]=IntU2

        # helper: from KE -> everything
        pack!(s, KEs) = begin
            KE[s,j]=KEs; APE[s,j]=KEs*rat; E[s,j]=KE[s,j]+APE[s,j]; F[s,j]=E[s,j]*Cg
            amp[s,j]=sqrt(4*KEs/(rho0*pol*IntU2)); us[s,j]=amp[s,j]*U0
        end

        # scenario 1: fix usurf
        a1 = usurf_target/U0
        pack!(1, 0.25*rho0*pol*a1^2*IntU2)
        # scenario 2: fix KE
        pack!(2, KE_target)
        # scenario 3: fix F  (E=F/Cg, KE=E*pol/2)
        pack!(3, (F_target/Cg)*pol/2)
    end
    results[src] = (KE=KE, APE=APE, E=E, F=F, us=us, amp=amp,
                    fw2=fw2v, Cg=Cgv, U0=U0v, IntU2=IU2v)
end

# ---------------- print tables --------------------------------------------
for src in N2sources
    r = results[src]
    println("\n================ N2source = $src ================")
    for s in 1:nscen
        println("--- scenario $s: ", scen_names[s], " ---")
        @printf("%-6s %-7s %-7s | %-9s %-9s %-9s %-9s %-9s\n",
                "lat","f/ω","Cg","usurf","KE[kJ]","APE[kJ]","E[kJ]","F[kW/m]")
        for (j,lat) in enumerate(LATS)
            isnan(r.KE[s,j]) && continue
            @printf("%-6.1f %-7.3f %-7.3f | %-9.4f %-9.2f %-9.2f %-9.2f %-9.2f\n",
                    lat, sqrt(r.fw2[j]), r.Cg[j], r.us[s,j],
                    r.KE[s,j]/1e3, r.APE[s,j]/1e3, r.E[s,j]/1e3, r.F[s,j]/1e3)
        end
    end
end

# ---------------- overview figure: 5 rows × 3 scenario cols ----------------
cols   = [:black, :red, :dodgerblue]              # one per N2source
qty    = ("KE [kJ/m²]","APE [kJ/m²]","F [kW/m]","u(—) v(- -) surf [m/s]","KE/APE")
getq(r,q,s) = q==1 ? r.KE[s,:]./1e3 : q==2 ? r.APE[s,:]./1e3 :
              q==3 ? r.F[s,:]./1e3 : q==4 ? r.us[s,:] : r.KE[s,:]./r.APE[s,:]

# CHECK: theoretical polarization ratio (ω²+f²)/(ω²−f²), depends only on f/ω (latitude)
ratio_theory = [ (1+(coriolis(float(l))/ω)^2)/(1-(coriolis(float(l))/ω)^2) for l in LATS ]

fig = Figure(size=(1150, 1180))
for qi in 1:5, s in 1:3
    ax = Axis(fig[qi, s],
        title  = qi==1 ? scen_names[s] : "",
        xlabel = qi==5 ? "latitude [°]" : "",
        ylabel = s==1 ? qty[qi] : "")
    for (ci,src) in enumerate(N2sources)
        r = results[src]
        lines!(ax, LATS, getq(r,qi,s), color=cols[ci], linewidth=2.5, label=src)
        if qi==4    # v surface velocity = (f/ω)*usurf : dashed, same colour as u
            lines!(ax, LATS, r.us[s,:].*sqrt.(r.fw2), color=cols[ci], linewidth=2.0, linestyle=:dash)
        end
    end
    if qi==5    # computed KE/APE (colored) should lie exactly on this dashed curve
        lines!(ax, LATS, ratio_theory, color=:black, linestyle=:dash, linewidth=2,
               label="(ω²+f²)/(ω²−f²)")
    end
    (qi==4 && s==3) && axislegend(ax, position=:lt, labelsize=9, framevisible=false)
    (qi==5 && s==1) && axislegend(ax, position=:lt, labelsize=9, framevisible=false)
end
Label(fig[0, :], "Mode-$Nm source scenarios vs latitude (rows: quantity, cols: scenario; lines: N2 profile)",
      fontsize=14)
display(fig)

if figflag==1
    fout = string(dirfig, "Energy_scenarios_mode$(Nm).png")
    save(fout, fig); println("\nsaved ", fout)
end