#= IW_KEt_GM_res_compare.jl
Maarten Buijsman, USM DMS, 2026-9-20

KEt(x) at one latitude -- 28.8 N by default, the M2 critical latitude -- for the
THREE GM+tide runs that share the same 25 kW/m forcing, plus the tide-only
reference. Companion to IW_KEt_efold_ppr.jl, which is the 200 m paper figure and
deliberately contains no 4 km runs; the resolution comparison lives here.

  11.34  tide only,  200 m NH                       -- no GM
  12.34  GM + tide,  4 km  HYD, k-clamp GM IC       -- twin of 13 at 4 km
  13.34  GM + tide,  200 m NH, k-clamp GM IC        -- twin of 12 at 200 m
  15.34  GM + tide,  200 m NH, redistribution GM IC -- the paper series

12 and 13 differ ONLY in resolution, so 12 vs 13 is the clean
hydrostatic/nonhydrostatic test. 13 and 15 differ only in the GM initial
condition, so that pair isolates the IC fix. Pairing 15 against 12 confounds
the two and must not be read as a resolution comparison.

Markers are the two points that define the bg=0 e-fold, as in
IW_KEt_efold_ppr.jl: the peak searched in 50-150 km, and the first crossing of
KEt_max/e inside min(1800 km, XSRC + Cg1*t1). Each run carries its own dotted
KEt_max/e level because each has its own peak -- the 200 m runs peak higher
because the 148 km peak catches a resolved oscillation crest, which raises
their threshold relative to the smooth 4 km trace.

Set LAT to any latitude in the LAT13 list.
=#
using Printf, JLD2, Statistics, CairoMakie

pathname  = "/home/mbui/Documents/julia-codes/functions/"
pth0      = "/home/mbui/ModelOutput/"
dirout    = string(pth0, "diagout/")
dirfig    = string(pth0, "figs/")
dirEIG    = string(pth0, "IW/forcingfiles/")
dirparams = "/home/mbui/Documents/julia-codes/oceananigans_IW/input_params/"
include(string(pathname, "include_functions.jl"))
include(string(dirparams, "run_master.jl"))

const LSM=1600.0; const XBOUND=1800e3; const XSRC=80e3
const T2=12+25.2/60; const TAVG1=10.0+2*T2/24
const LAT=28.8

RUNS = [(11, 34, "tide only, 200 m NH",                :black,     :solid),
        (13, 34, "GM+tide, 200 m NH, k-clamp IC",      :darkorange, :solid),
        (15, 34, "GM+tide, 200 m NH, redistrib. IC",   :crimson,   :solid),
        (12, 34, "GM+tide, 4 km HYD, k-clamp IC",      :royalblue, :dash)]

function grab(mainnm, rn)
    row=get_runs(mainnm,[rn])[1]
    @load string(dirout,@sprintf("energetics_AMZexpt%02i.%02i.jld2",mainnm,rn)) xc KEt
    @load string(dirEIG,@sprintf("EIG_AMZexpt%02i.%02i_LAT_%04.1f.jld2",mainnm,rn,row.lat)) Cgn
    cg=Cgn[1]; dxE=xc[2]-xc[1]
    y = dxE<500 ? gaussfilt(xc,KEt,LSM) : KEt
    xv=min(XBOUND, XSRC+cg*TAVG1*86400)
    I=findall(50e3 .<= xc .<= 150e3); i1=I[argmax(y[I])]
    tgt=y[i1]/ℯ
    K=findall(v->xc[i1]<=v<=xv, xc); ic=findfirst(y[K].<tgt)
    if ic===nothing
        jd=K[argmin(y[K])]
        return (; xc, y, cg, dxE, xv, i1, tgt, x2=xc[jd], y2=y[jd], hit=false,
                  Te=NaN, nef=log(y[i1]/y[jd]), minfrac=y[jd]/y[i1])
    end
    k=K[ic]; x2=xc[k-1]+(tgt-y[k-1])/(y[k]-y[k-1])*(xc[k]-xc[k-1])
    jd=K[argmin(y[K])]
    return (; xc, y, cg, dxE, xv, i1, tgt, x2, y2=tgt, hit=true,
              Te=(x2-xc[i1])/cg/86400, nef=1.0, minfrac=y[jd]/y[i1])
end

R=[(r, grab(r[1],r[2])) for r in RUNS]

println("KEt at ", LAT, "N, all with F = 25 kW/m\n")
println(rpad("run",8),rpad("configuration",36),rpad("dx",8),rpad("peak",16),
        rpad("min/peak",10),rpad("e-folds",9),"Te")
for (m,g) in R
    println(rpad(@sprintf("%d.%d",m[1],m[2]),8),rpad(m[3],36),
        rpad(@sprintf("%.0f m",g.dxE),8),
        rpad(@sprintf("%.0f km %.2f kJ",g.xc[g.i1]/1e3,g.y[g.i1]/1e3),16),
        rpad(@sprintf("%.3f",g.minfrac),10),rpad(@sprintf("%.2f",g.nef),9),
        g.hit ? @sprintf("%.2f d",g.Te) : "no 1/e")
end

## figure
cm=72/2.54
fig=Figure(size=(17cm,9cm), fontsize=10)
ax=Axis(fig[1,1], xlabel="x [km]", ylabel="KEt [kJ m⁻²]", xticks=0:250:2000,
    title=@sprintf("tidal-band KE at %.1f°N, F = 25 kW m⁻¹  —  the three GM runs and the tide-only reference",LAT),
    titlesize=10)
vspan!(ax, 1800, 2000, color=(:gray85,0.6))
for (m,g) in R
    lines!(ax, g.xc/1e3, g.y/1e3, color=m[4], linewidth=2, linestyle=m[5], label=m[3])
    hlines!(ax, [g.tgt/1e3], color=m[4], linewidth=0.8, linestyle=:dot)
    scatter!(ax, [g.xc[g.i1]/1e3],[g.y[g.i1]/1e3], color=m[4], marker=:circle,
        markersize=9, strokecolor=:black, strokewidth=0.5)
    scatter!(ax, [g.x2/1e3],[g.y2/1e3], color = g.hit ? m[4] : :transparent,
        marker=:diamond, markersize=11, strokecolor=m[4], strokewidth=g.hit ? 0.5 : 1.6)
end
xlims!(ax,0,2000); ylims!(ax,0,nothing)
axislegend(ax, position=:rb, framevisible=false, labelsize=9, backgroundcolor=:white)
text!(ax, 0.985, 0.985,
    text="● peak    ◆ KEt_max/e crossing (open: never reached)\ndotted: each run's own KEt_max/e level    shaded: sponge",
    align=(:right,:top), space=:relative, fontsize=8, color=:gray35)
display(fig)
fout=string(dirfig,@sprintf("KEt_GM_res_compare_%04.1fN.png",LAT))
savefig300(fout,fig)
println("\nsaved ",fout)
