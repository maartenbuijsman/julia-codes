#= run_master.jl
Maarten Buijsman, USM DMS, 2026-9-30
Master lookup table for the AMZ Oceananigans internal-wave (mainnm/runnm) runs.
One row (RunInfo) per individual run: latitude, mode-1 forcing flux, nominal
grid spacing, and which N2 stratification profile IW_flux_LAT_2000km_bash_cuda.jl
used to force it (see gausW_center/gausW_width there for the generation site).

Usage from an analysis script:
    include(string(dirparams,"run_master.jl"))
    rows  = get_runs(mainnm, runnms)     # errors loudly if a runnm is missing
    LATS  = [r.lat for r in rows]
    fname = n2_filename(row)             # per-row N2 forcing filename
    Elim, Flim = elim_flim(row)          # per-row KE/APE/flux plot y-limits

Add new rows here as new run blocks come online.
=#

struct RunInfo
    mainnm::Int
    runnm::Int
    lat::Float64
    Flux::Float64        # mode-1 flux [W/m]
    DX::Float64          # nominal grid spacing [m]
    N2source::String     # "zonalmean" (per-run varying Mercator N2), "zonalmeanfixed" (single fixed-lat N2 for every run in the block), or "amz1" (constant WOCE AMZ N2, N2_amz1.jld2)
    latfix::Float64      # latitude of the fixed N2 profile when N2source=="zonalmeanfixed"; 99 = unused
end

# expand one run_batch-style block (shared lat/Flux vectors, scalar DX/N2source/latfix)
# into one RunInfo row per runnm
function expand_block(mainnm, runnm, lat, Flux, DX, N2source, latfix)
    n = length(runnm)
    @assert length(lat)  == n "lat length must match runnm length"
    @assert length(Flux) == n "Flux length must match runnm length"
    return [RunInfo(mainnm, runnm[i], lat[i], Flux[i], DX, N2source, latfix) for i in 1:n]
end

RUN_TABLE = RunInfo[]

const LAT13 = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 25.0, 28.8, 30.0, 35.0, 40.0, 45.0, 50.0]

# mainnm 11 (200 m / 4000 m D2 NH flux forcing) -------------------------------
append!(RUN_TABLE, expand_block(11, collect(1:13),  LAT13, fill(12.5e3,13), 200,  "zonalmean",      99))
append!(RUN_TABLE, expand_block(11, collect(14:26), LAT13, fill(12.5e3,13), 200,  "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(11, collect(27:39), LAT13, fill(25e3,13), 200,  "zonalmean",      99))
append!(RUN_TABLE, expand_block(11, collect(40:52), LAT13, fill(25e3,13), 200,  "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(11, collect(53:65), LAT13, fill(25e3,13), 200,  "zonalmeanfixed", 50.0))
append!(RUN_TABLE, expand_block(11, collect(66:78), LAT13, fill(50e3,13), 200,  "zonalmean",      99))

# mainnm 10 (4000 m D2 NH flux forcing) ---------------------------------------
append!(RUN_TABLE, expand_block(10, collect(1:13),  LAT13, fill(12.5e3,13), 4000, "zonalmean",      99))
append!(RUN_TABLE, expand_block(10, collect(14:26), LAT13, fill(12.5e3,13), 4000, "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(10, collect(27:39), LAT13, fill(25e3,13), 4000, "zonalmean",      99))
append!(RUN_TABLE, expand_block(10, collect(40:52), LAT13, fill(25e3,13), 4000, "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(10, collect(53:65), LAT13, fill(25e3,13), 4000, "zonalmeanfixed", 50.0))
append!(RUN_TABLE, expand_block(10, collect(66:78), LAT13, fill(50e3,13), 4000,  "zonalmean",      99))

# mainnm 12 (4000 m, Garrett-Munk-spectrum-initialized D2 NH flux forcing) ----
# IW_GM_flux_LAT_2000km_bash_cuda.jl; runnm 27:39 matches the 10-series
# "varying N2, 25kW/m" numbering convention (params_12.jl)
append!(RUN_TABLE, expand_block(12, collect(27:39), LAT13, fill(25e3,13), 4000, "zonalmean", 99))

# mainnm 13 (200 m, Garrett-Munk-spectrum-initialized D2 NH flux forcing) -----
# same as mainnm 12 but DX=200m (11-series grid); params_13.jl
append!(RUN_TABLE, expand_block(13, collect(27:39), LAT13, fill(25e3,13), 200, "zonalmean", 99))

# mainnm 13, runnm 1:13 (200 m, GM spectrum only, NO tidal forcing, Flux=0) ---
# free-decay/background GM-spectrum evolution, no external tidal energy input;
# params_13_noforce.jl
append!(RUN_TABLE, expand_block(13, collect(1:13),  LAT13, fill(0.0,13), 200, "zonalmean", 99))

# mainnm 14, runnm 1:13 (200 m, GM spectrum only, NO tidal forcing, Flux=0) ---
# w,b-consistent GM IC + k-clamp domain-length-cutoff fix (v1 of that fix; see
# chat). PARTIAL: only runnm 1:5 (lat 0-15) actually completed -- batch was
# stopped after finding the k-clamp fix caused anomalously slow energy decay
# at low latitude (large-scale coherent structure artifact from piling
# excess energy onto one wavenumber); superseded by mainnm 15 (redistribution
# fix). runnm 6 (lat 20) is a partial/truncated file, runnm 7:13 never ran.
# Generated from a since-deleted v2 snapshot of IW_GM_flux_LAT_2000km_bash_cuda.jl;
# params_14_noforce.jl
append!(RUN_TABLE, expand_block(14, collect(1:13),  LAT13, fill(0.0,13), 200, "zonalmean", 99))

# mainnm 15, runnm 1:13 (200 m, GM spectrum only, NO tidal forcing, Flux=0) ---
# same as mainnm 14's noforce block, but with the REDISTRIBUTION fix instead
# of k-clamp: dropped (Lw>L) components' energy is spread proportionally
# across the surviving resolvable spectrum instead of being represented at a
# single domain-filling wavenumber -- avoids the large-scale-structure
# artifact seen in mainnm 14 (see chat). From the current (updated)
# IW_GM_flux_LAT_2000km_bash_cuda.jl; params_15_noforce.jl
append!(RUN_TABLE, expand_block(15, collect(1:13),  LAT13, fill(0.0,13), 200, "zonalmean", 99))

# mainnm 15, runnm 27:39 (200 m, Garrett-Munk + M2 tidal forcing) ------------
# same as mainnm 13's runnm 27:39 GM+tide block (25 kW/m, matching the 10-13
# series for direct comparison), but with the redistribution-fix GM IC;
# params_15.jl
append!(RUN_TABLE, expand_block(15, collect(27:39), LAT13, fill(25e3,13), 200, "zonalmean", 99))

# mainnm 16 (200 m, GM81 initial condition with the CORRECTED amplitude) -------
# claudecodes/IW_GM81_flux_LAT_2000km_bash_cuda.jl + claudecodes/gm81_ic.jl:
# A² = 2 b² N0 <N> E0 B H Δω (series 13-15 used b² N0² (1+f²/ω²), ~3x GM81),
# model f in the modes/polarization (v = 0 at the equator), and the IC
# rescaled to E(0) = GMs x E_GM81 so that the day 10-20 mean is 1x GM81.
# runnm 91:93: calibration, GM only (Flux=0), hourly output; params_16_cal.jl
append!(RUN_TABLE, expand_block(16, [91, 92, 93], [0.0, 5.0, 28.8], fill(0.0,3), 200, "zonalmean", 99))
# runnm 1:12 (GM only, Flux=0) and 27:38 (GM + 25 kW/m D2 tide), lat 0-45 N:
# PLANNED -- GMs per latitude set after the calibration runs
append!(RUN_TABLE, expand_block(16, collect(1:12),  LAT13[1:12], fill(0.0,12),  200, "zonalmean", 99))
append!(RUN_TABLE, expand_block(16, collect(27:38), LAT13[1:12], fill(25e3,12), 200, "zonalmean", 99))

# --- lookup helpers -----------------------------------------------------------

# return RUN_TABLE rows for (mainnm, runnms), in the order runnms was given;
# errors immediately if any (mainnm, runnm) pair is not in the table, instead of
# silently pairing it with the wrong latitude/N2 profile downstream
function get_runs(mainnm::Integer, runnms)
    rows = RunInfo[]
    for rn in runnms
        idx = findfirst(r -> r.mainnm == mainnm && r.runnm == rn, RUN_TABLE)
        idx === nothing && error("run_master.jl: no entry for mainnm=$mainnm, runnm=$rn -- add it to RUN_TABLE")
        push!(rows, RUN_TABLE[idx])
    end
    return rows
end

# N2 forcing filename for a run, matching the naming convention written by the
# N2 zonal-mean extraction pipeline (N2_ZonalMeanAtl_lat%04.1f.jld2)
function n2_filename(row::RunInfo)
    if row.N2source == "zonalmean"
        return @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", row.lat)
    elseif row.N2source == "zonalmeanfixed"
        return @sprintf("N2_ZonalMeanAtl_lat%04.1f.jld2", row.latfix)
    elseif row.N2source == "amz1"
        return "N2_amz1.jld2"   # constant WOCE AMZ N2 (mainnm 9 sims)
    else
        error("run_master.jl: unknown N2source '$(row.N2source)' for mainnm=$(row.mainnm), runnm=$(row.runnm)")
    end
end

# KE/APE/flux plot y-limits, keyed off the run's forcing flux magnitude
function elim_flim(row::RunInfo)
    if row.Flux <= 15e3
        return [0, 10], [0, 17]  
    elseif row.Flux > 15e3 && row.Flux <= 25e3 
        return [0, 20], [0, 26]
    else
        return [0, 40], [0, 52]
    end
end
