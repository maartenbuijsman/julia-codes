#= run_master.jl
Maarten Buijsman, USM DMS, 2026-8-15
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
append!(RUN_TABLE, expand_block(11, collect(1:13),  LAT13, fill(15e3,13), 200,  "zonalmean",      99))
append!(RUN_TABLE, expand_block(11, collect(14:26), LAT13, fill(15e3,13), 200,  "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(11, collect(27:39), LAT13, fill(25e3,13), 200,  "zonalmean",      99))
append!(RUN_TABLE, expand_block(11, collect(40:52), LAT13, fill(25e3,13), 200,  "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(11, collect(53:65), LAT13, fill(25e3,13), 200,  "zonalmeanfixed", 50.0))

# mainnm 10 (4000 m D2 NH flux forcing) ---------------------------------------
append!(RUN_TABLE, expand_block(10, collect(1:13),  LAT13, fill(15e3,13), 4000, "zonalmean",      99))
append!(RUN_TABLE, expand_block(10, collect(14:26), LAT13, fill(15e3,13), 4000, "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(10, collect(27:39), LAT13, fill(25e3,13), 4000, "zonalmean",      99))
append!(RUN_TABLE, expand_block(10, collect(40:52), LAT13, fill(25e3,13), 4000, "zonalmeanfixed", 2.5))
append!(RUN_TABLE, expand_block(10, collect(53:65), LAT13, fill(25e3,13), 4000, "zonalmeanfixed", 50.0))

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
        return [0, 12], [0, 17]
    else
        return [0, 22], [0, 26]
    end
end
