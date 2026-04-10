#####
##### Download and process Lake Superior Western Mooring observations.
#####
# Identifies the start of winter cooling (last time the full water column
# is isothermal at ~4°C in Oct–Dec), then extracts the mixed layer depth
# (MLD) at +30 and +60 days after that start.
#
# Inputs  : two hourly-data ZIP archives from the UMN Data Repository
# Outputs : figure_data/lake_superior_western_mooring/western_mooring_observed_mld.jld2
#####

using Downloads
using MAT
using JLD2
using Dates
using Statistics
using CairoMakie
using LaTeXStrings

const DATA_DIR   = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/Austin2023"
const OUTPUT_DIR = joinpath(@__DIR__, "..", "figure_data", "lake_superior_western_mooring")

mkpath(DATA_DIR)
mkpath(OUTPUT_DIR)

# ── Download ─────────────────────────────────────────────────────────────────

download_specs = [
    ("https://conservancy.umn.edu/bitstreams/d70f1508-6393-40b5-bce6-581300613daf/download",
     "hourly_data_2005_2015.zip"),
    ("https://conservancy.umn.edu/bitstreams/8cb700d5-2a6f-487a-b7c4-e56bf7ea0ec1/download",
     "hourly_data_2015_2021.zip"),
]

for (url, fname) in download_specs
    fpath = joinpath(DATA_DIR, fname)
    if !isfile(fpath)
        @info "Downloading $fname …"
        Downloads.download(url, fpath)
    else
        @info "$fname already present, skipping download."
    end
    @info "Unzipping $fname …"
    run(`unzip -n $fpath -d $DATA_DIR`)
end

# ── Time conversion ───────────────────────────────────────────────────────────
# t is MATLAB datenum (days since Jan 0, year 0); datenum(2000,1,1) = 730486
t2dt(t::Real) = DateTime(2000, 1, 1) + Millisecond(round(Int64, (t - 730486.0) * 86_400_000.0))

# ── Mixed layer depth ─────────────────────────────────────────────────────────
# dep: positive-downward depth vector (m), sorted shallow → deep.
# T_col: temperature (°C) at those depths.
# Uses same ΔT = 0.1°C threshold as plot_lake_superior_western_mooring.jl.
const ΔT_mld = 0.1   # °C

const H_OBS = 184.0   # maximum meaningful MLD (lake depth, m)

function mld_from_obs(dep, T_col)
    idx   = sortperm(dep)            # shallow (small dep) first
    d     = dep[idx]
    T     = T_col[idx]
    T_sfc = T[1]
    for i in 2:length(T)
        if abs(T[i] - T_sfc) > ΔT_mld
            return min(d[i], H_OBS)
        end
    end
    return H_OBS                     # fully mixed
end

# ── Common depth grid (matches model: 23 levels, dz = 8 m) ───────────────────
const DEP_COMMON = collect(4.0:8.0:180.0)   # 23 levels, shallow → deep (m)

# Piecewise-linear interpolation of a profile onto DEP_COMMON.
# dep_in must be sorted shallow → deep with no NaNs.
function interp_to_common(dep_in, T_in)
    T_out = similar(DEP_COMMON)
    for (j, d) in enumerate(DEP_COMMON)
        if d <= dep_in[1]
            T_out[j] = T_in[1]
        elseif d >= dep_in[end]
            T_out[j] = T_in[end]
        else
            i = searchsortedlast(dep_in, d)
            α = (d - dep_in[i]) / (dep_in[i+1] - dep_in[i])
            T_out[j] = T_in[i] + α * (T_in[i+1] - T_in[i])
        end
    end
    return T_out
end

# ── Process one .mat file ─────────────────────────────────────────────────────
function process_mat(filepath)
    data = try
        matread(filepath)
    catch e
        @warn "Could not read $(basename(filepath)): $e"
        return nothing
    end

    # Require temperature, depth, and time
    for v in ("T", "dep", "t")
        haskey(data, v) || return nothing
    end

    dep   = vec(Float64.(reshape(data["dep"], :)))
    t_raw = vec(Float64.(reshape(data["t"],   :)))
    T_raw = data["T"]

    Ndep = length(dep)
    Nt   = length(t_raw)

    # Ensure T is (Ndep × Nt)
    T = if size(T_raw) == (Ndep, Nt)
        Float64.(T_raw)
    elseif size(T_raw) == (Nt, Ndep)
        Float64.(T_raw')
    else
        @warn "Unexpected T size $(size(T_raw)) in $(basename(filepath))"
        return nothing
    end

    datetimes = t2dt.(t_raw)

    # ── Find last isothermal-4°C timestep in Oct–Dec ──────────────────────────
    last_iso_i = 0
    for i in eachindex(datetimes)
        month(datetimes[i]) ∈ (11, 12, 1) || continue
        col = T[:, i]
        any(isnan, col) && continue
        spread = maximum(col) - minimum(col)
        tmean  = mean(col)
        if spread < 0.3 && 3.97 ≤ tmean ≤ 4.1
            last_iso_i = i
        end
    end
    if last_iso_i == 0
        @warn "$(basename(filepath)): no isothermal 4°C state found in Nov–Jan"
        return nothing
    end

    start_dt = datetimes[last_iso_i]

    # ── Find nearest timestep to start + 30 and start + 60 days ──────────────
    dt_ms    = Dates.value.(datetimes)
    t1_i = argmin(abs.(dt_ms .- Dates.value(start_dt + Day(30))))
    t2_i = argmin(abs.(dt_ms .- Dates.value(start_dt + Day(60))))

    # Require the nearest time is within ±3 days of target
    if abs(datetimes[t1_i] - (start_dt + Day(30))) > Day(3)
        @warn "$(basename(filepath)): no data within 3 days of start+30 (deployment ends $(datetimes[end]))"
        return nothing
    end
    if abs(datetimes[t2_i] - (start_dt + Day(60))) > Day(3)
        @warn "$(basename(filepath)): no data within 3 days of start+60 (deployment ends $(datetimes[end]))"
        return nothing
    end

    col1 = T[:, t1_i]
    col2 = T[:, t2_i]
    (any(isnan, col1) || any(isnan, col2)) && return nothing

    mld1 = mld_from_obs(dep, col1)
    mld2 = mld_from_obs(dep, col2)

    # Sort shallow → deep then interpolate onto common grid
    idx     = sortperm(dep)
    d_sort  = dep[idx]
    T_ini   = interp_to_common(d_sort, T[:, last_iso_i][idx])
    T_1mo   = interp_to_common(d_sort, col1[idx])
    T_2mo   = interp_to_common(d_sort, col2[idx])
    Tbot1 = T_1mo[end]   # bottom temperature (+1 month) at deepest DEP_COMMON level
    Tbot2 = T_2mo[end]   # bottom temperature (+2 months)
    return (start_dt, mld1, mld2, T_ini, T_1mo, T_2mo, Tbot1, Tbot2)
end

# ── Collect western mooring .mat files (WM*.mat) ─────────────────────────────
mat_files = String[]
for (root, _, files) in walkdir(DATA_DIR)
    for f in files
        startswith(f, "WM") && endswith(f, ".mat") && push!(mat_files, joinpath(root, f))
    end
end
@info "Found $(length(mat_files)) WM*.mat files in $DATA_DIR"

results_raw = filter(!isnothing, [process_mat(f) for f in mat_files])
@info "Raw valid results before deduplication: $(length(results_raw))"
isempty(results_raw) && error("No valid results — check variable names or isothermal threshold.")

# Deduplicate: group by winter year (Nov/Dec → year+1, Jan → year) and keep
# the earliest start date per season (= the actual 4°C overturn event).
winter_year(dt) = month(dt) >= 11 ? year(dt) + 1 : year(dt)
season_map = Dict{Int, eltype(results_raw)}()
for r in results_raw
    wy = winter_year(r[1])
    if !haskey(season_map, wy) || r[1] < season_map[wy][1]
        season_map[wy] = r
    end
end
results = [season_map[wy] for wy in sort(collect(keys(season_map)))]
@info "Valid western mooring winter seasons after deduplication: $(length(results))"

start_dates  = [r[1] for r in results]
mld1_obs     = Float64[r[2] for r in results]
mld2_obs     = Float64[r[3] for r in results]
T0_profiles  = hcat([r[4] for r in results]...)   # initial (isothermal), (Ndep × Nseasons)
T1_profiles  = hcat([r[5] for r in results]...)   # +1 month
T2_profiles  = hcat([r[6] for r in results]...)   # +2 months
Tbot1_obs    = Float64[r[7] for r in results]     # bottom T at +1 month (deepest DEP_COMMON = 180m)
Tbot2_obs    = Float64[r[8] for r in results]     # bottom T at +2 months

@info "MLD at +1 month (m):   min=$(minimum(mld1_obs)), median=$(median(mld1_obs)), max=$(maximum(mld1_obs))"
@info "MLD at +2 months (m):  min=$(minimum(mld2_obs)), median=$(median(mld2_obs)), max=$(maximum(mld2_obs))"
@info "Tbot at +1 month (°C): min=$(minimum(Tbot1_obs)), median=$(median(Tbot1_obs)), max=$(maximum(Tbot1_obs))"
@info "Tbot at +2 months (°C):min=$(minimum(Tbot2_obs)), median=$(median(Tbot2_obs)), max=$(maximum(Tbot2_obs))"

# ── Write start-date table ────────────────────────────────────────────────────
csv_file = joinpath(OUTPUT_DIR, "western_mooring_winter_start_dates.csv")
open(csv_file, "w") do io
    println(io, "winter_year,start_date,mld_1month_m,mld_2month_m")
    for (dt, mld1, mld2) in zip(start_dates, mld1_obs, mld2_obs)
        wy = month(dt) >= 11 ? year(dt) + 1 : year(dt)
        println(io, "$(wy),$(Date(dt)),$(round(mld1, digits=1)),$(round(mld2, digits=1))")
    end
end
@info "Saved start dates → $csv_file"

outfile = joinpath(OUTPUT_DIR, "western_mooring_observed_mld.jld2")
jldsave(outfile;
    start_dates  = string.(start_dates),
    mld1_obs,
    mld2_obs,
    Tbot1_obs,
    Tbot2_obs,
    dep_obs      = DEP_COMMON,
    T0_profiles,
    T1_profiles,
    T2_profiles)
@info "Saved → $outfile"

#####
##### Figure — Observed temperature profiles at t=0, +1 month, +2 months
#####

const H_lake     = 184.0   # m
const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

z_obs = -DEP_COMMON   # convert positive-downward dep to negative z

with_theme(theme_latexfonts()) do

    fig = Figure(size = (500, 380), fontsize = 12,
                 figure_padding = (4, 8, 4, 4))

    time_panels_obs = [
        (T1_profiles, "After 1 month"),
        (T2_profiles, "After 2 months"),
    ]
    Nseasons = size(T0_profiles, 2)

    ax_obs = Axis[]
    for (panel, (T_profiles, panel_title)) in enumerate(time_panels_obs)
        ax = CairoMakie.Axis(fig[1, panel],
                 xlabel = L"T \; (^\circ\mathrm{C})",
                 ylabel = panel == 1 ? L"z \; \mathrm{(m)}" : "",
                 title  = panel_title,
                 xticks = 0:1:5)
        push!(ax_obs, ax)

        # Initial (isothermal) profiles — dashed blue
        for s in 1:Nseasons
            lines!(ax, T0_profiles[:, s], z_obs,
                   color = (:blue, 0.3), linewidth = 1.0, linestyle = :dash)
        end
        T0_median = median(T0_profiles, dims = 2)[:, 1]
        lines!(ax, T0_median, z_obs,
               color = :blue, linewidth = 2.0, linestyle = :dash)

        # Profiles at target time
        for s in 1:Nseasons
            lines!(ax, T_profiles[:, s], z_obs,
                   color = (:steelblue, 0.4), linewidth = 1.2)
        end
        T_median = median(T_profiles, dims = 2)[:, 1]
        lines!(ax, T_median, z_obs,
               color = :steelblue, linewidth = 2.5)

        xlims!(ax, (0.0, 4.3))
        ylims!(ax, (-H_lake - 5, 5))
        panel > 1 && hideydecorations!(ax, ticks = false, grid = false)
    end

    linkaxes!(ax_obs...)

    # Legend: t=0 isothermal | bold median | thin individual years
    legend_elems = [
        LineElement(color = :blue,                linewidth = 2.0, linestyle = :dash),
        LineElement(color = :steelblue,           linewidth = 2.5, linestyle = :solid),
        LineElement(color = (:steelblue, 0.4),    linewidth = 1.2, linestyle = :solid),
    ]
    legend_labels = ["t = 0 (isothermal, median)", "Median", "Individual years"]
    Legend(fig[2, :], legend_elems, legend_labels,
           orientation = :horizontal, tellwidth = false, labelsize = 11,
           framevisible = false, padding = (0, 0, 0, 0))

    Label(fig[0, :],
          "Lake Superior, Western Mooring\nObserved temperature profiles",
          fontsize = 13, font = :bold, justification = :center)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 4)

    outfig = joinpath(FIGURE_DIR, "lake_superior_western_mooring_obs_T_profiles.pdf")
    save(outfig, fig)
    @info "Saved → $outfig"
end
