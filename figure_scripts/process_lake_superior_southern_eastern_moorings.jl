#####
##### Analyse Lake Superior Southern (SM) and Eastern (EM) Mooring observations.
#####
# For each mooring type (SM / EM), finds the last isothermal ~4 °C state in
# Nov–Jan, then extracts and plots temperature profiles at +1 month and +2 months.
# Each plotted profile is labelled with its winter year.
#
# Inputs  : Austin2023 hourly .mat files in /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/Austin2023/
# Outputs : figures/lake_superior_{southern,eastern}_mooring_obs_T_profiles.pdf
#           figure_data/lake_superior_{southern,eastern}_mooring/observed_mld.jld2
#####

using MAT
using JLD2
using Dates
using Statistics
using CairoMakie
using LaTeXStrings

const DATA_DIR = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/Austin2023"

const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# ── Time conversion ────────────────────────────────────────────────────────────
# MATLAB datenum: days since Jan 0, year 0; datenum(2000,1,1) = 730486
t2dt(t::Real) = DateTime(2000, 1, 1) + Millisecond(round(Int64, (t - 730486.0) * 86_400_000.0))

# ── MLD from observed profile ──────────────────────────────────────────────────
const ΔT_mld = 0.1   # °C threshold

# Max physically-plausible daily-mean current speed (m/s) for the ADCP obs;
# higher values are near-surface side-lobe / surface contamination and discarded.
const MAX_OBS_SPEED = 0.4

function mld_from_obs(dep, T_col, H_obs)
    idx   = sortperm(dep)
    d     = dep[idx]
    T     = T_col[idx]
    T_sfc = T[1]
    for i in 2:length(T)
        if abs(T[i] - T_sfc) > ΔT_mld
            return min(d[i], H_obs)
        end
    end
    return H_obs
end

# ── Common depth grid for a given max depth (8 m spacing) ─────────────────────
make_dep_common(H_obs) = collect(4.0:8.0:H_obs)

function interp_to_common(dep_common, dep_in, T_in)
    T_out = similar(dep_common)
    for (j, d) in enumerate(dep_common)
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

# ── Process one .mat file ──────────────────────────────────────────────────────
function process_mat(filepath, dep_common; T4_tol = 0.05, start_override = nothing)
    data = try
        matread(filepath)
    catch e
        @warn "Could not read $(basename(filepath)): $e"
        return nothing
    end

    for v in ("T", "dep", "t")
        haskey(data, v) || return nothing
    end

    dep   = vec(Float64.(reshape(data["dep"], :)))
    t_raw = vec(Float64.(reshape(data["t"],   :)))
    T_raw = data["T"]

    Ndep = length(dep)
    Nt   = length(t_raw)

    T = if size(T_raw) == (Ndep, Nt)
        Float64.(T_raw)
    elseif size(T_raw) == (Nt, Ndep)
        Float64.(T_raw')
    else
        @warn "Unexpected T size $(size(T_raw)) in $(basename(filepath))"
        return nothing
    end

    datetimes = t2dt.(t_raw)

    if start_override === nothing
        # Pick the initial-condition timestep as the freshly-homogenized winter
        # plateau: the most isothermal (minimum top-to-bottom spread) column near
        # 4 °C, before deep cooling sets in.  The column cools through 4 °C in late
        # Dec / early Jan, but keying on the *last* in-band timestep (or on the column
        # mean) can land on the cool, partly-stratified side of that transition;
        # minimizing spread among near-4 °C columns instead favors the uniform plateau.
        # Restricted to 00:00 snapshots so the IC matches the model's midnight start
        # time (and so the saved t=0 profile is consistent with start_dt below).
        # T4_tol (kwarg): |column mean − 4 °C| tolerance for a candidate IC
        last_iso_i  = 0
        best_spread = Inf
        for i in eachindex(datetimes)
            month(datetimes[i]) ∈ (11, 12, 1) || continue
            hour(datetimes[i]) == 0           || continue   # model starts at 00:00
            col = T[:, i]
            any(isnan, col) && continue
            abs(mean(col) - 4.0) ≤ T4_tol || continue
            spread = maximum(col) - minimum(col)
            if spread < best_spread
                best_spread = spread
                last_iso_i  = i
            end
        end
        if last_iso_i == 0
            @warn "$(basename(filepath)): no isothermal 4°C state found in Nov–Jan"
            return nothing
        end
    else
        # Use externally-provided start dates (winter_year => DateTime), e.g. the
        # old MLD dates the LES runs were initialized on.  Select the override date
        # that falls within this file's coverage (leaving ≥60 days after it) and
        # snap to the nearest available timestep.
        cand = nothing
        for (_, sd) in start_override
            (datetimes[1] <= sd <= datetimes[end] - Day(60)) || continue
            cand = sd
            break
        end
        isnothing(cand) && return nothing
        last_iso_i = argmin(abs.(Dates.value.(datetimes .- cand)))
    end

    # Floor to midnight so target days align with model calendar days
    start_dt = DateTime(Date(datetimes[last_iso_i]))

    # Daily averages: mean of all hourly obs in [target, target + 1 day)
    target1  = start_dt + Day(30)
    target2  = start_dt + Day(60)
    targetd15 = start_dt + Day(15)
    targetd45 = start_dt + Day(45)
    idxs1  = findall(dt -> target1  <= dt < target1  + Day(1), datetimes)
    idxs2  = findall(dt -> target2  <= dt < target2  + Day(1), datetimes)
    idxsd15 = findall(dt -> targetd15 <= dt < targetd15 + Day(1), datetimes)
    idxsd45 = findall(dt -> targetd45 <= dt < targetd45 + Day(1), datetimes)

    if isempty(idxs1)
        @warn "$(basename(filepath)): no data in [start+30, start+31) days (ends $(datetimes[end]))"
        return nothing
    end
    if isempty(idxs2)
        @warn "$(basename(filepath)): no data in [start+60, start+61) days (ends $(datetimes[end]))"
        return nothing
    end
    if isempty(idxsd15)
        @warn "$(basename(filepath)): no data in [start+15, start+16) days (ends $(datetimes[end]))"
        return nothing
    end
    if isempty(idxsd45)
        @warn "$(basename(filepath)): no data in [start+45, start+46) days (ends $(datetimes[end]))"
        return nothing
    end

    any(any(isnan, T[:, i]) for i in idxs1) && return nothing
    any(any(isnan, T[:, i]) for i in idxs2) && return nothing
    any(any(isnan, T[:, i]) for i in idxsd15) && return nothing
    any(any(isnan, T[:, i]) for i in idxsd45) && return nothing

    col1   = vec(mean(T[:, idxs1], dims = 2))
    col2   = vec(mean(T[:, idxs2], dims = 2))
    cold15 = vec(mean(T[:, idxsd15], dims = 2))
    cold45 = vec(mean(T[:, idxsd45], dims = 2))

    H_obs = dep_common[end]
    mld1  = mld_from_obs(dep, col1, H_obs)
    mld2  = mld_from_obs(dep, col2, H_obs)

    idx    = sortperm(dep)
    d_sort = dep[idx]
    T_ini  = interp_to_common(dep_common, d_sort, T[:, last_iso_i][idx])
    T_1mo  = interp_to_common(dep_common, d_sort, col1[idx])
    T_2mo  = interp_to_common(dep_common, d_sort, col2[idx])

    # ADCP total current speed sqrt(east²+north²) — present in EM files, absent in SM files
    spd_bins_raw = nothing
    spd1_raw     = nothing
    spd2_raw     = nothing
    if haskey(data, "east_vel") && haskey(data, "north_vel") && haskey(data, "bins")
        bins_raw = vec(Float64.(reshape(data["bins"], :)))
        Nbins    = length(bins_raw)
        function reshape_vel(v_raw)
            if size(v_raw) == (Nt, Nbins); Float64.(v_raw')
            elseif size(v_raw) == (Nbins, Nt); Float64.(v_raw)
            else; nothing; end
        end
        ev = reshape_vel(data["east_vel"])
        nv = reshape_vel(data["north_vel"])
        if !isnothing(ev) && !isnothing(nv)
            spd = sqrt.(ev.^2 .+ nv.^2)   # (Nbins, Nt)
            idxs1_spd = filter(i -> !any(isnan, spd[:, i]), idxs1)
            idxs2_spd = filter(i -> !any(isnan, spd[:, i]), idxs2)
            if !isempty(idxs1_spd) && !isempty(idxs2_spd)
                spd_bins_raw = bins_raw
                spd1_raw     = vec(mean(spd[:, idxs1_spd], dims = 2))
                spd2_raw     = vec(mean(spd[:, idxs2_spd], dims = 2))
                # Discard near-surface ADCP side-lobe / surface contamination:
                # winter Lake Superior currents are ~0.05–0.2 m/s, so daily-mean
                # speeds above MAX_OBS_SPEED are non-physical (e.g. EM 2009's top
                # bins read ~0.7–1.1 m/s).  NaN'd bins are skipped when plotting.
                spd1_raw[spd1_raw .> MAX_OBS_SPEED] .= NaN
                spd2_raw[spd2_raw .> MAX_OBS_SPEED] .= NaN
            end
        end
    end

    return (start_dt, mld1, mld2, T_ini, T_1mo, T_2mo,
            d_sort, col1[idx], col2[idx],
            spd_bins_raw, spd1_raw, spd2_raw,
            T[:, last_iso_i][idx],   # raw isothermal (t=0) temps at sensor depths
            cold15[idx], cold45[idx])   # raw temps at sensor depths, day 15 / day 45
end

# ── Helper: collect + deduplicate results for a given file prefix ──────────────
winter_year(dt) = month(dt) >= 11 ? year(dt) + 1 : year(dt)

function collect_results(prefix; T4_tol = 0.05, start_override = nothing)
    mat_files = String[]
    for (root, _, files) in walkdir(DATA_DIR)
        for f in files
            startswith(f, prefix) && endswith(f, ".mat") && push!(mat_files, joinpath(root, f))
        end
    end
    @info "Found $(length(mat_files)) $(prefix)*.mat files"

    # Determine common depth grid from max depth across all files for this prefix
    max_depth = 0.0
    for f in mat_files
        data = try matread(f) catch; continue end
        haskey(data, "dep") || continue
        max_depth = max(max_depth, maximum(vec(Float64.(reshape(data["dep"], :)))))
    end
    dep_common = make_dep_common(max_depth)
    @info "$(prefix): max observed depth = $(max_depth) m → $(length(dep_common))-level common grid (bottom = $(dep_common[end]) m)"

    results_raw = filter(!isnothing, [process_mat(f, dep_common; T4_tol, start_override) for f in mat_files])
    @info "Valid results before deduplication: $(length(results_raw))"
    isempty(results_raw) && return nothing

    # Keep earliest start date per winter season
    season_map = Dict{Int, eltype(results_raw)}()
    for r in results_raw
        wy = winter_year(r[1])
        if !haskey(season_map, wy) || r[1] < season_map[wy][1]
            season_map[wy] = r
        end
    end
    wys     = sort(collect(keys(season_map)))
    results = [season_map[wy] for wy in wys]
    @info "Valid winter seasons after deduplication: $(length(results))"
    return wys, results, dep_common
end

# ── Save JLD2 and make figure for one mooring type ────────────────────────────
function process_and_plot(prefix, mooring_name, jld2_subdir, fig_stem;
                          T4_tol = 0.05, start_override = nothing,
                          tag = "", write_csv = true)
    out = collect_results(prefix; T4_tol, start_override)
    isnothing(out) && return

    wys, results, dep_common = out

    start_dates  = [r[1] for r in results]
    mld1_obs     = Float64[r[2] for r in results]
    mld2_obs     = Float64[r[3] for r in results]
    T0_profiles  = hcat([r[4] for r in results]...)
    T1_profiles  = hcat([r[5] for r in results]...)
    T2_profiles  = hcat([r[6] for r in results]...)
    dep_raw_obs  = [r[7]  for r in results]   # Vector{Vector{Float64}}: sensor depths per year
    T1_raw_obs   = [r[8]  for r in results]   # temperatures at sensor depths, day 30
    T2_raw_obs   = [r[9]  for r in results]   # temperatures at sensor depths, day 60
    spd_bins_obs = [r[10] for r in results]   # ADCP bin depths (nothing if no ADCP)
    spd1_raw_obs = [r[11] for r in results]   # total current speed at bins, day 30
    spd2_raw_obs = [r[12] for r in results]   # total current speed at bins, day 60
    T0_raw_obs   = [r[13] for r in results]   # isothermal (t=0) temps at sensor depths
    T15_raw_obs  = [r[14] for r in results]   # temperatures at sensor depths, day 15
    T45_raw_obs  = [r[15] for r in results]   # temperatures at sensor depths, day 45

    # Save
    out_dir = joinpath(@__DIR__, "..", "figure_data", jld2_subdir)
    mkpath(out_dir)
    outfile = joinpath(out_dir, "observed_mld$(tag).jld2")
    jldsave(outfile;
        winter_years = wys,
        start_dates  = string.(start_dates),
        mld1_obs,
        mld2_obs,
        dep_obs      = dep_common,
        T0_profiles,
        T1_profiles,
        T2_profiles,
        dep_raw_obs,
        T0_raw_obs,
        T15_raw_obs,
        T1_raw_obs,
        T45_raw_obs,
        T2_raw_obs,
        spd_bins_obs,
        spd1_raw_obs,
        spd2_raw_obs)
    @info "Saved → $outfile"

    if write_csv
        csv_file = joinpath(out_dir, "$(fig_stem)_winter_start_dates.csv")
        open(csv_file, "w") do io
            println(io, "winter_year,start_date,mld_1month_m,mld_2month_m")
            for (wy, dt, mld1, mld2) in zip(wys, start_dates, mld1_obs, mld2_obs)
                println(io, "$(wy),$(Date(dt)),$(round(mld1, digits=1)),$(round(mld2, digits=1))")
            end
        end
        @info "Saved start dates → $csv_file"
    end

    # ── Figure ────────────────────────────────────────────────────────────────
    Nseasons = length(results)
    H_obs    = dep_common[end]
    z_obs    = -dep_common

    get_color(s) = Makie.ColorSchemes.tableau_20[mod1(s, 20)]

    with_theme(theme_latexfonts()) do

        fig = Figure(size = (620, 420), fontsize = 12,
                     figure_padding = (4, 10, 4, 4))

        time_panels = [
            (T1_profiles, T1_raw_obs, "After 1 month"),
            (T2_profiles, T2_raw_obs, "After 2 months"),
        ]

        axs = Axis[]
        for (panel, (T_profiles, T_raw, panel_title)) in enumerate(time_panels)
            ax = CairoMakie.Axis(fig[1, panel],
                     xlabel = L"T \; (^\circ\mathrm{C})",
                     ylabel = panel == 1 ? L"z \; \mathrm{(m)}" : "",
                     title  = panel_title,
                     xticks = 0:1:5)
            push!(axs, ax)

            # Initial isothermal (t=0) profiles: individual lines + sensor markers (all grey)
            for s in 1:Nseasons
                lines!(ax, T0_profiles[:, s], z_obs,
                       color = (:grey50, 0.5), linewidth = 0.9, linestyle = :dash)
                scatter!(ax, T0_raw_obs[s], -dep_raw_obs[s];
                         color = (:grey50, 0.7), marker = :circle, markersize = 5)
            end

            # Individual year profiles: interpolated line + raw sensor markers
            for s in 1:Nseasons
                col = get_color(s)
                lines!(ax, T_profiles[:, s], z_obs,
                       color = (col, 0.7), linewidth = 1.4)
                scatter!(ax, T_raw[s], -dep_raw_obs[s];
                         color = (col, 0.9), marker = :circle, markersize = 5)
                # Label at the shallowest point
                text!(ax, T_profiles[1, s] + 0.04, z_obs[1] - (s-1)*6;
                      text  = string(wys[s]),
                      color = col,
                      fontsize = 8,
                      align = (:left, :center))
            end

            # Bold median
            lines!(ax, median(T_profiles, dims=2)[:, 1], z_obs,
                   color = :black, linewidth = 2.2, linestyle = :solid)

            xlims!(ax, (0.0, 5.0))
            ylims!(ax, (-H_obs - 5, 5))
            panel > 1 && hideydecorations!(ax, ticks=false, grid=false)
        end

        linkaxes!(axs...)

        legend_elems = [
            [LineElement(color = (:grey50, 0.7), linewidth = 0.9, linestyle = :dash),
             MarkerElement(color = (:grey50, 0.7), marker = :circle, markersize = 5)],
            [LineElement(color = :black,  linewidth = 1.4),
             MarkerElement(color = :black, marker = :circle, markersize = 5)],
            LineElement(color = :black,   linewidth = 2.2, linestyle = :solid),
        ]
        legend_labels = ["t = 0 isothermal (interp. line + sensor markers)",
                         "Individual year (interp. line + sensor markers)",
                         "Median (+1 or +2 mo)"]
        Legend(fig[2, :], legend_elems, legend_labels,
               orientation = :horizontal, tellwidth = false, labelsize = 10,
               framevisible = false, padding = (0, 0, 0, 0))

        Label(fig[0, :],
              "Lake Superior, $(mooring_name)\nObserved temperature profiles",
              fontsize = 13, font = :bold, justification = :center)

        colgap!(fig.layout, 8)
        rowgap!(fig.layout, 4)

        outfig = joinpath(FIGURE_DIR, "$(fig_stem)$(tag)_obs_T_profiles.pdf")
        save(outfig, fig)
        @info "Saved → $outfig"
    end
end

# ── Run for both mooring types ─────────────────────────────────────────────────
# EM homogenizes cleanly at 4 °C, so a tight tolerance keeps its ICs right at 4.
# SM never sits that close to 4 °C while well-mixed; ±0.025 would drop both SM
# winters, so it uses a looser ±0.05 window.
process_and_plot("SM", "Southern Mooring",
                 "lake_superior_southern_mooring",
                 "lake_superior_southern_mooring"; T4_tol = 0.05)

process_and_plot("EM", "Eastern Mooring",
                 "lake_superior_eastern_mooring",
                 "lake_superior_eastern_mooring"; T4_tol = 0.025)

# ── Eastern Mooring obs at the OLD MLD start dates, for LES comparison ─────────
# The LES runs were initialized on the old MLD dates (…_winter_start_dates_LES.csv),
# so regenerate a matching obs file (observed_mld_LES.jld2) with the same full
# format (raw T markers + ADCP speed) but pinned to those dates rather than the
# re-derived isothermal dates.
function read_start_dates_csv(csv_file)
    isfile(csv_file) || (@warn "LES start-date CSV not found: $csv_file"; return nothing)
    dates = Dict{Int, DateTime}()
    open(csv_file) do io
        readline(io)   # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) >= 2 || continue
            wy = tryparse(Int, strip(parts[1]))
            isnothing(wy) && continue
            dates[wy] = DateTime(Date(strip(parts[2])))
        end
    end
    return dates
end

les_csv   = joinpath(@__DIR__, "..", "figure_data", "lake_superior_eastern_mooring",
                     "lake_superior_eastern_mooring_winter_start_dates_LES.csv")
les_dates = read_start_dates_csv(les_csv)
if !isnothing(les_dates)
    process_and_plot("EM", "Eastern Mooring (LES dates)",
                     "lake_superior_eastern_mooring",
                     "lake_superior_eastern_mooring";
                     start_override = les_dates, tag = "_LES", write_csv = false)
end
