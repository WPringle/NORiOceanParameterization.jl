#####
##### GLEN-forced Lake Superior Moorings — depth-averaged temperature evolution
#####
# Implements the analysis suggested by the reviewer: evaluate the column-integrated
# heat budget rather than infer it from T_bot. Since the initial temperature profile
# and surface heat flux are known, the expected evolution of depth-averaged temperature
# can be calculated directly from heat conservation and compared with the observed
# depth-averaged temperature. This tests whether surface forcing is sufficient and
# provides evidence for additional heat-budget terms (e.g. horizontal transport).
#
# Compares the depth-averaged temperature evolution predicted by integrating surface
# heat fluxes against the observed depth-averaged temperature for the same years and
# moorings as the Tbot timeseries figure, for both direct and coare_wind forcing.
#
# Inputs  : GLEN forcing (direct and coare_wind variants)
#           Austin2023 hourly .mat files (observed temperature profiles)
# Outputs : figures/lake_superior_mooring_depth_averaged_temp_evolution.pdf
#
# Usage:
#   julia plot_heat_flux_vs_depth_averaged_temp_change.jl
#####

using Oceananigans
using CairoMakie
using JLD2
using MAT
using NCDatasets
using Dates
using LaTeXStrings
using GibbsSeaWater
using Statistics

const S_lake    = 0.05    # g/kg
const ρ₀        = 999.8
const g_grav    = 9.80665
const cₚ        = 4182.0  # J/(kg·K)
const albedo_sw = 0.08
const ε_water   = 0.98

const OBS_DIR  = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/Austin2023"
FIGURE_DIR      = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# ── Mooring site definitions ───────────────────────────────────────────────────
struct MooringSite
    name       :: String
    turb_dir   :: String
    les_dir    :: Union{String, Nothing}
    csv_file   :: String
    obs_prefix :: String    # "EM" or "SM"
end

const EM_CSV_FILE = joinpath(@__DIR__, "..", "figure_data", "lake_superior_eastern_mooring",
                              "lake_superior_eastern_mooring_winter_start_dates.csv")

const EM_SITE = MooringSite(
    "Eastern",
    joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced"),
    joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN"),
    EM_CSV_FILE,
    "EM")

const SM_SITE = MooringSite(
    "Southern",
    joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced"),
    joinpath(@__DIR__, "..", "data", "LES_outputs", "southern_mooring_GLEN"),
    EM_CSV_FILE,
    "SM")

const PANELS = [(site = EM_SITE, year = 2009), (site = EM_SITE, year = 2010),
                (site = EM_SITE, year = 2011), (site = EM_SITE, year = 2014),
                (site = EM_SITE, year = 2015), (site = SM_SITE, year = 2010),
                (site = SM_SITE, year = 2011)]

const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"

const LES_STEM = "Lxy256_Lz212_Nxy128_Nz106"

function read_isothermal_date(csv_file, year)
    isfile(csv_file) || return nothing
    return open(csv_file) do f
        readline(f)   # skip header
        for line in eachline(f)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) >= 2 || continue
            wy = tryparse(Int, strip(parts[1]))
            wy == year || continue
            return DateTime(Date(strip(parts[2])))
        end
        nothing
    end
end

function fill_missing_linear(v)
    out = copy(v)
    n   = length(out)
    i   = 1
    while i <= n
        if isnan(out[i])
            j = i
            while j <= n && isnan(out[j])
                j += 1
            end
            lo = i > 1 ? out[i-1] : (j <= n ? out[j] : 0.0)
            hi = j <= n ? out[j] : (i > 1 ? out[i-1] : 0.0)
            for k in i:(j-1)
                frac    = (k - i + 1) / (j - i + 1)
                out[k]  = lo + frac * (hi - lo)
            end
            i = j
        else
            i += 1
        end
    end
    return out
end

# Load GLEN forcing: net heat flux (positive downward)
# Returns (t_days, Q_direct, Q_coare_wind) for both forcing sources
function load_glen_forcing_both(t_iso; ndays = 60)
    isnothing(t_iso) && return (Float64[], Float64[], Float64[])
    return NCDataset(GLEN_FILE) do ds
        times = DateTime.(ds["time"][:])
        t_end = t_iso + Day(ndays)
        idx   = findall(t -> t_iso <= t <= t_end, times)
        isempty(idx) && return (Float64[], Float64[], Float64[])

        getv(name, src_tag) = Float64[ismissing(x) ? NaN : Float64(x)
                                      for x in ds[name * src_tag][idx]]
        sw   = getv("downwelling_shortwave_flux", "")
        lw   = getv("downwelling_longwave_flux", "")

        t_days = Float64[Dates.value(times[i] - t_iso) / (1000 * 86400.0) for i in idx]

        # Net downward heat flux for both forcing sources
        shf_direct = getv("sensible_heat_flux", "")
        lhf_direct = getv("latent_heat_flux", "")
        Q_direct = @. shf_direct + lhf_direct - (1.0 - albedo_sw) * sw - ε_water * lw

        shf_coare = getv("sensible_heat_flux", "_coare_wind")
        lhf_coare = getv("latent_heat_flux", "_coare_wind")
        Q_coare = @. shf_coare + lhf_coare - (1.0 - albedo_sw) * sw - ε_water * lw

        return t_days, fill_missing_linear(Q_direct), fill_missing_linear(Q_coare)
    end
end

function cumtrapz(t, y)
    out = zeros(length(t))
    for i in 2:length(t)
        step = 0.5 * (y[i] + y[i-1]) * (t[i] - t[i-1])
        out[i] = out[i-1] + (isnan(step) ? 0.0 : step)
    end
    return out
end

# Depth-averaged temperature using trapezoidal integration
function depth_averaged_temp(depths, T_profile)
    # Filter out NaN values
    valid = .!isnan.(T_profile)
    if !any(valid)
        return NaN
    end
    d_valid = depths[valid]
    T_valid = T_profile[valid]
    # Sort by depth
    idx = sortperm(d_valid)
    d_valid = d_valid[idx]
    T_valid = T_valid[idx]
    # Trapezoidal integration over depth
    integral = 0.0
    for i in 2:length(d_valid)
        ddepth = d_valid[i] - d_valid[i-1]
        integral += 0.5 * (T_valid[i] + T_valid[i-1]) * ddepth
    end
    H = d_valid[end] - d_valid[1]
    return integral / H
end

# Linear interpolation helper
function interp_at(days, values, query_days)
    out = similar(query_days, Float64)
    for (k, qd) in enumerate(query_days)
        if qd <= days[1]
            out[k] = values[1]
        elseif qd >= days[end]
            out[k] = values[end]
        else
            i = searchsortedlast(days, qd)
            frac = (qd - days[i]) / (days[i+1] - days[i])
            out[k] = values[i] + frac * (values[i+1] - values[i])
        end
    end
    return out
end

# Apply depth correction for SMS09h.mat (Southern Mooring 2010)
function apply_known_depth_corrections!(dep, filename)
    filename == "SMS09h.mat" && (dep .-= 20.0)
    return dep
end

t2dt(t::Real) = DateTime(2000, 1, 1) + Millisecond(round(Int64, (t - 730486.0) * 86_400_000.0))

# Compute depth-averaged in-situ temperature from model field time series
# Tbar is conservative temperature; convert to in-situ before averaging
function model_depth_averaged_temp(fts)
    times_days = Float64.(fts.times) ./ 86400.0
    zC = znodes(fts.grid, Center())
    T_davg = Float64[]
    for i in eachindex(fts.times)
        # Get interior Tbar at this time, average over z (columns); then spatial mean
        Tbar_slice = interior(fts[i])  # dims: (x, y, z)
        # Average over x and y first (grid cell averaging at this location)
        Theta_z_profile = dropdims(mean(Tbar_slice, dims=(1, 2)), dims=(1, 2))  # z-profile of conservative temp

        # Convert conservative temperature to in-situ temperature
        T_z_profile = Float64[]
        for k in eachindex(zC)
            p_dbar = ρ₀ * g_grav * abs(zC[k]) / 1e4  # pressure in dbar
            T_insitu = gsw_t_from_ct(S_lake, Float64(Theta_z_profile[k]), p_dbar)
            push!(T_z_profile, T_insitu)
        end

        # Then integrate over z using trapezoidal rule
        integral = 0.0
        for k in 2:length(zC)
            dz = zC[k] - zC[k-1]
            integral += 0.5 * (T_z_profile[k] + T_z_profile[k-1]) * dz
        end
        H = abs(zC[end] - zC[1])
        push!(T_davg, integral / H)
    end
    return times_days, T_davg
end

# Load model files and compute depth-averaged temperature for both forcing sources
function load_model_series(site, year, t_iso)
    turb_days_coare, turb_T_coare, les_days_coare, les_T_coare = load_model_forcing(site, year, t_iso, "coare_wind")
    turb_days_direct, turb_T_direct, les_days_direct, les_T_direct = load_model_forcing(site, year, t_iso, "direct")
    return (turb_days_coare, turb_T_coare, turb_days_direct, turb_T_direct,
            les_days_coare, les_T_coare, les_days_direct, les_T_direct)
end

# Load model files for a specific forcing
function load_model_forcing(site, year, t_iso, forcing)
    src_tag = forcing == "direct" ? "" : "_coare_wind"
    turb_file = joinpath(site.turb_dir, "winter$(year)",
        "kepsilon$(src_tag)_UVstress_winter$(year).jld2")

    # Find LES file - SM has variable grid dimensions, so search for matching dir
    les_file = nothing
    if !isnothing(site.les_dir)
        if site.obs_prefix == "SM"
            # Search for matching directory pattern
            for dir in readdir(site.les_dir, join=true)
                if isdir(dir) && contains(dir, "LES_GLEN_southern_winter$(year)_$(forcing)_UVstress")
                    les_file = joinpath(dir, "hourly_averaged_timeseries.jld2")
                    break
                end
            end
        else
            # For Eastern Mooring, use fixed grid dimensions
            les_file = joinpath(site.les_dir, "LES_GLEN_winter$(year)_$(forcing)_UVstress_$(LES_STEM)",
                               "hourly_averaged_timeseries.jld2")
        end
    end

    turb_days, turb_T = if isfile(turb_file)
        fts = FieldTimeSeries(turb_file, "Tbar")
        model_depth_averaged_temp(fts)
    else
        (Float64[], Float64[])
    end

    les_days, les_T = if !isnothing(les_file) && isfile(les_file)
        fts = FieldTimeSeries(les_file, "Tbar")
        model_depth_averaged_temp(fts)
    else
        (Float64[], Float64[])
    end

    return turb_days, turb_T, les_days, les_T
end

# Observed sensor time series (returns days, T, Z, sensor_depth for given sensor type)
function observed_sensor_series(site, year, t_iso, sensor::Symbol; ndays = 60)
    isnothing(t_iso) && return (Float64[], Float64[], NaN, NaN)
    for f in sort(readdir(OBS_DIR; join = true))
        isdir(f) || continue
        for fn in sort(readdir(f))
            (startswith(fn, site.obs_prefix) && endswith(fn, ".mat")) || continue
            path = joinpath(f, fn)
            data = try matread(path) catch; continue end
            haskey(data, "dep") || continue
            t_raw = vec(Float64.(reshape(data["t"], :)))
            dts   = t2dt.(t_raw)
            (dts[1] <= t_iso <= dts[end] - Day(ndays)) || continue

            dep   = apply_known_depth_corrections!(vec(Float64.(reshape(data["dep"], :))), fn)
            T_raw = data["T"]
            Ndep, Nt = length(dep), length(t_raw)
            T = size(T_raw) == (Ndep, Nt) ? Float64.(T_raw) : Float64.(T_raw')
            sensor_idx = sensor === :bottom ? argmax(dep) : argmin(dep)

            idx  = findall(dt -> t_iso <= dt <= t_iso + Day(ndays), dts)
            days = [Dates.value(dts[i] - t_iso) / (1000 * 86400.0) for i in idx]
            Z    = haskey(data, "Z") ? Float64(data["Z"]) : NaN
            return days, T[sensor_idx, idx], Z, dep[sensor_idx]
        end
    end
    return Float64[], Float64[], NaN, NaN
end

# Load observed temperature profiles at all depths
function observed_temp_profiles(site, year, t_iso; ndays = 60)
    isnothing(t_iso) && return (Float64[], Float64[], NaN)
    for f in sort(readdir(OBS_DIR; join = true))
        isdir(f) || continue
        for fn in sort(readdir(f))
            (startswith(fn, site.obs_prefix) && endswith(fn, ".mat")) || continue
            path = joinpath(f, fn)
            data = try matread(path) catch; continue end
            haskey(data, "dep") || continue
            t_raw = vec(Float64.(reshape(data["t"], :)))
            dts   = t2dt.(t_raw)
            (dts[1] <= t_iso <= dts[end] - Day(ndays)) || continue

            dep   = apply_known_depth_corrections!(vec(Float64.(reshape(data["dep"], :))), fn)
            T_raw = data["T"]
            Ndep, Nt = length(dep), length(t_raw)
            T = size(T_raw) == (Ndep, Nt) ? Float64.(T_raw) : Float64.(T_raw')

            idx  = findall(dt -> t_iso <= dt <= t_iso + Day(ndays), dts)
            days = [Dates.value(dts[i] - t_iso) / (1000 * 86400.0) for i in idx]
            Z    = haskey(data, "Z") ? Float64(data["Z"]) : NaN

            # Compute depth-averaged temperature at each time using trapezoidal integration
            T_subset = T[:, idx]
            T_davg = [depth_averaged_temp(dep, T_subset[:, i]) for i in 1:size(T_subset, 2)]

            return days, T_davg, Z
        end
    end
    return Float64[], Float64[], NaN
end

#####
##### Load everything
#####
panel_data = Dict{Tuple{String,Int}, NamedTuple}()
for (site, year) in PANELS
    key   = (site.obs_prefix, year)
    t_iso = read_isothermal_date(site.csv_file, year)

    # Load GLEN forcing for both direct and coare_wind (precomputed without upward LW)
    t_glen_days, Q_precomp_direct, Q_precomp_coare = load_glen_forcing_both(t_iso)
    isempty(t_glen_days) && continue

    # Load observed temperature profiles
    obs_days, obs_T_davg, obs_Z = observed_temp_profiles(site, year, t_iso)
    isempty(obs_days) && continue

    # Load observed surface temperature to compute upward longwave
    obs_sfc_days, obs_sfc_T, _, _ = observed_sensor_series(site, year, t_iso, :surface)
    isempty(obs_sfc_days) && continue

    H = isnan(obs_Z) ? 212.0 : obs_Z

    # Add upward longwave contribution based on observed surface temperature
    # (same as in Tbot analytical model)
    T_obs_sfc_glen = interp_at(obs_sfc_days, obs_sfc_T, t_glen_days)
    LW_up_kin = @. ε_water * 5.67e-8 * (T_obs_sfc_glen + 273.15)^4 / (ρ₀ * cₚ)
    Q_direct_Wm2 = @. Q_precomp_direct + LW_up_kin * ρ₀ * cₚ
    Q_coare_Wm2 = @. Q_precomp_coare + LW_up_kin * ρ₀ * cₚ

    # Analytical depth-averaged temperature evolution from surface heat flux
    # T_davg(t) = T_davg(0) - ∫₀ᵗ Q(t') dt' / (H·ρ·cₚ)
    # Q_net is downward positive, so negate for cooling (upward flux)
    Q_out_kin_direct = @. -Q_direct_Wm2 / (ρ₀ * cₚ)
    Q_out_kin_coare  = @. -Q_coare_Wm2 / (ρ₀ * cₚ)

    t_sec_glen = t_glen_days .* 86400.0
    cumulative_direct = cumtrapz(t_sec_glen, Q_out_kin_direct) ./ H
    cumulative_coare  = cumtrapz(t_sec_glen, Q_out_kin_coare) ./ H

    # Interpolate to observation times and add initial observed depth-averaged temperature
    T_davg_0 = obs_T_davg[1]
    T_davg_direct = T_davg_0 .+ interp_at(t_glen_days, cumulative_direct, obs_days)
    T_davg_coare  = T_davg_0 .+ interp_at(t_glen_days, cumulative_coare, obs_days)

    # Load model outputs (both coare_wind and direct forcing)
    turb_days_coare, turb_T_coare, turb_days_direct, turb_T_direct,
        les_days_coare, les_T_coare, les_days_direct, les_T_direct = load_model_series(site, year, t_iso)

    panel_data[key] = (
        obs_days = obs_days,
        obs_T_davg = obs_T_davg,
        T_davg_direct = T_davg_direct,
        T_davg_coare = T_davg_coare,
        turb_days_coare = turb_days_coare,
        turb_T_coare = turb_T_coare,
        turb_days_direct = turb_days_direct,
        turb_T_direct = turb_T_direct,
        les_days_coare = les_days_coare,
        les_T_coare = les_T_coare,
        les_days_direct = les_days_direct,
        les_T_direct = les_T_direct,
        obs_Z = obs_Z,
        t_iso = t_iso
    )
end

isempty(panel_data) && error("No complete data found")

@info "Panels with data: $(length(panel_data))"
for (site, year) in PANELS
    d = get(panel_data, (site.obs_prefix, year), nothing)
    isnothing(d) && continue
    @info "  $(site.obs_prefix) Winter $year: Z = $(round(d.obs_Z, digits=1)) m"
end

const XTICKS = 0:10:60   # day-since-start ticks

function plot_depth_averaged_temp_evolution(filename)
    fig = Figure(size = (195 * 4 + 40, 210 * 2 + 70), fontsize = 11, figure_padding = (6, 10, 6, 4))

    positions = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]
    axes      = Axis[]
    for (i, (site, year)) in enumerate(PANELS)
        row, col = positions[i]
        d = get(panel_data, (site.obs_prefix, year), nothing)

        xlabel_str = isnothing(d) ? "Days since start" :
                     "Days since $(Dates.format(d.t_iso, "yyyy-mm-dd"))"

        ax = CairoMakie.Axis(fig[row, col];
                 title              = "$(site.name) Mooring, Winter $year",
                 xlabel             = xlabel_str,
                 ylabel             = col == 1 ? L"T_{\mathrm{davg}} \; (^\circ\mathrm{C})" : "",
                 yticklabelsvisible = col == 1,
                 yticksvisible      = col == 1,
                 xticks             = collect(XTICKS))
        push!(axes, ax)

        if isnothing(d)
            text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                  align = (:center, :center), fontsize = 11, color = :gray)
            continue
        end

        # Plot observed depth-averaged temperature
        lines!(ax, d.obs_days, d.obs_T_davg; color = :firebrick, linewidth = 1.5)

        # Plot model predictions (solid = coare_wind, dashed = direct)
        !isempty(d.les_days_coare) && lines!(ax, d.les_days_coare, d.les_T_coare;
                                             color = :black, linewidth = 2.0, linestyle = :solid)
        !isempty(d.les_days_direct) && lines!(ax, d.les_days_direct, d.les_T_direct;
                                              color = :black, linewidth = 2.0, linestyle = :dash)
        !isempty(d.turb_days_coare) && lines!(ax, d.turb_days_coare, d.turb_T_coare;
                                              color = :steelblue4, linewidth = 2.0, linestyle = :solid)
        !isempty(d.turb_days_direct) && lines!(ax, d.turb_days_direct, d.turb_T_direct;
                                               color = :steelblue4, linewidth = 2.0, linestyle = :dash)

        # Plot analytical predictions from surface heat flux (orange, solid = coare_wind, dashed = direct)
        lines!(ax, d.obs_days, d.T_davg_coare;
               color = :darkorange, linewidth = 2.0, linestyle = :solid)
        lines!(ax, d.obs_days, d.T_davg_direct;
               color = :darkorange, linewidth = 2.0, linestyle = :dash)

        xlims!(ax, (0.0, 60.0))
    end

    linkyaxes!(axes...)
    linkxaxes!(axes...)

    legend_elems  = [LineElement(color = :firebrick, linewidth = 1.5),
                     LineElement(color = :black, linewidth = 2.0, linestyle = :solid),
                     LineElement(color = :black, linewidth = 2.0, linestyle = :dash),
                     LineElement(color = :steelblue4, linewidth = 2.0, linestyle = :solid),
                     LineElement(color = :steelblue4, linewidth = 2.0, linestyle = :dash),
                     LineElement(color = :darkorange, linewidth = 2.0, linestyle = :solid),
                     LineElement(color = :darkorange, linewidth = 2.0, linestyle = :dash)]
    legend_labels = ["Observed", "LES (coare_wind)", "LES (direct)", "k-ε (coare_wind)", "k-ε (direct)", "Analytical (coare_wind)", "Analytical (direct)"]
    Legend(fig[2, 4], legend_elems, legend_labels;
           tellwidth = false, labelsize = 11, framevisible = false, padding = (0, 0, 0, 0))

    Label(fig[0, 1:4],
          "Lake Superior Moorings — depth-averaged temperature evolution (GLEN-forced)";
          fontsize = 11, font = :bold, justification = :center)

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)

    outfile = joinpath(FIGURE_DIR, filename)
    save(outfile, fig)
    @info "Saved → $outfile"
end

plot_depth_averaged_temp_evolution("lake_superior_mooring_depth_averaged_temp_evolution.pdf")
