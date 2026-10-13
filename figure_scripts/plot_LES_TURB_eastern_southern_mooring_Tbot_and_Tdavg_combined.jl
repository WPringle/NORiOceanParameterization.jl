#####
##### GLEN-forced Lake Superior Moorings — T_bot + T_davg combined timeseries
#####
# 4-row combined figure: rows 1-2 show T_bot (bottom temperature), rows 3-4 show
# T_davg (depth-averaged temperature evolution). T_bot uses COARE_wind forcing.
# T_davg shows both direct and coare_wind forcings (solid = coare_wind, dashed = direct).
# Subtitles "T_bot" and "T_davg" appear in blue above rows 1 and 3.
# Legends positioned at [2,4] (end of T_bot section) and [4,4] (end of T_davg section).
# Analytical model for T_davg uses purple color matching T_bot's analytical model.
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
using Printf

# ── Constants ──────────────────────────────────────────────────────────────

const S_lake    = 0.05
const ρ₀        = 999.8
const g_grav    = 9.80665
const κ_vonK    = 0.41
const cₚ        = 4182.0
const albedo_sw = 0.08
const ε_water   = 0.98
const T_MD      = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, 0.0), 0.0) + 273.15
const T_START_FALLBACK = 4.0

const FORCING_SOURCE = "coare_wind"
const SRC_TAG = "_coare_wind"

const TURB_DIR    = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
const TURB_DIR_SM = joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced")
const LES_DIR  = joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN")
const LES_STEM = "Lxy256_Lz212_Nxy128_Nz106"
const OBS_DIR  = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/Austin2023"
FIGURE_DIR     = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"
const CSV_FILE  = joinpath(@__DIR__, "..", "figure_data",
                          "lake_superior_eastern_mooring",
                          "lake_superior_eastern_mooring_winter_start_dates.csv")

const TBOT_XTICKS = collect(0:10:60)
const DAVG_XTICKS = collect(0:10:60)

# ── Mooring site definitions ───────────────────────────────────────────────

struct MooringSite
    name       :: String
    turb_dir   :: String
    les_dir    :: Union{String, Nothing}
    csv_file   :: String
    obs_prefix :: String
end

const EM_SITE = MooringSite(
    "Eastern",
    TURB_DIR,
    LES_DIR,
    CSV_FILE,
    "EM")

const SM_SITE = MooringSite(
    "Southern",
    TURB_DIR_SM,
    joinpath(@__DIR__, "..", "data", "LES_outputs", "southern_mooring_GLEN"),
    CSV_FILE,
    "SM")

const PANELS = [(site = EM_SITE, year = 2009), (site = EM_SITE, year = 2010),
                (site = EM_SITE, year = 2011), (site = EM_SITE, year = 2014),
                (site = EM_SITE, year = 2015), (site = SM_SITE, year = 2010),
                (site = SM_SITE, year = 2011)]

struct Variant
    label     :: String
    color     :: Symbol
    linestyle :: Symbol
end

const VARIANTS = [
    Variant("k-ε (UVstress)", :steelblue4, :solid),
    Variant("k-ε (unidirectional)", :darkorange, :dash),
    Variant("k-ε (no thermobaricity)", :seagreen, :dot)
]

# ── Helper functions ───────────────────────────────────────────────────────

function read_isothermal_date(csv_file, year)
    isfile(csv_file) || return nothing
    return open(csv_file) do f
        readline(f)
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

function cumtrapz(t, y)
    out = zeros(length(t))
    for i in 2:length(t)
        step = 0.5 * (y[i] + y[i-1]) * (t[i] - t[i-1])
        out[i] = out[i-1] + (isnan(step) ? 0.0 : step)
    end
    return out
end

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

function apply_known_depth_corrections!(dep, filename)
    filename == "SMS09h.mat" && (dep .-= 20.0)
    return dep
end

t2dt(t::Real) = DateTime(2000, 1, 1) + Millisecond(round(Int64, (t - 730486.0) * 86_400_000.0))

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

            T_subset = T[:, idx]
            T_davg = [depth_averaged_temp(dep, T_subset[:, i]) for i in 1:size(T_subset, 2)]

            return days, T_davg, Z
        end
    end
    return Float64[], Float64[], NaN
end

function depth_averaged_temp(depths, T_profile)
    valid = .!isnan.(T_profile)
    if !any(valid)
        return NaN
    end
    d_valid = depths[valid]
    T_valid = T_profile[valid]
    idx = sortperm(d_valid)
    d_valid = d_valid[idx]
    T_valid = T_valid[idx]
    integral = 0.0
    for i in 2:length(d_valid)
        ddepth = d_valid[i] - d_valid[i-1]
        integral += 0.5 * (T_valid[i] + T_valid[i-1]) * ddepth
    end
    H = d_valid[end] - d_valid[1]
    return integral / H
end

function bottom_Tinsitu_series(fts, target_depth::Real)
    zC       = znodes(fts.grid, Center())
    z_target = clamp(-abs(target_depth), zC[1], zC[end])
    k_lo     = clamp(searchsortedlast(zC, z_target), 1, length(zC) - 1)
    k_hi     = k_lo + 1
    frac     = (z_target - zC[k_lo]) / (zC[k_hi] - zC[k_lo])
    p_dbar   = ρ₀ * g_grav * abs(z_target) / 1e4
    Θ_bot    = [(1 - frac) * Float64(interior(fts[i])[1, 1, k_lo]) +
                      frac  * Float64(interior(fts[i])[1, 1, k_hi])
                for i in eachindex(fts.times)]
    T_bot    = [gsw_t_from_ct(S_lake, Θ, p_dbar) for Θ in Θ_bot]
    return Float64.(fts.times) ./ 86400.0, T_bot
end

function model_depth_averaged_temp(fts)
    times_days = Float64.(fts.times) ./ 86400.0
    zC = znodes(fts.grid, Center())
    T_davg = Float64[]
    for i in eachindex(fts.times)
        Tbar_slice = interior(fts[i])
        Theta_z_profile = dropdims(mean(Tbar_slice, dims=(1, 2)), dims=(1, 2))
        T_z_profile = Float64[]
        for k in eachindex(zC)
            p_dbar = ρ₀ * g_grav * abs(zC[k]) / 1e4
            T_insitu = gsw_t_from_ct(S_lake, Float64(Theta_z_profile[k]), p_dbar)
            push!(T_z_profile, T_insitu)
        end
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

function load_glen_forcing_raw(t_iso; ndays = 60)
    isnothing(t_iso) && return (Float64[], Float64[], Float64[], Float64[])
    return NCDataset(GLEN_FILE) do ds
        times = DateTime.(ds["time"][:])
        t_end = t_iso + Day(ndays)
        idx   = findall(t -> t_iso <= t <= t_end, times)
        isempty(idx) && return (Float64[], Float64[], Float64[], Float64[])
        getv(name) = Float64[ismissing(x) ? NaN : Float64(x) for x in ds[name][idx]]
        shf  = getv("sensible_heat_flux" * SRC_TAG)
        lhf  = getv("latent_heat_flux" * SRC_TAG)
        sw   = getv("downwelling_shortwave_flux")
        lw   = getv("downwelling_longwave_flux")
        mom  = getv("momentum_flux" * SRC_TAG)
        wdir = getv("wind_direction")

        t_days = Float64[Dates.value(times[i] - t_iso) / (1000 * 86400.0) for i in idx]
        Q_precomp_Wm2 = @. shf + lhf - (1.0 - albedo_sw) * sw - ε_water * lw

        ex = sind.(wdir)
        ey = cosd.(wdir)
        τ_mag_kin = mom ./ ρ₀
        Qu_vals = τ_mag_kin .* ex
        Qv_vals = τ_mag_kin .* ey

        return t_days, fill_missing_linear(Q_precomp_Wm2),
               fill_missing_linear(Qu_vals), fill_missing_linear(Qv_vals)
    end
end

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

        shf_direct = getv("sensible_heat_flux", "")
        lhf_direct = getv("latent_heat_flux", "")
        Q_direct = @. shf_direct + lhf_direct - (1.0 - albedo_sw) * sw - ε_water * lw

        shf_coare = getv("sensible_heat_flux", "_coare_wind")
        lhf_coare = getv("latent_heat_flux", "_coare_wind")
        Q_coare = @. shf_coare + lhf_coare - (1.0 - albedo_sw) * sw - ε_water * lw

        return t_days, fill_missing_linear(Q_direct), fill_missing_linear(Q_coare)
    end
end

mixed_to_bottom(L_MO, QT, H) = (L_MO < 0 && QT < 0) || (L_MO > H)

function simple_Tbot_model(t_iso, H, T_bot0, obs_sfc_days, obs_sfc_T; ndays = 60)
    t_glen_days, Q_precomp_Wm2, Qu, Qv = load_glen_forcing_raw(t_iso; ndays)
    (isempty(t_glen_days) || isempty(obs_sfc_days)) && return (Float64[], Float64[], Bool[])

    t_days = collect(0.0:(1 / 24):ndays)
    t_sec  = t_days .* 86400.0

    Q_precomp_Wm2_h = interp_at(t_glen_days, Q_precomp_Wm2, t_days)
    Qu_h            = interp_at(t_glen_days, Qu, t_days)
    Qv_h            = interp_at(t_glen_days, Qv, t_days)
    T_obs_sfc       = interp_at(obs_sfc_days, obs_sfc_T, t_days)

    CT_obs_sfc = [gsw_ct_from_t(S_lake, T, 0.0) for T in T_obs_sfc]
    α_obs_sfc  = [gsw_alpha(S_lake, CT, 0.0) for CT in CT_obs_sfc]
    LW_up_kin  = @. ε_water * 5.67e-8 * (T_obs_sfc + 273.15)^4 / (ρ₀ * cₚ)
    QT         = Q_precomp_Wm2_h ./ (ρ₀ * cₚ) .+ LW_up_kin

    Qmag = sqrt.(Qu_h .^ 2 .+ Qv_h .^ 2)
    L_MO = map(eachindex(t_days)) do i
        abs(QT[i]) < 1e-10 && return -Inf
        -Qmag[i]^1.5 / (κ_vonK * α_obs_sfc[i] * g_grav * QT[i])
    end

    heaviside_is_one = [mixed_to_bottom(L_MO[i], QT[i], H) for i in eachindex(L_MO)]
    integrand = QT .* heaviside_is_one
    T_bot     = T_bot0 .- cumtrapz(t_sec, integrand) ./ H
    return t_days, T_bot, .!heaviside_is_one
end

function load_model_forcing(site, year, t_iso, forcing)
    src_tag = forcing == "direct" ? "" : "_coare_wind"
    turb_file = joinpath(site.turb_dir, "winter$(year)",
        "kepsilon$(src_tag)_UVstress_winter$(year).jld2")

    les_file = nothing
    if !isnothing(site.les_dir)
        if site.obs_prefix == "SM"
            for dir in readdir(site.les_dir, join=true)
                if isdir(dir) && contains(dir, "LES_GLEN_southern_winter$(year)_$(forcing)_UVstress")
                    les_file = joinpath(dir, "hourly_averaged_timeseries.jld2")
                    break
                end
            end
        else
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

function load_model_series(site, year, t_iso)
    turb_days_coare, turb_T_coare, les_days_coare, les_T_coare = load_model_forcing(site, year, t_iso, "coare_wind")
    turb_days_direct, turb_T_direct, les_days_direct, les_T_direct = load_model_forcing(site, year, t_iso, "direct")
    return (turb_days_coare, turb_T_coare, turb_days_direct, turb_T_direct,
            les_days_coare, les_T_coare, les_days_direct, les_T_direct)
end

# ── Data loading (combined from both scripts) ──────────────────────────────

function load_combined_data()
    panel_data = Dict()

    for (site, year) in PANELS
        try
            t_iso = read_isothermal_date(site.csv_file, year)
            isnothing(t_iso) && continue

            panel_key = (site.obs_prefix, year)

            # ── Load T_bot data ──
            turb_f = joinpath(site.turb_dir, "winter$(year)",
                "kepsilon$(SRC_TAG)_UVstress_winter$(year).jld2")
            if !isfile(turb_f)
                @warn "Missing k-ε output: $turb_f"
                continue
            end
            turb_fts = FieldTimeSeries(turb_f, "Tbar")
            H = turb_fts.grid.Lz

            obs_days, obs_T, obs_Z, obs_bottom_dep = observed_sensor_series(site, year, t_iso, :bottom)
            obs_sfc_days, obs_sfc_T, _, _ = observed_sensor_series(site, year, t_iso, :surface)

            sample_depth = isnan(obs_bottom_dep) ? H : obs_bottom_dep
            turb_days, turb_T = bottom_Tinsitu_series(turb_fts, sample_depth)

            variant_series = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, length(VARIANTS))
            for (vi, v) in enumerate(VARIANTS)
                if vi == 1
                    variant_series[vi] = (turb_days, turb_T)
                    continue
                end
                uv_tag = vi == 2 ? "" : "_UVstress"
                eos_tag = vi == 3 ? "_EOSteos10cabbeling" : ""
                vf = joinpath(site.turb_dir, "winter$(year)",
                    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")
                variant_series[vi] = if isfile(vf)
                    bottom_Tinsitu_series(FieldTimeSeries(vf, "Tbar"), sample_depth)
                else
                    (Float64[], Float64[])
                end
            end

            les_f = nothing
            if !isnothing(site.les_dir)
                if site.obs_prefix == "SM"
                    for dir in readdir(site.les_dir, join=true)
                        if isdir(dir) && contains(dir, "LES_GLEN_southern_winter$(year)_$(FORCING_SOURCE)_UVstress")
                            les_f = joinpath(dir, "hourly_averaged_timeseries.jld2")
                            break
                        end
                    end
                else
                    les_f = joinpath(site.les_dir, "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_UVstress_$(LES_STEM)",
                                    "hourly_averaged_timeseries.jld2")
                end
            end

            les_days, les_T = if !isnothing(les_f) && isfile(les_f)
                bottom_Tinsitu_series(FieldTimeSeries(les_f, "Tbar"), sample_depth)
            else
                (Float64[], Float64[])
            end

            T_bot0 = isempty(obs_days) ? T_START_FALLBACK : interp_at(obs_days, obs_T, [0.0])[1]
            H_analytical = isnan(obs_Z) ? H : obs_Z
            model_days, model_T, heaviside_zero = simple_Tbot_model(t_iso, H_analytical, T_bot0, obs_sfc_days, obs_sfc_T)

            # ── Load T_davg data ──
            t_glen_days, Q_precomp_direct, Q_precomp_coare = load_glen_forcing_both(t_iso)
            if isempty(t_glen_days)
                @warn "GLEN forcing not found for $(site.obs_prefix) $year"
                continue
            end

            obs_days_davg, obs_T_davg, obs_Z_davg = observed_temp_profiles(site, year, t_iso)
            if isempty(obs_days_davg)
                @warn "Observed T_davg profiles not found for $(site.obs_prefix) $year"
                continue
            end

            obs_sfc_days_davg, obs_sfc_T_davg, _, _ = observed_sensor_series(site, year, t_iso, :surface)

            T_obs_sfc_glen = interp_at(obs_sfc_days_davg, obs_sfc_T_davg, t_glen_days)
            LW_up_kin = @. ε_water * 5.67e-8 * (T_obs_sfc_glen + 273.15)^4 / (ρ₀ * cₚ)
            Q_direct_Wm2 = @. Q_precomp_direct + LW_up_kin * ρ₀ * cₚ
            Q_coare_Wm2 = @. Q_precomp_coare + LW_up_kin * ρ₀ * cₚ

            Q_out_kin_direct = @. -Q_direct_Wm2 / (ρ₀ * cₚ)
            Q_out_kin_coare  = @. -Q_coare_Wm2 / (ρ₀ * cₚ)

            t_sec_glen = t_glen_days .* 86400.0
            cumulative_direct = cumtrapz(t_sec_glen, Q_out_kin_direct) ./ H
            cumulative_coare  = cumtrapz(t_sec_glen, Q_out_kin_coare) ./ H

            T_davg_0 = obs_T_davg[1]
            T_davg_direct = T_davg_0 .+ interp_at(t_glen_days, cumulative_direct, obs_days_davg)
            T_davg_coare  = T_davg_0 .+ interp_at(t_glen_days, cumulative_coare, obs_days_davg)

            turb_days_coare, turb_T_coare, turb_days_direct, turb_T_direct,
                les_days_coare, les_T_coare, les_days_direct, les_T_direct = load_model_series(site, year, t_iso)

            panel_data[panel_key] = (
                t_iso = t_iso,
                obs_days = obs_days,
                obs_T = obs_T,
                obs_T_davg = obs_T_davg,
                obs_days_davg = obs_days_davg,
                les_days = les_days,
                les_T = les_T,
                les_days_coare = les_days_coare,
                les_T_coare = les_T_coare,
                les_days_direct = les_days_direct,
                les_T_direct = les_T_direct,
                variant_series = variant_series,
                model_days = model_days,
                model_T = model_T,
                heaviside_zero = heaviside_zero,
                turb_days_coare = turb_days_coare,
                turb_T_coare = turb_T_coare,
                turb_days_direct = turb_days_direct,
                turb_T_direct = turb_T_direct,
                T_davg_coare = T_davg_coare,
                T_davg_direct = T_davg_direct
            )
        catch e
            @warn "Error loading data for $(site.obs_prefix) $year: $e"
        end
    end

    return panel_data
end

# ── Main plotting function ─────────────────────────────────────────────────

function plot_combined_Tbot_Tdavg()
    panel_data = load_combined_data()

    fig = Figure(size = (195 * 4 + 40, 210 * 4 + 180), fontsize = 11,
                 figure_padding = (6, 10, 6, 4))

    positions = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]

    # ── T_bot section (rows 1-2) ──────────────────────────────────────────

    axes_tbot = Axis[]
    for (i, (site, year)) in enumerate(PANELS)
        row, col = positions[i]
        d = get(panel_data, (site.obs_prefix, year), nothing)

        xlabel_str = isnothing(d) ? "Days since start" :
                     "Days since $(Dates.format(d.t_iso, "yyyy-mm-dd"))"

        ax = CairoMakie.Axis(fig[row, col];
                 title              = "$(site.name) Mooring, Winter $year",
                 xlabel             = xlabel_str,
                 ylabel             = col == 1 ? L"T_{\mathrm{bot}} \; (^\circ\mathrm{C})" : "",
                 yticklabelsvisible = col == 1,
                 yticksvisible      = col == 1,
                 xticks             = collect(TBOT_XTICKS))
        push!(axes_tbot, ax)

        if isnothing(d)
            text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                  align = (:center, :center), fontsize = 11, color = :gray)
            continue
        end

        !isempty(d.obs_days)   && lines!(ax, d.obs_days, d.obs_T; color = :firebrick, linewidth = 1.5)
        !isempty(d.les_days)   && lines!(ax, d.les_days, d.les_T; color = :black, linewidth = 2.0)
        for (vi, v) in enumerate(VARIANTS)
            vdays, vT = d.variant_series[vi]
            !isempty(vdays) && lines!(ax, vdays, vT; color = v.color, linewidth = 2.0, linestyle = v.linestyle)
        end
        if !isempty(d.model_days)
            lines!(ax, d.model_days, d.model_T;
                   color = :purple, linewidth = 2.0, linestyle = :solid)
            if length(d.heaviside_zero) == length(d.model_days)
                zero_indices = findall(d.heaviside_zero)
                if !isempty(zero_indices)
                    scatter!(ax, d.model_days[zero_indices], d.model_T[zero_indices];
                            color = :black, markersize = 5, marker = '·')
                end
            end
        end

        xlims!(ax, (0.0, 60.0))
    end

    linkyaxes!(axes_tbot...)
    linkxaxes!(axes_tbot...)

    legend_elems_tbot = Any[LineElement(color = :firebrick, linewidth = 1.5),
                            LineElement(color = :black, linewidth = 2.0)]
    legend_labels_tbot = Any["Observations", "LES"]
    for v in VARIANTS
        push!(legend_elems_tbot,  LineElement(color = v.color, linewidth = 2.0, linestyle = v.linestyle))
        push!(legend_labels_tbot, v.label)
    end
    push!(legend_elems_tbot,  LineElement(color = :purple, linewidth = 2.0, linestyle = :solid))
    push!(legend_labels_tbot, "Analytical model")
    Legend(fig[2, 4], legend_elems_tbot, legend_labels_tbot;
           tellwidth = false, labelsize = 11, framevisible = false, padding = (0, 0, 0, 0))

    Label(fig[0, 1:4], "T_bot"; color = :steelblue4, font = :bold, fontsize = 14,
          halign = :center, padding = (0, 0, 0, 2))

    # ── T_davg section (rows 3-4) ─────────────────────────────────────────

    axes_tdavg = Axis[]
    for (i, (site, year)) in enumerate(PANELS)
        row_offset = 2
        row, col = positions[i] .+ (row_offset, 0)
        d = get(panel_data, (site.obs_prefix, year), nothing)

        xlabel_str = isnothing(d) ? "Days since start" :
                     "Days since $(Dates.format(d.t_iso, "yyyy-mm-dd"))"

        ax = CairoMakie.Axis(fig[row, col];
                 title              = "$(site.name) Mooring, Winter $year",
                 xlabel             = xlabel_str,
                 ylabel             = col == 1 ? L"T_{\mathrm{davg}} \; (^\circ\mathrm{C})" : "",
                 yticklabelsvisible = col == 1,
                 yticksvisible      = col == 1,
                 xticks             = collect(DAVG_XTICKS))
        push!(axes_tdavg, ax)

        if isnothing(d)
            text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                  align = (:center, :center), fontsize = 11, color = :gray)
            continue
        end

        lines!(ax, d.obs_days_davg, d.obs_T_davg; color = :firebrick, linewidth = 1.5)

        !isempty(d.les_days_coare) && lines!(ax, d.les_days_coare, d.les_T_coare;
                                             color = :black, linewidth = 2.0, linestyle = :solid)
        !isempty(d.les_days_direct) && lines!(ax, d.les_days_direct, d.les_T_direct;
                                              color = :black, linewidth = 2.0, linestyle = :dash)
        !isempty(d.turb_days_coare) && lines!(ax, d.turb_days_coare, d.turb_T_coare;
                                              color = :steelblue4, linewidth = 2.0, linestyle = :solid)
        !isempty(d.turb_days_direct) && lines!(ax, d.turb_days_direct, d.turb_T_direct;
                                               color = :steelblue4, linewidth = 2.0, linestyle = :dash)

        !isempty(d.T_davg_coare) && lines!(ax, d.obs_days_davg, d.T_davg_coare;
                                          color = :purple, linewidth = 2.0, linestyle = :solid)
        !isempty(d.T_davg_direct) && lines!(ax, d.obs_days_davg, d.T_davg_direct;
                                           color = :purple, linewidth = 2.0, linestyle = :dash)

        xlims!(ax, (0.0, 60.0))
    end

    linkyaxes!(axes_tdavg...)
    linkxaxes!(axes_tdavg...)

    legend_elems_tdavg = [LineElement(color = :firebrick, linewidth = 1.5),
                          LineElement(color = :black, linewidth = 2.0, linestyle = :solid),
                          LineElement(color = :black, linewidth = 2.0, linestyle = :dash),
                          LineElement(color = :steelblue4, linewidth = 2.0, linestyle = :solid),
                          LineElement(color = :steelblue4, linewidth = 2.0, linestyle = :dash),
                          LineElement(color = :purple, linewidth = 2.0, linestyle = :solid),
                          LineElement(color = :purple, linewidth = 2.0, linestyle = :dash)]
    legend_labels_tdavg = ["Observed", "LES (coare_wind)", "LES (direct)",
                           "k-ε (coare_wind)", "k-ε (direct)",
                           "Analytical (coare_wind)", "Analytical (direct)"]
    Legend(fig[4, 4], legend_elems_tdavg, legend_labels_tdavg;
           tellwidth = false, labelsize = 11, framevisible = false, padding = (0, 0, 0, 0))

    Label(fig[2, 1:4], "T_davg"; color = :steelblue4, font = :bold, fontsize = 14,
          halign = :center, padding = (0, 0, 0, 2))

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)

    outfile = joinpath(FIGURE_DIR, "lake_superior_mooring_Tbot_and_Tdavg_combined.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
end

# ── Main execution ────────────────────────────────────────────────────────

plot_combined_Tbot_Tdavg()
