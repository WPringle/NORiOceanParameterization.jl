#####
##### GLEN-forced Eastern + Southern Mooring — LES vs TURB k-ε temperature comparison (30 & 60-day combined)
#####
# Combined figure showing both 30-day (rows 1-2) and 60-day (rows 3-4) snapshots.
# Subtitles "30-day" and "60-day" appear above rows 1 and 3 in blue.
# For detailed description of the comparison and annotations, see
# plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl

using Oceananigans
using CairoMakie
using JLD2
using NCDatasets
using Dates
using LaTeXStrings
using GibbsSeaWater
using Statistics
using Printf

function sci_latex(x; dec = 1)
    x == 0 && return "0"
    s     = @sprintf("%.*e", dec, x)
    m, e  = split(s, 'e')
    expo  = parse(Int, e)
    return "$(m)\\times10^{$(expo)}"
end

const S_lake    = 0.05
const ρ₀        = 999.8
const g_grav    = 9.80665
const cₚ        = 4182.0
const albedo_sw = 0.08
const ε_water   = 0.98
const T_MD      = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, 0.0), 0.0) + 273.15

m_rate(Qh_Wm2, QU, H) = abs(QU)^1.5 * T_MD / (g_grav * (Qh_Wm2 / (ρ₀ * cₚ)) * H)

const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"

const TURB_DIR    = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
const TURB_DIR_SM = joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced")
const LES_DIR  = joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN")
const LES_STEM = "Lxy256_Lz212_Nxy128_Nz106"
FIGURE_DIR     = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

turb_file(year, uv_tag, eos_tag) = joinpath(TURB_DIR, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")
turb_file_sm(year, uv_tag, eos_tag) = joinpath(TURB_DIR_SM, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")

function les_file_lookup(year)
    for dir in readdir(LES_DIR, join=true)
        if isdir(dir) && contains(dir, "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_UVstress")
            return joinpath(dir, "hourly_averaged_timeseries.jld2")
        end
    end
    return nothing
end

function les_file_sm_lookup(year)
    les_dir_sm = joinpath(@__DIR__, "..", "data", "LES_outputs", "southern_mooring_GLEN")
    for dir in readdir(les_dir_sm, join=true)
        if isdir(dir) && contains(dir, "LES_GLEN_southern_winter$(year)_$(FORCING_SOURCE)_UVstress")
            return joinpath(dir, "hourly_averaged_timeseries.jld2")
        end
    end
    return nothing
end

const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"
const CSV_FILE  = joinpath(@__DIR__, "..", "figure_data",
                          "lake_superior_eastern_mooring",
                          "lake_superior_eastern_mooring_winter_start_dates.csv")

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

function load_glen_forcing(year; ndays = 60, csv_file = CSV_FILE)
    t_iso = read_isothermal_date(csv_file, year)
    (isnothing(t_iso) || !isfile(GLEN_FILE)) && return nothing
    return NCDataset(GLEN_FILE) do ds
        times = DateTime.(ds["time"][:])
        t_end = t_iso + Day(ndays)
        idx   = findall(t -> t_iso <= t <= t_end, times)
        isempty(idx) && return nothing
        getv(name) = Float64[ismissing(x) ? NaN : Float64(x) for x in ds[name][idx]]
        shf = getv("sensible_heat_flux" * SRC_TAG)
        lhf = getv("latent_heat_flux"   * SRC_TAG)
        mom = getv("momentum_flux"      * SRC_TAG)
        sw  = getv("downwelling_shortwave_flux")
        lw  = getv("downwelling_longwave_flux")
        t_forc  = Float64[Dates.value(times[i] - t_iso) / 1000.0 for i in idx]
        Q_pre   = @. shf + lhf - (1.0 - albedo_sw) * sw - ε_water * lw
        tau_kin = @. -mom / ρ₀
        (t_forc = t_forc, Q_pre = Q_pre, tau_kin = tau_kin)
    end
end

function load_obs(obs_file, label)
    isfile(obs_file) || (@warn "$label obs not found ($obs_file) — run process_lake_superior_southern_eastern_moorings.jl first"; return nothing)
    jldopen(obs_file) do f
        (wys       = Int.(f["winter_years"]),
         dep_raw   = f["dep_raw_obs"],
         Tday1_raw = haskey(f, "Tday1_raw_obs") ? f["Tday1_raw_obs"] : nothing,
         T15_raw   = f["T15_raw_obs"],
         T1_raw    = f["T1_raw_obs"],
         T45_raw   = f["T45_raw_obs"],
         T2_raw    = f["T2_raw_obs"])
    end
end

const OBS_EM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_eastern_mooring", "observed_mld.jld2")
const OBS_SM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_southern_mooring", "observed_mld_EManchor.jld2")
obs_em = load_obs(OBS_EM_FILE, "EM")
obs_sm = load_obs(OBS_SM_FILE, "SM")

obs_profile_for_day(obs, day) = day == 1  ? obs.Tday1_raw :
                                 day == 15 ? obs.T15_raw   :
                                 day == 30 ? obs.T1_raw    :
                                 day == 45 ? obs.T45_raw   :
                                 day == 60 ? obs.T2_raw    : nothing

winter_years    = [2009, 2010, 2011, 2014, 2015]
winter_years_sm = [2010, 2011]
const DAYS = [30, 60]

const VARIANTS = [
    (uv_tag = "_UVstress", eos_tag = "",                        color = :steelblue4, label = "k-ε",                               linestyle = :solid),
    (uv_tag = "",          eos_tag = "",                        color = :darkorange, label = "k-ε (unidirectional wind stress)", linestyle = :dash),
    (uv_tag = "_UVstress", eos_tag = "_EOSteos10cabbeling",     color = :seagreen,   label = "k-ε (No thermobaricity)",          linestyle = :dot),
]
const LES_COLOR = :black
const LES_LABEL = "LES"

# Load all model data
variant_data = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years
        f = turb_file(year, v.uv_tag, v.eos_tag)
        isfile(f) || continue
        variant_data[vi][year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
    end
end

les_data = Dict{Int, Dict{String, Any}}()
for year in winter_years
    f = les_file_lookup(year)
    (isnothing(f) || !isfile(f)) && continue
    les_data[year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
end

variant_data_sm = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years_sm
        f = turb_file_sm(year, v.uv_tag, v.eos_tag)
        isfile(f) || continue
        variant_data_sm[vi][year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
    end
end

les_data_sm = Dict{Int, Dict{String, Any}}()
for year in winter_years_sm
    f = les_file_sm_lookup(year)
    (isnothing(f) || !isfile(f)) && continue
    les_data_sm[year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
end

reference_variant(year)    = _reference_variant(variant_data, year)
reference_variant_sm(year) = _reference_variant(variant_data_sm, year)
function _reference_variant(vdata, year)
    for vd in vdata
        haskey(vd, year) && return vd[year]
    end
    return nothing
end

ref_turb = let r = nothing
    for vd in variant_data, (_, dk) in vd
        r = dk["Tbar"]; break
    end
    r
end
zC_TURB     = znodes(ref_turb.grid, Center())
const Lz_TURB = ref_turb.grid.Lz

ref_les = isempty(les_data) ? nothing : first(values(les_data))["Tbar"]
zC_LES      = isnothing(ref_les) ? Float64[] : znodes(ref_les.grid, Center())
const Lz_LES  = isnothing(ref_les) ? Lz_TURB : ref_les.grid.Lz

ref_turb_sm = let r = nothing
    for vd in variant_data_sm, (_, dk) in vd
        r = dk["Tbar"]; break
    end
    r
end
zC_TURB_SM    = isnothing(ref_turb_sm) ? Float64[] : znodes(ref_turb_sm.grid, Center())
const Lz_TURB_SM = isnothing(ref_turb_sm) ? Lz_TURB : ref_turb_sm.grid.Lz

ref_les_sm = isempty(les_data_sm) ? nothing : first(values(les_data_sm))["Tbar"]
zC_LES_SM      = isnothing(ref_les_sm) ? Float64[] : znodes(ref_les_sm.grid, Center())
const Lz_LES_SM  = isnothing(ref_les_sm) ? Lz_TURB_SM : ref_les_sm.grid.Lz

function daily_avg_profile(fts, target_day, transform)
    td = fts.times ./ 86400.0
    if target_day == 0
        return transform(Float64.(interior(fts[1], 1, 1, :)))
    end
    idxs = findall(t -> target_day <= t < target_day + 1.0, td)
    if isempty(idxs)
        _, i = findmin(abs.(td .- target_day))
        return transform(Float64.(interior(fts[i], 1, 1, :)))
    end
    return mean([transform(Float64.(interior(fts[i], 1, 1, :))) for i in idxs])
end

function day_available(dk, d)
    d == 0 && return true
    td = dk["Tbar"].times ./ 86400.0
    isempty(td) && return false
    any(t -> d <= t < d + 1.0, td) && return true
    # Some runs end just short of the target day (e.g. 59.96 instead of 60.0);
    # fall back to the nearest snapshot, matching daily_avg_profile's own fallback.
    return minimum(abs.(td .- d)) < 1.0
end

function make_T_converter(zC)
    p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
    return Θ_prof -> [gsw_t_from_ct(S_lake, Θ_prof[k], p_dbar[k]) for k in eachindex(Θ_prof)]
end
const Θ_to_Tinsitu_TURB = make_T_converter(zC_TURB)
const Θ_to_Tinsitu_LES  = isempty(zC_LES) ? nothing : make_T_converter(zC_LES)
const Θ_to_Tinsitu_SM   = isempty(zC_TURB_SM) ? nothing : make_T_converter(zC_TURB_SM)
const Θ_to_Tinsitu_LES_SM = isempty(zC_LES_SM) ? nothing : make_T_converter(zC_LES_SM)

panel_profile_turb(dk, d)    = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu_TURB)
panel_profile_les(dk, d)     = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu_LES)
panel_profile_turb_sm(dk, d) = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu_SM)

function panel_profile_les_sm(dk, d, year)
    # Get the profile from this year's LES grid
    fts = dk["Tbar"]
    zC_local = znodes(fts.grid, Center())
    converter = make_T_converter(zC_local)
    profile = daily_avg_profile(fts, d, converter)
    return profile
end

function get_les_sm_z_for_year(year)
    f = les_file_sm_lookup(year)
    isnothing(f) || !isfile(f) && return Float64[]
    fts = FieldTimeSeries(f, "Tbar")
    return znodes(fts.grid, Center())
end

function interp_model_at(zC, Tmodel, ztarget)
    if ztarget <= zC[1]
        return Tmodel[1]
    elseif ztarget >= zC[end]
        return Tmodel[end]
    else
        i = searchsortedlast(zC, ztarget)
        α = (ztarget - zC[i]) / (zC[i+1] - zC[i])
        return Tmodel[i] + α * (Tmodel[i+1] - Tmodel[i])
    end
end

function model_obs_stats(zC, Tmodel, dep_obs, Tobs)
    valid = .!isnan.(Tobs)
    any(valid) || return (ME = NaN, RMSE = NaN)
    d   = dep_obs[valid]
    T   = Tobs[valid]
    idx = sortperm(d)
    d   = d[idx]
    T   = T[idx]
    errs = [interp_model_at(zC, Tmodel, -d[i]) - T[i] for i in eachindex(d)]
    n = length(d)
    n == 1 && return (ME = errs[1], RMSE = abs(errs[1]))
    w        = similar(d, Float64)
    w[1]     = (d[2] - d[1]) / 2
    w[end]   = (d[end] - d[end - 1]) / 2
    for i in 2:(n - 1)
        w[i] = (d[i + 1] - d[i - 1]) / 2
    end
    W = sum(w)
    return (ME = sum(w .* errs) / W, RMSE = sqrt(sum(w .* errs .^ 2) / W))
end

function compute_mean_forcing(years, csv_file, ref_variant_fn, H, les_data_dict=nothing)
    result = Dict{Int, Dict{Int, NamedTuple}}()
    for year in years
        forcing  = load_glen_forcing(year; ndays = maximum(DAYS), csv_file = csv_file)
        dk_ref   = ref_variant_fn(year)
        Tbar_fts = isnothing(dk_ref) ? nothing : dk_ref["Tbar"]

        if isnothing(forcing) || isnothing(Tbar_fts)
            @warn "No GLEN forcing for winter $year"
            result[year] = Dict(d => (Qh_Wm2 = NaN, QU_m2s2 = NaN, Msfc = NaN) for d in DAYS)
            continue
        end

        t_forc  = forcing.t_forc
        Q_pre   = forcing.Q_pre
        tau_kin = forcing.tau_kin
        T_sfc_times = Float64.(Tbar_fts.times)
        T_sfc_vals  = [Float64(interior(Tbar_fts[i])[1, 1, end]) for i in eachindex(Tbar_fts.times)]

        function lw_up_at(t_s)
            i = searchsortedlast(T_sfc_times, t_s)
            T_s = if i == 0
                T_sfc_vals[1]
            elseif i >= length(T_sfc_vals)
                T_sfc_vals[end]
            else
                α = (t_s - T_sfc_times[i]) / (T_sfc_times[i+1] - T_sfc_times[i])
                T_sfc_vals[i] + α * (T_sfc_vals[i+1] - T_sfc_vals[i])
            end
            return ε_water * 5.67e-8 * (T_s + 273.15)^4
        end

        nanmean(v) = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))
        function window_means(t_lo, t_hi)
            mask = (t_forc .> t_lo * 86400.0) .& (t_forc .<= t_hi * 86400.0)
            any(mask) || return (NaN, NaN)
            ts = t_forc[mask]
            Qh = nanmean(Q_pre[mask] .+ lw_up_at.(ts))
            QU = nanmean(tau_kin[mask])
            return (Qh, QU)
        end

        function cumulative_Msfc(τ)
            total, any_valid = 0.0, false
            for d in 1:τ
                Qh_d, QU_d = window_means(d - 1, d)
                (isnan(Qh_d) || isnan(QU_d)) && continue
                m_d = m_rate(Qh_d, QU_d, H)
                isnan(m_d) && continue
                total     += (d - 0.5) * m_d
                any_valid  = true
            end
            any_valid || return NaN
            return total / τ^2
        end

        result[year] = Dict(
            d => (Qh_Wm2 = Qh, QU_m2s2 = QU, Msfc = cumulative_Msfc(d))
            for (d, (Qh, QU)) in zip(DAYS, (window_means(0, d) for d in DAYS))
        )
    end
    return result
end

mean_forcing    = compute_mean_forcing(winter_years,    CSV_FILE,    reference_variant,    Lz_TURB)
mean_forcing_sm = compute_mean_forcing(winter_years_sm, CSV_FILE, reference_variant_sm, Lz_TURB_SM)

const EM_ROW1_YEARS = [2009, 2010, 2011, 2014]
const EM_ROW2_YEAR  = 2015
const NCOLS         = 4
const T_XLIMS       = (0.0, 4.5)
const T_XTICKS      = 0:1:4
const T_XMINORTICKS = IntervalsBetween(2)

const PANELS = vcat(
    [(mooring = :EM, year = y, row = i, col = j) for i in 1:2, j in 1:2, (idx, y) in enumerate([EM_ROW1_YEARS; EM_ROW2_YEAR]) if idx <= 4 + (i-1)],
    [(mooring = :SM, year = y, row = 1 + i, col = 2 + j) for i in 0:1, j in 0:1, (idx, y) in enumerate(winter_years_sm) if idx <= 2],
)

show_yticklabels(col) = col <= 2
show_ylabel(col)      = col == 1
show_xlabel(row)      = row != 2   # row 2 sits right above the "t = 60 days" subtitle

function plot_combined_profiles()
    fig = Figure(size = (195 * NCOLS + 40, 210 * 4 + 180),
                 fontsize = 11, figure_padding = (6, 10, 6, 4))

    # Rows 1,2 hold the 30-day panels, rows 4,5 the 60-day panels; rows 0 and 3
    # are dedicated (auto-sized) rows for the "t = 30/60 days" subtitles.
    day_starts = [1, 4]
    day_labels = ["t = 30 days", "t = 60 days"]

    all_axes = Axis[]

    for (day_section, day) in enumerate(DAYS)
        row_offset = day_starts[day_section]

        Label(fig[row_offset - 1, 1:2], day_labels[day_section];
              color = :steelblue4, font = :bold, fontsize = 16, halign = :center,
              padding = (0, 0, 0, 2))

        # EM panels
        em_axes = Axis[]
        for (idx, year) in enumerate(EM_ROW1_YEARS)
            col = idx
            row = row_offset
            ax = CairoMakie.Axis(fig[row, col];
                     title                = "EM Winter $year",
                     xlabel               = L"T \; (^\circ\mathrm{C})",
                     ylabel               = show_ylabel(col) ? L"z\;(\mathrm{m})" : "",
                     ylabelrotation       = π/2,
                     yticklabelsvisible   = show_yticklabels(col),
                     xticks               = T_XTICKS,
                     xminorticks          = T_XMINORTICKS,
                     xminorticksvisible   = true,
                     xgridvisible         = true,
                     ygridvisible         = true)
            push!(em_axes, ax)

            stats_entries = NamedTuple[]
            model_profiles = NamedTuple[]
            v1_profile = nothing

            dk_ref = reference_variant(year)
            if !isnothing(dk_ref)
                lines!(ax, panel_profile_turb(dk_ref, 0), zC_TURB;
                       color = :gray40, linewidth = 1.5, linestyle = :dash)
            end

            if haskey(les_data, year) && day_available(les_data[year], day)
                les_profile = panel_profile_les(les_data[year], day)
                lines!(ax, les_profile, zC_LES;
                       color = LES_COLOR, linewidth = 2.5)
                push!(model_profiles, (color = LES_COLOR, zC = zC_LES, profile = les_profile))
            end

            for (vi, v) in enumerate(VARIANTS)
                dk = get(variant_data[vi], year, nothing)
                (isnothing(dk) || !day_available(dk, day)) && continue
                profile = panel_profile_turb(dk, day)
                vi == 1 && (v1_profile = profile)
                lines!(ax, profile, zC_TURB;
                       color = v.color, linewidth = 2.0, linestyle = v.linestyle)
                push!(model_profiles, (color = v.color, zC = zC_TURB, profile = profile))
            end

            obs_T  = obs_profile_for_day(obs_em, day)
            wy_idx = nothing
            if !isnothing(obs_em) && !isnothing(obs_T)
                wy_idx = findfirst(==(year), obs_em.wys)
                if !isnothing(wy_idx)
                    scatter!(ax, obs_T[wy_idx], -obs_em.dep_raw[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                end
            end

            if !isnothing(wy_idx)
                dep_o, T_o = obs_em.dep_raw[wy_idx], obs_T[wy_idx]
                for mp in model_profiles
                    s = model_obs_stats(mp.zC, mp.profile, dep_o, T_o)
                    isnan(s.ME) || push!(stats_entries, (color = mp.color, ME = s.ME, RMSE = s.RMSE))
                end
            end

            ylims!(ax, (-max(Lz_TURB, Lz_LES) - 5, 5))
            xlims!(ax, T_XLIMS)

            # ME/RMSE annotation
            mid_T     = sum(T_XLIMS) / 2
            surface_T = !isnothing(v1_profile) ? v1_profile[end] :
                        !isempty(model_profiles) ? model_profiles[1].profile[end] : nothing
            anchor_x, halign = isnothing(surface_T) || surface_T > mid_T ? (0.03, :left) : (0.97, :right)
            if !isempty(stats_entries)
                text!(ax, anchor_x, 0.97;
                      text = "ME (RMSE)", space = :relative, align = (halign, :top), fontsize = 10)
                for (i, e) in enumerate(stats_entries)
                    text!(ax, anchor_x, 0.97 - i * 0.08;
                          text  = latexstring(@sprintf("%+.2f", e.ME), "\\ (", @sprintf("%.2f", e.RMSE),
                                              ")\\;^\\circ\\mathrm{C}"),
                          space = :relative, align = (halign, :top), color = e.color, fontsize = 10)
                end
            end

            # Mean forcing annotation
            mf = get(mean_forcing, year, nothing)
            if !isnothing(mf) && haskey(mf, day) && !isnan(mf[day].Qh_Wm2)
                f = mf[day]
                text!(ax, 0.03, 0.11;
                      text  = latexstring("\\bar{Q}_h = ", round(Int, f.Qh_Wm2), "\\;\\mathrm{W\\,m^{-2}}"),
                      space = :relative, align = (:left, :bottom), fontsize = 11)
                text!(ax, 0.03, 0.03;
                      text  = latexstring("\\bar{Q}_U = ", sci_latex(abs(f.QU_m2s2)), "\\;\\mathrm{m^2\\,s^{-2}}"),
                      space = :relative, align = (:left, :bottom), fontsize = 11)
            end
        end

        # EM row 2 (only winter 2015)
        col = 1
        row = row_offset + 1
        year = EM_ROW2_YEAR
        ax = CairoMakie.Axis(fig[row, col];
                 title                = "EM Winter $year",
                 xlabel               = show_xlabel(row) ? L"T \; (^\circ\mathrm{C})" : "",
                 xticklabelsvisible   = show_xlabel(row),
                 ylabel               = show_ylabel(col) ? L"z\;(\mathrm{m})" : "",
                 ylabelrotation       = π/2,
                 yticklabelsvisible   = show_yticklabels(col),
                 xticks               = T_XTICKS,
                 xminorticks          = T_XMINORTICKS,
                 xminorticksvisible   = true,
                 xgridvisible         = true,
                 ygridvisible         = true)
        push!(em_axes, ax)

        # (Same plotting logic as above, abbreviated for EM row 2)
        stats_entries = NamedTuple[]
        model_profiles = NamedTuple[]
        v1_profile = nothing

        dk_ref = reference_variant(year)
        if !isnothing(dk_ref)
            lines!(ax, panel_profile_turb(dk_ref, 0), zC_TURB;
                   color = :gray40, linewidth = 1.5, linestyle = :dash)
        end

        if haskey(les_data, year) && day_available(les_data[year], day)
            les_profile = panel_profile_les(les_data[year], day)
            lines!(ax, les_profile, zC_LES;
                   color = LES_COLOR, linewidth = 2.5)
            push!(model_profiles, (color = LES_COLOR, zC = zC_LES, profile = les_profile))
        end

        for (vi, v) in enumerate(VARIANTS)
            dk = get(variant_data[vi], year, nothing)
            (isnothing(dk) || !day_available(dk, day)) && continue
            profile = panel_profile_turb(dk, day)
            vi == 1 && (v1_profile = profile)
            lines!(ax, profile, zC_TURB;
                   color = v.color, linewidth = 2.0, linestyle = v.linestyle)
            push!(model_profiles, (color = v.color, zC = zC_TURB, profile = profile))
        end

        obs_T  = obs_profile_for_day(obs_em, day)
        wy_idx = nothing
        if !isnothing(obs_em) && !isnothing(obs_T)
            wy_idx = findfirst(==(year), obs_em.wys)
            if !isnothing(wy_idx)
                scatter!(ax, obs_T[wy_idx], -obs_em.dep_raw[wy_idx];
                         color = :firebrick, marker = :circle, markersize = 8)
            end
        end

        if !isnothing(wy_idx)
            dep_o, T_o = obs_em.dep_raw[wy_idx], obs_T[wy_idx]
            for mp in model_profiles
                s = model_obs_stats(mp.zC, mp.profile, dep_o, T_o)
                isnan(s.ME) || push!(stats_entries, (color = mp.color, ME = s.ME, RMSE = s.RMSE))
            end
        end

        ylims!(ax, (-max(Lz_TURB, Lz_LES) - 5, 5))
        xlims!(ax, T_XLIMS)

        # ME/RMSE annotation
        mid_T     = sum(T_XLIMS) / 2
        surface_T = !isnothing(v1_profile) ? v1_profile[end] :
                    !isempty(model_profiles) ? model_profiles[1].profile[end] : nothing
        anchor_x, halign = isnothing(surface_T) || surface_T > mid_T ? (0.03, :left) : (0.97, :right)
        if !isempty(stats_entries)
            text!(ax, anchor_x, 0.97;
                  text = "ME (RMSE)", space = :relative, align = (halign, :top), fontsize = 10)
            for (i, e) in enumerate(stats_entries)
                text!(ax, anchor_x, 0.97 - i * 0.08;
                      text  = latexstring(@sprintf("%+.2f", e.ME), "\\ (", @sprintf("%.2f", e.RMSE),
                                          ")\\;^\\circ\\mathrm{C}"),
                      space = :relative, align = (halign, :top), color = e.color, fontsize = 10)
            end
        end

        # Mean forcing annotation
        mf = get(mean_forcing, year, nothing)
        if !isnothing(mf) && haskey(mf, day) && !isnan(mf[day].Qh_Wm2)
            f = mf[day]
            text!(ax, 0.03, 0.11;
                  text  = latexstring("\\bar{Q}_h = ", round(Int, f.Qh_Wm2), "\\;\\mathrm{W\\,m^{-2}}"),
                  space = :relative, align = (:left, :bottom), fontsize = 11)
            text!(ax, 0.03, 0.03;
                  text  = latexstring("\\bar{Q}_U = ", sci_latex(abs(f.QU_m2s2)), "\\;\\mathrm{m^2\\,s^{-2}}"),
                  space = :relative, align = (:left, :bottom), fontsize = 11)
        end

        # SM panels
        sm_axes = Axis[]
        for (idx, year) in enumerate(winter_years_sm)
            col = idx + 1
            row = row_offset + 1
            ax = CairoMakie.Axis(fig[row, col];
                     title                = "SM Winter $year",
                     xlabel               = show_xlabel(row) ? L"T \; (^\circ\mathrm{C})" : "",
                     xticklabelsvisible   = show_xlabel(row),
                     ylabel               = show_ylabel(col) ? L"z\;(\mathrm{m})" : "",
                     ylabelrotation       = π/2,
                     yticklabelsvisible   = show_yticklabels(col),
                     xticks               = T_XTICKS,
                     xminorticks          = T_XMINORTICKS,
                     xminorticksvisible   = true,
                     xgridvisible         = true,
                     ygridvisible         = true)
            push!(sm_axes, ax)

            stats_entries = NamedTuple[]
            model_profiles = NamedTuple[]
            v1_profile = nothing

            dk_ref = reference_variant_sm(year)
            if !isnothing(dk_ref)
                lines!(ax, panel_profile_turb_sm(dk_ref, 0), zC_TURB_SM;
                       color = :gray40, linewidth = 1.5, linestyle = :dash)
            end

            if haskey(les_data_sm, year) && day_available(les_data_sm[year], day)
                les_profile = panel_profile_les_sm(les_data_sm[year], day, year)
                zC_les_year = get_les_sm_z_for_year(year)
                lines!(ax, les_profile, zC_les_year;
                       color = LES_COLOR, linewidth = 2.5)
                push!(model_profiles, (color = LES_COLOR, zC = zC_les_year, profile = les_profile))
            end

            for (vi, v) in enumerate(VARIANTS)
                dk = get(variant_data_sm[vi], year, nothing)
                (isnothing(dk) || !day_available(dk, day)) && continue
                profile = panel_profile_turb_sm(dk, day)
                vi == 1 && (v1_profile = profile)
                lines!(ax, profile, zC_TURB_SM;
                       color = v.color, linewidth = 2.0, linestyle = v.linestyle)
                push!(model_profiles, (color = v.color, zC = zC_TURB_SM, profile = profile))
            end

            obs_T  = obs_profile_for_day(obs_sm, day)
            wy_idx = nothing
            if !isnothing(obs_sm) && !isnothing(obs_T)
                wy_idx = findfirst(==(year), obs_sm.wys)
                if !isnothing(wy_idx)
                    scatter!(ax, obs_T[wy_idx], -obs_sm.dep_raw[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                end
            end

            if !isnothing(wy_idx)
                dep_o, T_o = obs_sm.dep_raw[wy_idx], obs_T[wy_idx]
                for mp in model_profiles
                    s = model_obs_stats(mp.zC, mp.profile, dep_o, T_o)
                    isnan(s.ME) || push!(stats_entries, (color = mp.color, ME = s.ME, RMSE = s.RMSE))
                end
            end

            ylims!(ax, (-max(Lz_TURB_SM, Lz_LES_SM) - 5, 5))
            xlims!(ax, T_XLIMS)

            # ME/RMSE annotation
            mid_T     = sum(T_XLIMS) / 2
            surface_T = !isnothing(v1_profile) ? v1_profile[end] :
                        !isempty(model_profiles) ? model_profiles[1].profile[end] : nothing
            anchor_x, halign = isnothing(surface_T) || surface_T > mid_T ? (0.03, :left) : (0.97, :right)
            if !isempty(stats_entries)
                text!(ax, anchor_x, 0.97;
                      text = "ME (RMSE)", space = :relative, align = (halign, :top), fontsize = 10)
                for (i, e) in enumerate(stats_entries)
                    text!(ax, anchor_x, 0.97 - i * 0.08;
                          text  = latexstring(@sprintf("%+.2f", e.ME), "\\ (", @sprintf("%.2f", e.RMSE),
                                              ")\\;^\\circ\\mathrm{C}"),
                          space = :relative, align = (halign, :top), color = e.color, fontsize = 10)
                end
            end

            # Mean forcing annotation
            mf = get(mean_forcing_sm, year, nothing)
            if !isnothing(mf) && haskey(mf, day) && !isnan(mf[day].Qh_Wm2)
                f = mf[day]
                text!(ax, 0.03, 0.11;
                      text  = latexstring("\\bar{Q}_h = ", round(Int, f.Qh_Wm2), "\\;\\mathrm{W\\,m^{-2}}"),
                      space = :relative, align = (:left, :bottom), fontsize = 11)
                text!(ax, 0.03, 0.03;
                      text  = latexstring("\\bar{Q}_U = ", sci_latex(abs(f.QU_m2s2)), "\\;\\mathrm{m^2\\,s^{-2}}"),
                      space = :relative, align = (:left, :bottom), fontsize = 11)
            end
        end

        # Add all axes from this section to the global list
        append!(all_axes, em_axes)
        append!(all_axes, sm_axes)
    end

    # Link all x-axes together for consistency
    length(all_axes) > 1 && linkxaxes!(all_axes...)

    # Legend
    legend_elems  = Any[LineElement(color = :gray40, linewidth = 1.5, linestyle = :dash),
                        MarkerElement(color = :firebrick, marker = :circle, markersize = 8),
                        LineElement(color = LES_COLOR, linewidth = 2.5)]
    legend_labels = Any["t = 0 (initial)", "Observations", LES_LABEL]
    for v in VARIANTS
        push!(legend_elems,  LineElement(color = v.color, linewidth = 2.0, linestyle = v.linestyle))
        push!(legend_labels, v.label)
    end
    Legend(fig[5, 4], legend_elems, legend_labels;
           labelsize = 11, framevisible = false, patchsize = (20, 12), tellwidth = false)

    # Set row and column spacing now that all rows/columns have content
    # (rows 0 and 3 hold the day-subtitle Labels and are left auto-sized)
    for row in (1, 2, 4, 5)
        rowsize!(fig.layout, row, Aspect(1, 1.0))
    end
    for col in 1:NCOLS
        colsize!(fig.layout, col, Relative(1 / NCOLS))
    end

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 6)

    # Trim the figure canvas down to the actual layout size (removes leftover
    # top/bottom margin left by the fixed Figure size vs. the Aspect-sized rows).
    resize_to_layout!(fig)

    outfile = joinpath(FIGURE_DIR, "LES_TURB_eastern_southern_mooring_comparison_T_profiles_combined_$(FORCING_SOURCE).pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
end

plot_combined_profiles()
