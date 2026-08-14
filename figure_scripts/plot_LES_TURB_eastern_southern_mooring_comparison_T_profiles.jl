#####
##### GLEN-forced Eastern + Southern Mooring — LES vs TURB k-ε temperature comparison
#####
# One figure per snapshot day since each run's own start date (15, 30, 45, 60).
# Layout: 2 x 4 grid — top row is the eastern mooring winters 2009/2010/2011/2014;
# bottom row is the legend, southern mooring winters 2010/2011, then the remaining
# eastern mooring winter 2015. Each panel overlays:
#   • LES                              (full LES, UV-decomposed wind stress)  (black)
#                                        [eastern mooring only — no southern LES runs]
#   • k-ε                              (UVstress + TEOS-10 EOS, the default)  (blue)
#   • k-ε (unidirectional wind stress) (all wind momentum on the u-stress)    (orange)
#   • k-ε (No thermobaricity)          (UVstress + full TEOS-10 with Z clamped to 0,  (green)
#                                        i.e. the exact cabbeling curve, no thermobaricity)
# against the observed profile, plus the shared initial (day 0) profile as a grey
# dashed line. Each panel is annotated with the mean forcing (Q̄_h, Q̄_U) from day 0
# up to the figure's snapshot day, plus the dimensionless surface mixing parameter
# M_sfc(τ) — a time-weighted average of the instantaneous forcing ratio, cumulative
# from the simulation start (τ = 0) through the snapshot day, that up-weights more
# recent forcing (see the Msfc section below).
#
# Inputs  : data/TURB_outputs/eastern_mooring_GLEN_forced/winter<YEAR>/
#               kepsilon[_coare_wind][_UVstress][_EOSteos10cabbeling]_winter<YEAR>.jld2  (Tbar)
#           data/TURB_outputs/southern_mooring_GLEN_forced/winter<YEAR>/
#               kepsilon[_coare_wind][_UVstress][_EOSteos10cabbeling]_winter<YEAR>.jld2  (Tbar)
#           data/LES_outputs/eastern_mooring_GLEN/
#               LES_GLEN_winter<YEAR>_<forcing>_UVstress_Lxy256_Lz212_Nxy128_Nz106/
#                   hourly_averaged_timeseries.jld2   (Tbar)
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv  (isothermal dates)
#               observed_mld.jld2                                     (obs)
#           figure_data/lake_superior_southern_mooring/
#               lake_superior_southern_mooring_winter_start_dates.csv (isothermal dates)
#               observed_mld.jld2                                     (obs)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
# Outputs : figures/LES_TURB_eastern_southern_mooring_comparison_T_profiles_day{15,30,45,60}_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_LES_TURB_eastern_mooring_comparison_T_profiles.jl               # direct (default)
#   julia plot_LES_TURB_eastern_mooring_comparison_T_profiles.jl coare_wind
#####


using Oceananigans
using CairoMakie
using JLD2
using NCDatasets
using Dates
using LaTeXStrings
using GibbsSeaWater
using Statistics
using Printf

# LaTeX scientific notation, e.g. sci_latex(4.56e-5) -> "4.6\times10^{-5}" (for
# embedding in a latexstring(...) so it renders with a proper minus/exponent).
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
const cₚ        = 4182.0   # J/(kg·K) — freshwater specific heat (matches experiment)
const albedo_sw = 0.08     # broadband shortwave albedo (matches experiment)
const ε_water   = 0.98     # longwave emissivity of water (matches experiment)

# In-situ temperature of maximum density at the surface (p = 0 dbar), in Kelvin, for M_sfc below.
const T_MD = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, 0.0), 0.0) + 273.15   # K

# Instantaneous surface mixing "rate" m(t) = |Q_U|^(3/2) T_MD / (g Q_T H) — dimensionless.
# Q_U is signed in this codebase (kinematic momentum flux), so |Q_U| is used to
# keep the fractional power real. Q_T here is the *kinematic* heat flux (K·m/s);
# Q_h is the reported heat flux in W/m², so Q_T = Q_h / (ρ₀ cₚ).
m_rate(Qh_Wm2, QU, H) = abs(QU)^1.5 * T_MD / (g_grav * (Qh_Wm2 / (ρ₀ * cₚ)) * H)

# Surface mixing parameter, evaluated cumulatively from the start of the simulation
# (t = 0) up to time τ via a time-weighted average that up-weights more recent
# forcing — since |α| grows as the surface cools further below T_MD, forcing late
# in the record matters more than forcing right at the start:
#   M_sfc(τ) = (1/τ²) ∫₀^τ ∫_t^τ m(t') dt' dt = (1/τ²) ∫₀^τ t' m(t') dt'   (swap order)
# Discretized on calendar days (see cumulative_Msfc below), for the same reason
# Q_h/Q_U are daily-averaged before applying m(): Q_h crosses zero often enough at
# half-hourly resolution that evaluating m(t) on individual samples blows up.

# Forcing source: "direct" (measured EC fluxes) or "coare_wind".
const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"   # JLD2 / NetCDF flux suffix

const TURB_DIR    = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
const TURB_DIR_SM = joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced")
const LES_DIR  = joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN")
const LES_STEM = "Lxy256_Lz212_Nxy128_Nz106"
FIGURE_DIR     = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# k-ε output file for a given winter, UV-stress tag, and EOS tag (eastern or southern mooring).
turb_file(year, uv_tag, eos_tag) = joinpath(TURB_DIR, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")
turb_file_sm(year, uv_tag, eos_tag) = joinpath(TURB_DIR_SM, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")

# LES output file — the only LES variant available is UV-decomposed wind stress
# (eastern mooring only — no southern mooring LES runs exist).
les_file(year) = joinpath(LES_DIR,
    "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_UVstress_$(LES_STEM)", "hourly_averaged_timeseries.jld2")

# ── GLEN forcing file + isothermal (winter start) dates ───────────────────────
const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"
const CSV_FILE  = joinpath(@__DIR__, "..", "figure_data",
                          "lake_superior_eastern_mooring",
                          "lake_superior_eastern_mooring_winter_start_dates.csv")
const CSV_FILE_SM = joinpath(@__DIR__, "..", "figure_data",
                          "lake_superior_southern_mooring",
                          "lake_superior_southern_mooring_winter_start_dates.csv")

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

# Forcing time series (W/m², m²/s²) over [t_iso, t_iso + ndays], read directly from
# the gap-filled GLEN NetCDF and reduced exactly as the experiment does.
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

# ── Observed mooring profiles (eastern + southern share the same jld2 schema) ──
function load_obs(obs_file, label)
    isfile(obs_file) || (@warn "$label obs not found ($obs_file) — run process_lake_superior_southern_eastern_moorings.jl first"; return nothing)
    jldopen(obs_file) do f
        (wys     = Int.(f["winter_years"]),
         dep_raw = f["dep_raw_obs"],
         T15_raw = f["T15_raw_obs"],
         T1_raw  = f["T1_raw_obs"],   # day-30 observed profile
         T45_raw = f["T45_raw_obs"],
         T2_raw  = f["T2_raw_obs"])   # day-60 observed profile
    end
end

const OBS_EM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_eastern_mooring", "observed_mld.jld2")
const OBS_SM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_southern_mooring", "observed_mld.jld2")
obs_em = load_obs(OBS_EM_FILE, "EM")
obs_sm = load_obs(OBS_SM_FILE, "SM")

# Observed raw profile (per year) for a given snapshot day.
obs_profile_for_day(obs, day) = day == 15 ? obs.T15_raw :
                                 day == 30 ? obs.T1_raw  :
                                 day == 45 ? obs.T45_raw :
                                 day == 60 ? obs.T2_raw  : nothing

winter_years    = [2009, 2010, 2011, 2014, 2015]   # eastern mooring
winter_years_sm = [2010, 2011]                     # southern mooring

#####
##### Load data — the three TURB k-ε variants + LES
#####
const VARIANTS = [
    (uv_tag = "_UVstress", eos_tag = "",              color = :steelblue4, label = "k-ε",                               linestyle = :solid),
    (uv_tag = "",          eos_tag = "",              color = :darkorange, label = "k-ε (unidirectional wind stress)", linestyle = :dash),
    (uv_tag = "_UVstress", eos_tag = "_EOSteos10cabbeling",  color = :seagreen,   label = "k-ε (No thermobaricity)",    linestyle = :dot),
]
const LES_COLOR = :black
const LES_LABEL = "LES"

variant_data = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years
        f = turb_file(year, v.uv_tag, v.eos_tag)
        isfile(f) || (@warn "Missing: $f"; continue)
        variant_data[vi][year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
    end
end
all(isempty, variant_data) && error("No k-ε TURB outputs found for FORCING_SOURCE=$(FORCING_SOURCE)")

les_data = Dict{Int, Dict{String, Any}}()
for year in winter_years
    f = les_file(year)
    isfile(f) || (@warn "Missing LES output: $f"; continue)
    les_data[year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
end
isempty(les_data) && @warn "No LES outputs found for FORCING_SOURCE=$(FORCING_SOURCE)"

# Southern mooring — same k-ε variants, no LES counterpart.
variant_data_sm = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years_sm
        f = turb_file_sm(year, v.uv_tag, v.eos_tag)
        isfile(f) || (@warn "Missing: $f"; continue)
        variant_data_sm[vi][year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
    end
end
all(isempty, variant_data_sm) && @warn "No southern mooring k-ε TURB outputs found for FORCING_SOURCE=$(FORCING_SOURCE)"

# First variant (in VARIANTS order) that has data for this winter — used for the
# shared initial-condition profile, which is common to all four series.
reference_variant(year)    = _reference_variant(variant_data, year)
reference_variant_sm(year) = _reference_variant(variant_data_sm, year)
function _reference_variant(vdata, year)
    for vd in vdata
        haskey(vd, year) && return vd[year]
    end
    return nothing
end

# ── Grid geometry (TURB and LES use different z-grids) ─────────────────────────
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

# ── Daily-averaging helper (times taken from each field's own record) ─────────
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
    return any(t -> d <= t < d + 1.0, td)
end

# Conservative Temperature → in-situ temperature, specific to each grid's pressure.
function make_T_converter(zC)
    p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
    return Θ_prof -> [gsw_t_from_ct(S_lake, Θ_prof[k], p_dbar[k]) for k in eachindex(Θ_prof)]
end
const Θ_to_Tinsitu_TURB = make_T_converter(zC_TURB)
const Θ_to_Tinsitu_LES  = isempty(zC_LES) ? nothing : make_T_converter(zC_LES)
const Θ_to_Tinsitu_SM   = isempty(zC_TURB_SM) ? nothing : make_T_converter(zC_TURB_SM)

panel_profile_turb(dk, d)    = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu_TURB)
panel_profile_les(dk, d)     = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu_LES)
panel_profile_turb_sm(dk, d) = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu_SM)

#####
##### Mean forcing per year, cumulative from day 0 to each snapshot day
#####
const DAYS = [15, 30, 45, 60]

# M_sfc(τ) = (1/τ²) Σᵢ tᵢ m(Qh_i, QU_i, H) · Δt, tᵢ = day midpoint, Δt = 1 day —
# the discretized (1/τ²) ∫₀^τ t' m(t') dt', with m() evaluated on daily-mean
# Qh/QU (not half-hourly samples) to avoid the Qh-crosses-zero blow-up.
function compute_mean_forcing(years, csv_file, ref_variant_fn, H)
    result = Dict{Int, Dict{Int, NamedTuple}}()
    for year in years
        forcing  = load_glen_forcing(year; ndays = maximum(DAYS), csv_file = csv_file)
        dk_ref   = ref_variant_fn(year)
        Tbar_fts = isnothing(dk_ref) ? nothing : dk_ref["Tbar"]

        if isnothing(forcing) || isnothing(Tbar_fts)
            @warn "No GLEN forcing for winter $year (missing date or NetCDF)"
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
mean_forcing_sm = compute_mean_forcing(winter_years_sm, CSV_FILE_SM, reference_variant_sm, Lz_TURB_SM)

#####
##### Plotting: 2 x 4 grid.
##### Row 1: eastern mooring winters 2009/2010/2011/2014.
##### Row 2: legend, southern mooring winters 2010/2011, eastern mooring winter 2015.
#####
const EM_ROW1_YEARS = [2009, 2010, 2011, 2014]
const EM_ROW2_YEAR  = 2015
const NCOLS         = 4
const T_XLIMS       = (0.0, 4.2)   # narrowed from 5.0 °C — little/no data above ~4.2 °C

# (mooring, year, row, col) for every data panel; the legend fills (2, 1).
const PANELS = vcat(
    [(mooring = :EM, year = y,           row = 1, col = i)     for (i, y) in enumerate(EM_ROW1_YEARS)],
    [(mooring = :SM, year = y,           row = 2, col = i + 1) for (i, y) in enumerate(winter_years_sm)],
    [(mooring = :EM, year = EM_ROW2_YEAR, row = 2, col = NCOLS)],
)
const LEGEND_ROW, LEGEND_COL = 2, 1

function plot_LES_TURB_profiles(filename, day)
    fig = Figure(size = (195 * NCOLS + 40, 210 * 2 + 70),
                 fontsize = 11, figure_padding = (6, 10, 6, 4))

    em_axes = Axis[]
    sm_axes = Axis[]
    for p in PANELS
        row, col, year = p.row, p.col, p.year
        is_leftmost_data_col = (row == 1) ? (col == 1) : (col == 2)
        ax = CairoMakie.Axis(fig[row, col];
                 title              = p.mooring == :EM ? "EM Winter $year" : "SM Winter $year",
                 xlabel             = L"T \; (^\circ\mathrm{C})",
                 ylabel             = is_leftmost_data_col ? L"z\;(\mathrm{m})" : "",
                 ylabelrotation     = π/2,
                 yticklabelsvisible = is_leftmost_data_col,
                 xgridvisible       = true,
                 ygridvisible       = true,
                 xticksize          = 4,
                 yticksize          = 4)
        push!(p.mooring == :EM ? em_axes : sm_axes, ax)

        if p.mooring == :EM
            dk_ref = reference_variant(year)
            if !isnothing(dk_ref)
                lines!(ax, panel_profile_turb(dk_ref, 0), zC_TURB;
                       color = :gray40, linewidth = 1.5, linestyle = :dash)
            end

            if haskey(les_data, year) && day_available(les_data[year], day)
                lines!(ax, panel_profile_les(les_data[year], day), zC_LES;
                       color = LES_COLOR, linewidth = 2.5)
            end

            for (vi, v) in enumerate(VARIANTS)
                dk = get(variant_data[vi], year, nothing)
                (isnothing(dk) || !day_available(dk, day)) && continue
                lines!(ax, panel_profile_turb(dk, day), zC_TURB;
                       color = v.color, linewidth = 2.0, linestyle = v.linestyle)
            end

            obs_T = obs_profile_for_day(obs_em, day)
            if !isnothing(obs_em) && !isnothing(obs_T)
                wy_idx = findfirst(==(year), obs_em.wys)
                if !isnothing(wy_idx)
                    scatter!(ax, obs_T[wy_idx], -obs_em.dep_raw[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                end
            end

            ylims!(ax, (-max(Lz_TURB, Lz_LES) - 5, 5))
        else   # :SM
            dk_ref = reference_variant_sm(year)
            if !isnothing(dk_ref)
                lines!(ax, panel_profile_turb_sm(dk_ref, 0), zC_TURB_SM;
                       color = :gray40, linewidth = 1.5, linestyle = :dash)
            end

            for (vi, v) in enumerate(VARIANTS)
                dk = get(variant_data_sm[vi], year, nothing)
                (isnothing(dk) || !day_available(dk, day)) && continue
                lines!(ax, panel_profile_turb_sm(dk, day), zC_TURB_SM;
                       color = v.color, linewidth = 2.0, linestyle = v.linestyle)
            end

            obs_T = obs_profile_for_day(obs_sm, day)
            if !isnothing(obs_sm) && !isnothing(obs_T)
                wy_idx = findfirst(==(year), obs_sm.wys)
                if !isnothing(wy_idx)
                    scatter!(ax, obs_T[wy_idx], -obs_sm.dep_raw[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                end
            end

            ylims!(ax, (-Lz_TURB_SM - 5, 5))
        end
        xlims!(ax, T_XLIMS)

        # Q̄_h / Q̄_U annotation — cumulative mean forcing from day 0 to this day.
        mf_dict = p.mooring == :EM ? mean_forcing : mean_forcing_sm
        mf = get(mf_dict, year, nothing)
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

    length(em_axes) > 1 && linkyaxes!(em_axes...)
    length(sm_axes) > 1 && linkyaxes!(sm_axes...)
    all_axes = vcat(em_axes, sm_axes)
    length(all_axes) > 1 && linkxaxes!(all_axes...)

    # ── Legend in the bottom-left grid position ───────────────────────────────
    legend_elems  = Any[LineElement(color = :gray40, linewidth = 1.5, linestyle = :dash),
                        MarkerElement(color = :firebrick, marker = :circle, markersize = 8),
                        LineElement(color = LES_COLOR, linewidth = 2.5)]
    legend_labels = Any["t = 0 (initial)", "Observations", LES_LABEL * " (EM only)"]
    for v in VARIANTS
        push!(legend_elems,  LineElement(color = v.color, linewidth = 2.0, linestyle = v.linestyle))
        push!(legend_labels, v.label)
    end
    Legend(fig[LEGEND_ROW, LEGEND_COL], legend_elems, legend_labels;
           labelsize = 11, framevisible = false, patchsize = (20, 12), tellwidth = false)

    Label(fig[0, 1:NCOLS],
          "Eastern & Southern Mooring — LES vs k-ε temperature, day $(day) (GLEN-forced, $(FORCING_SOURCE))";
          fontsize = 11, font = :bold, justification = :center)

    # Without this, the legend (row 2, col 1) reports a narrower natural width than
    # the axes and visibly shrinks that whole column relative to the others.
    for col in 1:NCOLS
        colsize!(fig.layout, col, Relative(1/NCOLS))
    end

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 6)

    outfile = joinpath(FIGURE_DIR, filename)
    save(outfile, fig)
    @info "Saved → $outfile"
end

#####
##### Generate figures — one per snapshot day
#####
for day in DAYS
    plot_LES_TURB_profiles(
        "LES_TURB_eastern_southern_mooring_comparison_T_profiles_day$(day)_$(FORCING_SOURCE).pdf", day)
end
