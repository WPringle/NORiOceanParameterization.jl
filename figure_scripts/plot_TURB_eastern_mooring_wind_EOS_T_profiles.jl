#####
##### GLEN-forced Eastern Mooring — TURB k-ε temperature profiles (wind stress × EOS)
#####
# Rows = winter year, columns = day 15 / 30 / 45 snapshots. Every panel overlays
# three k-ε variants against the observed temperature profile (T only):
#   • UVstress                          — directional wind stress; TEOS-10 EOS         (full model)  (blue,   solid)
#   • UVstress_EOScabbeling             — directional wind stress; Cabbeling EOS         (simplified)  (green,  dotted)
#   • UVstress_EOScabbelingthermobaric  — directional wind stress; Cabbeling-Thermobaric EOS (simplified) (purple, dash-dot)
# The shared initial (day 0) profile is drawn once per panel as a grey dashed line,
# and the observed profile (day 15 / 30 / 45) is overlaid in its matching column.
# Each panel is annotated with the mean forcing (Q̄_h, Q̄_U) over the window leading
# up to its snapshot day: 0–15 days, 15–30 days, or 30–45 days, plus the
# dimensionless surface mixing parameter M_sfc.
#
# Inputs  : data/TURB_outputs/eastern_mooring_GLEN_forced/winter<YEAR>/
#               kepsilon[_coare_wind][_UVstress][_EOScabbeling|_EOScabbelingthermobaric]_winter<YEAR>.jld2  (Tbar)
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv  (isothermal dates)
#               observed_mld.jld2                                     (obs)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
# Outputs : figures/TURB_eastern_mooring_wind_EOS_T_profiles_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_TURB_eastern_mooring_wind_EOS_T_profiles.jl               # direct (default)
#   julia plot_TURB_eastern_mooring_wind_EOS_T_profiles.jl coare_wind
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

# Surface mixing parameter M_sfc = |Q_U|^(3/2) T_MD / (g Q_T H) — dimensionless.
# Q_U is signed in this codebase (kinematic momentum flux), so |Q_U| is used to
# keep the fractional power real. Q_T here is the *kinematic* heat flux (K·m/s);
# Q_h is the reported heat flux in W/m², so Q_T = Q_h / (ρ₀ cₚ). Evaluated per
# calendar day (see window_Msfc_mean below), not on the window-mean Q_h/Q_U, since
# M_sfc is nonlinear and Q_h crosses zero often enough at half-hourly resolution
# to make an instantaneous per-sample average blow up.
Msfc(Qh_Wm2, QU, H) = abs(QU)^1.5 * T_MD / (g_grav * (Qh_Wm2 / (ρ₀ * cₚ)) * H)

# Forcing source: "direct" (measured EC fluxes) or "coare_wind".
const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"   # JLD2 flux suffix

const TURB_DIR = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
FIGURE_DIR     = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# k-ε output file for a given winter, UV-stress tag, and EOS tag.
turb_file(year, uv_tag, eos_tag) = joinpath(TURB_DIR, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")

# ── GLEN forcing file + isothermal (winter start) dates ───────────────────────
const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"
const CSV_FILE  = joinpath(@__DIR__, "..", "figure_data",
                          "lake_superior_eastern_mooring",
                          "lake_superior_eastern_mooring_winter_start_dates.csv")

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
function load_glen_forcing(year; ndays = 45)
    t_iso = read_isothermal_date(CSV_FILE, year)
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

# ── Observed eastern mooring profiles ─────────────────────────────────────────
const OBS_EM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_eastern_mooring", "observed_mld.jld2")
obs_em = if isfile(OBS_EM_FILE)
    jldopen(OBS_EM_FILE) do f
        (wys     = Int.(f["winter_years"]),
         dep_raw = f["dep_raw_obs"],
         T15_raw = f["T15_raw_obs"],
         T1_raw  = f["T1_raw_obs"],   # day-30 observed profile
         T45_raw = f["T45_raw_obs"])
    end
else
    @warn "EM obs not found ($OBS_EM_FILE) — run process_lake_superior_southern_eastern_moorings.jl first"
    nothing
end

# Observed raw profile (per year) for each column's snapshot day.
obs_profile_for_day(day) = day == 15 ? obs_em.T15_raw :
                            day == 30 ? obs_em.T1_raw  :
                            day == 45 ? obs_em.T45_raw : nothing

winter_years = [2009, 2010, 2011, 2014, 2015]

#####
##### Load data — the three k-ε variants (UVstress is the full/complete model)
#####
const VARIANTS = [
    (uv_tag = "_UVstress", eos_tag = "",                          color = :steelblue4, label = "Directional wind stress; TEOS-10 EOS", linestyle = :solid),
    (uv_tag = "_UVstress", eos_tag = "_EOScabbeling",              color = :seagreen,   label = "Cabbeling EOS",                        linestyle = :dot),
    (uv_tag = "_UVstress", eos_tag = "_EOScabbelingthermobaric",   color = :purple,     label = "Cabbeling-Thermobaric EOS",            linestyle = :dashdot),
]

variant_data = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years
        f = turb_file(year, v.uv_tag, v.eos_tag)
        isfile(f) || (@warn "Missing: $f"; continue)
        variant_data[vi][year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
    end
end
all(isempty, variant_data) && error("No k-ε TURB outputs found for FORCING_SOURCE=$(FORCING_SOURCE)")

# ── Grid geometry (from the first available run) ──────────────────────────────
ref = let r = nothing
    for vd in variant_data, (_, dk) in vd
        r = dk["Tbar"]; break
    end
    r
end
zC          = znodes(ref.grid, Center())
const Lz_EM = ref.grid.Lz

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

# Conservative Temperature → in-situ temperature (matches the LES / TURB figures)
p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
Θ_to_Tinsitu(Θ_prof) = [gsw_t_from_ct(S_lake, Θ_prof[k], p_dbar[k]) for k in eachindex(Θ_prof)]

panel_profile(dk, d) = daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu)

# First variant (in VARIANTS order) that has data for this winter — used for the
# shared initial-condition profile, which is common to all three variants.
function reference_variant(year)
    for vd in variant_data
        haskey(vd, year) && return vd[year]
    end
    return nothing
end

#####
##### Column configuration — target days, not variables
#####
const columns     = [(day = 15, label = "Day 15"), (day = 30, label = "Day 30"), (day = 45, label = "Day 45")]
const T_XLIMS     = (0.0, 5.0)
const WINDOW_EDGES = (0, 15, 30, 45)   # column i covers (WINDOW_EDGES[i], WINDOW_EDGES[i+1]] days

# ── Mean forcing per column window, from gap-filled GLEN NetCDF + T_sfc (LW_up) ──
mean_forcing = Dict{Int, Vector{NamedTuple}}()
for year in winter_years
    forcing  = load_glen_forcing(year)
    dk_ref   = reference_variant(year)
    Tbar_fts = isnothing(dk_ref) ? nothing : dk_ref["Tbar"]

    if isnothing(forcing) || isnothing(Tbar_fts)
        @warn "No GLEN forcing for winter $year (missing date or NetCDF)"
        mean_forcing[year] = [(Qh_Wm2 = NaN, QU_m2s2 = NaN, Msfc = NaN) for _ in 1:(length(WINDOW_EDGES) - 1)]
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

    # M_sfc is nonlinear in Q_h and Q_U (Q_h appears as 1/Q_h), and Q_h crosses zero
    # regularly at half-hourly resolution — computing M_sfc per half-hour sample and
    # averaging would let those near-zero-Q_h samples dominate the mean. Daily-mean
    # Q_h/Q_U first, then M_sfc per day, then mean of the daily M_sfc values instead.
    function window_Msfc_mean(day_lo, day_hi, H)
        vals = Float64[]
        for d in (day_lo + 1):day_hi
            Qh_d, QU_d = window_means(d - 1, d)
            (isnan(Qh_d) || isnan(QU_d)) && continue
            push!(vals, Msfc(Qh_d, QU_d, H))
        end
        isempty(vals) && return NaN
        return mean(vals)
    end

    window_data = NamedTuple[]
    for i in 1:(length(WINDOW_EDGES) - 1)
        lo, hi = WINDOW_EDGES[i], WINDOW_EDGES[i+1]
        Qh, QU = window_means(lo, hi)
        push!(window_data, (Qh_Wm2 = Qh, QU_m2s2 = QU, Msfc = window_Msfc_mean(lo, hi, Lz_EM)))
    end
    mean_forcing[year] = window_data
end

#####
##### Plotting: rows = winter year, columns = day snapshot
#####
function plot_TURB_T_profiles(filename)
    Nyears = length(winter_years)
    Ncols  = length(columns)

    with_theme(theme_latexfonts()) do
        fig = Figure(size = (230 * Ncols + 60, 175 * Nyears + 90),
                     fontsize = 10, figure_padding = (6, 34, 6, 4))

        for (col, c) in enumerate(columns)
            Label(fig[1, col], c.label; fontsize = 11, font = :bold,
                  tellwidth = false, halign = :center)
        end

        axes = Matrix{Any}(undef, Nyears, Ncols)

        for (row, year) in enumerate(winter_years)
            dk_ref  = reference_variant(year)
            for (col, c) in enumerate(columns)
                ax = CairoMakie.Axis(fig[row + 1, col];
                         xlabel             = row == Nyears ? L"T \; (^\circ\mathrm{C})" : "",
                         ylabel             = col == 1 ? L"z\;(\mathrm{m})" : "",
                         ylabelrotation     = π/2,
                         xticklabelsvisible = row == Nyears,
                         yticklabelsvisible = col == 1,
                         xgridvisible       = true,
                         ygridvisible       = true,
                         xticksize          = 4,
                         yticksize          = 4)
                axes[row, col] = ax

                if isnothing(dk_ref)
                    text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                          align = (:center, :center), fontsize = 9, color = :gray)
                    continue
                end

                # Shared initial profile — single grey dashed line.
                lines!(ax, panel_profile(dk_ref, 0), zC;
                       color = :gray40, linewidth = 1.5, linestyle = :dash)

                # Each of the three k-ε variants at this day.
                for (vi, v) in enumerate(VARIANTS)
                    haskey(variant_data[vi], year) || continue
                    dk = variant_data[vi][year]
                    day_available(dk, c.day) || continue
                    lines!(ax, panel_profile(dk, c.day), zC;
                           color = v.color, linewidth = 2.0, linestyle = v.linestyle)
                end

                ylims!(ax, (-Lz_EM - 5, 5))
                xlims!(ax, T_XLIMS)

                # M_sfc / Q̄_h / Q̄_U annotation for the window leading up to this column's day.
                mf = get(mean_forcing, year, nothing)
                if !isnothing(mf) && !isnan(mf[col].Qh_Wm2)
                    f = mf[col]
                    if !isnan(f.Msfc)
                        text!(ax, 0.03, 0.19;
                              text  = latexstring("M_{\\mathrm{sfc}} = ", sci_latex(f.Msfc)),
                              space = :relative, align = (:left, :bottom), fontsize = 8)
                    end
                    text!(ax, 0.03, 0.11;
                          text  = latexstring("\\bar{Q}_h = ", round(Int, f.Qh_Wm2), "\\;\\mathrm{W\\,m^{-2}}"),
                          space = :relative, align = (:left, :bottom), fontsize = 8)
                    text!(ax, 0.03, 0.03;
                          text  = latexstring("\\bar{Q}_U = ", sci_latex(f.QU_m2s2), "\\;\\mathrm{m^2\\,s^{-2}}"),
                          space = :relative, align = (:left, :bottom), fontsize = 8)
                end

                if col == 1
                    Label(fig[row + 1, 0], "Winter $year"; fontsize = 10, font = :bold,
                          rotation = π/2, tellheight = false)
                end
            end
        end

        all_axes = filter(x -> x isa Axis, vec(axes))
        length(all_axes) > 1 && linkyaxes!(all_axes...)
        length(all_axes) > 1 && linkxaxes!(all_axes...)

        # ── Observations — day 15 / 30 / 45, each in its matching column ─────
        has_obs = false
        if !isnothing(obs_em)
            for (col, c) in enumerate(columns)
                obs_T = obs_profile_for_day(c.day)
                isnothing(obs_T) && continue
                for (row, year) in enumerate(winter_years)
                    wy_idx = findfirst(==(year), obs_em.wys)
                    isnothing(wy_idx) && continue
                    ax = axes[row, col]
                    ax isa Axis || continue
                    scatter!(ax, obs_T[wy_idx], -obs_em.dep_raw[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                    has_obs = true
                end
            end
        end

        # ── Legend: initial profile, obs, variant colour key ─────────────────
        legend_elems  = Any[LineElement(color = :gray40, linewidth = 1.5, linestyle = :dash)]
        legend_labels = Any["t = 0 (initial)"]
        if has_obs
            push!(legend_elems,  MarkerElement(color = :firebrick, marker = :circle, markersize = 8))
            push!(legend_labels, "Observations")
        end
        for v in VARIANTS
            push!(legend_elems,  LineElement(color = v.color, linewidth = 2.0, linestyle = v.linestyle))
            push!(legend_labels, v.label)
        end
        Legend(fig[Nyears + 2, 1:Ncols], legend_elems, legend_labels;
               orientation = :horizontal, tellwidth = false, labelsize = 9,
               framevisible = false, padding = (0, 0, 0, 0), nbanks = 2)

        Label(fig[0, 1:Ncols],
              "Eastern Mooring — TURB k-ε temperature, GLEN-forced ($(FORCING_SOURCE))";
              fontsize = 11, font = :bold, justification = :center)

        colgap!(fig.layout, 10)
        rowgap!(fig.layout, 4)

        outfile = joinpath(FIGURE_DIR, filename)
        save(outfile, fig)
        @info "Saved → $outfile"
    end
end

#####
##### Generate figure
#####
plot_TURB_T_profiles("TURB_eastern_mooring_wind_EOS_T_profiles_$(FORCING_SOURCE).pdf")
