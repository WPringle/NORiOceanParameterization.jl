#####
##### GLEN-forced Eastern Mooring — LES UVstress implementation comparison (T only)
#####
# Rows = winter year, columns = day 30 / 45 / 60 snapshots. Every panel overlays
# three LES implementations, all with the same UV-decomposed (directional) wind
# stress, against the observed temperature profile (T only):
#   • UVstress                       — default LES (WENO(order=9) advection,      (full model)  (blue,   solid)
#                                        no explicit subgrid closure)
#   • WENO5_UVstress                 — lower-order (5th) WENO advection            (simplified)  (orange, dashed)
#   • SmagorinskyLilly_UVstress      — explicit Smagorinsky–Lilly subgrid closure  (simplified)  (green,  dotted)
# The full 3-way comparison is only available for FORCING_SOURCE=coare_wind for
# most winters (2009, 2010, 2011, 2014) — direct mostly only has the default
# implementation, and 2015/coare_wind is missing SmagorinskyLilly; missing runs
# are skipped with a warning rather than filling the panel.
# The shared initial (day 0) profile is drawn once per panel as a grey dashed line,
# and the observed profile (day 30 / 45 / 60) is overlaid in its matching column.
# Each panel is annotated with the mean forcing (Q̄_h, Q̄_U) over the window leading
# up to its snapshot day: 0–30 days, 30–45 days, or 45–60 days.
#
# Inputs  : data/LES_outputs/eastern_mooring_GLEN/
#               LES_GLEN_winter<YEAR>_<forcing>[_WENO5|_SmagorinskyLilly]_UVstress_
#                   Lxy256_Lz212_Nxy128_Nz106/hourly_averaged_timeseries.jld2  (Tbar)
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv  (isothermal dates)
#               observed_mld.jld2                                     (obs)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
# Outputs : figures/LES_eastern_mooring_UVstress_implementations_T_profiles_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_LES_eastern_mooring_UVstress_implementations_T_profiles.jl               # direct (default)
#   julia plot_LES_eastern_mooring_UVstress_implementations_T_profiles.jl coare_wind     # full 3-way comparison
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
const albedo_sw = 0.08     # broadband shortwave albedo (matches experiment)
const ε_water   = 0.98     # longwave emissivity of water (matches experiment)

# Forcing source: "direct" (measured EC fluxes) or "coare_wind".
const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"   # JLD2 flux suffix

const LES_DIR  = joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN")
const LES_STEM = "Lxy256_Lz212_Nxy128_Nz106"
FIGURE_DIR     = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# LES output file for a given winter and implementation tag ("" = default, "_WENO5", "_SmagorinskyLilly").
les_file(year, impl_tag) = joinpath(LES_DIR,
    "LES_GLEN_winter$(year)_$(FORCING_SOURCE)$(impl_tag)_UVstress_$(LES_STEM)",
    "hourly_averaged_timeseries.jld2")

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
function load_glen_forcing(year; ndays = 60)
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
         T1_raw  = f["T1_raw_obs"],   # day-30 observed profile
         T45_raw = f["T45_raw_obs"],
         T2_raw  = f["T2_raw_obs"])  # day-60 observed profile
    end
else
    @warn "EM obs not found ($OBS_EM_FILE) — run process_lake_superior_southern_eastern_moorings.jl first"
    nothing
end

# Observed raw profile (per year) for each column's snapshot day.
obs_profile_for_day(day) = day == 30 ? obs_em.T1_raw  :
                            day == 45 ? obs_em.T45_raw :
                            day == 60 ? obs_em.T2_raw  : nothing

winter_years = [2009, 2010, 2011, 2014, 2015]

#####
##### Load data — the three LES implementations (default UVstress is the full model)
#####
const VARIANTS = [
    (impl_tag = "",                   color = :steelblue4, label = "LES (default: WENO9, no closure)", linestyle = :solid),
    (impl_tag = "_WENO5",             color = :darkorange, label = "LES (WENO5)",                       linestyle = :dash),
    (impl_tag = "_SmagorinskyLilly",  color = :seagreen,   label = "LES (Smagorinsky-Lilly)",           linestyle = :dot),
]

variant_data = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years
        f = les_file(year, v.impl_tag)
        isfile(f) || (@warn "Missing: $f"; continue)
        variant_data[vi][year] = Dict("Tbar" => FieldTimeSeries(f, "Tbar"))
    end
end
all(isempty, variant_data) && error("No LES outputs found for FORCING_SOURCE=$(FORCING_SOURCE)")

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
# shared initial-condition profile, which is common to all three implementations.
function reference_variant(year)
    for vd in variant_data
        haskey(vd, year) && return vd[year]
    end
    return nothing
end

#####
##### Column configuration — target days, not variables
#####
const columns     = [(day = 30, label = "Day 30"), (day = 45, label = "Day 45"), (day = 60, label = "Day 60")]
const T_XLIMS     = (0.0, 5.0)
const WINDOW_EDGES = (0, 30, 45, 60)   # column i covers (WINDOW_EDGES[i], WINDOW_EDGES[i+1]] days

# ── Mean forcing per column window, from gap-filled GLEN NetCDF + T_sfc (LW_up) ──
mean_forcing = Dict{Int, Vector{NamedTuple}}()
for year in winter_years
    forcing  = load_glen_forcing(year)
    dk_ref   = reference_variant(year)
    Tbar_fts = isnothing(dk_ref) ? nothing : dk_ref["Tbar"]

    if isnothing(forcing) || isnothing(Tbar_fts)
        @warn "No GLEN forcing for winter $year (missing date or NetCDF)"
        mean_forcing[year] = [(Qh_Wm2 = NaN, QU_m2s2 = NaN) for _ in 1:(length(WINDOW_EDGES) - 1)]
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

    window_data = NamedTuple[]
    for i in 1:(length(WINDOW_EDGES) - 1)
        lo, hi = WINDOW_EDGES[i], WINDOW_EDGES[i+1]
        Qh, QU = window_means(lo, hi)
        push!(window_data, (Qh_Wm2 = Qh, QU_m2s2 = QU))
    end
    mean_forcing[year] = window_data
end

#####
##### Plotting: rows = winter year, columns = day snapshot
#####
function plot_LES_T_profiles(filename)
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

                # Each of the three LES implementations at this day.
                for (vi, v) in enumerate(VARIANTS)
                    haskey(variant_data[vi], year) || continue
                    dk = variant_data[vi][year]
                    day_available(dk, c.day) || continue
                    lines!(ax, panel_profile(dk, c.day), zC;
                           color = v.color, linewidth = 2.0, linestyle = v.linestyle)
                end

                ylims!(ax, (-Lz_EM - 5, 5))
                xlims!(ax, T_XLIMS)

                # Q̄_h / Q̄_U annotation for the window leading up to this column's day.
                mf = get(mean_forcing, year, nothing)
                if !isnothing(mf) && !isnan(mf[col].Qh_Wm2)
                    f = mf[col]
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
              "Eastern Mooring — LES UVstress implementations, GLEN-forced ($(FORCING_SOURCE))";
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
plot_LES_T_profiles("LES_eastern_mooring_UVstress_implementations_T_profiles_$(FORCING_SOURCE).pdf")
