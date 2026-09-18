#####
##### GLEN-forced Eastern Mooring — LES UVstress implementation comparison (T only, day 60)
#####
# One panel per winter year, all at the day-60 snapshot. Every panel overlays
# four LES implementations, all with the same UV-decomposed (directional) wind
# stress, against the observed temperature profile (T only):
#   • UVstress                       — default LES (WENO(order=9) advection,      (full model)  (blue,   solid)
#                                        no explicit subgrid closure, default Cd)
#   • WENO5_UVstress                 — lower-order (5th) WENO advection            (simplified)  (orange, dashed)
#   • SmagorinskyLilly_UVstress      — explicit Smagorinsky–Lilly subgrid closure  (simplified)  (green,  dotted)
#   • Cd0.002_UVstress               — default LES but with the surface drag       (simplified)  (purple, dash-dot)
#                                        coefficient fixed at Cd = 0.002
# This 4-way comparison is only available for FORCING_SOURCE=coare_wind, and only
# for winters 2009, 2010, 2011, 2014 (direct mostly only has the default
# implementation, and 2015/coare_wind is missing both SmagorinskyLilly and
# Cd0.002); missing runs are skipped with a warning rather than filling the panel.
# The shared initial (day 0) profile is drawn once per panel as a grey dashed line.
# Southern mooring is not included here (no UVstress-implementation LES runs
# exist for it yet).
#
# Each panel is annotated, along the top, with the depth-weighted
# (trapezoidal-integral) mean error (ME, model − obs) and RMSE of every plotted
# implementation against the raw observed sensor-depth temperatures — one
# "value_ME (value_RMSE) °C" row per implementation in that implementation's own
# line color, under a single unlabeled "ME (RMSE)" header — see
# model_obs_stats() below. The block anchors left or right depending on the
# default implementation's surface temperature: profiles converge near the
# surface, so a *small* surface value means the lines cluster on the left (open
# space on the right) and a *large* one means they cluster on the right (open
# space on the left).
# Each panel is also annotated with the mean forcing (Q̄_h, Q̄_U), cumulative
# from day 0 through day 60.
#
# Inputs  : data/LES_outputs/eastern_mooring_GLEN/
#               LES_GLEN_winter<YEAR>_<forcing>[_WENO5|_SmagorinskyLilly|_Cd0.002]_UVstress_
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
#   julia plot_LES_eastern_mooring_UVstress_implementations_T_profiles.jl coare_wind     # full 4-way comparison
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

const DAY = 60   # only snapshot day shown (SM not available for this comparison)

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
function load_glen_forcing(year; ndays = DAY)
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
         T2_raw  = f["T2_raw_obs"])   # day-60 observed profile
    end
else
    @warn "EM obs not found ($OBS_EM_FILE) — run process_lake_superior_southern_eastern_moorings.jl first"
    nothing
end

winter_years = [2009, 2010, 2011, 2014, 2015]

#####
##### Load data — the four LES implementations (default UVstress is the full model)
#####
const VARIANTS = [
    (impl_tag = "",                   color = :steelblue4, label = "LES (default: WENO9, no closure)", linestyle = :solid),
    (impl_tag = "_WENO5",             color = :darkorange, label = "LES (WENO5)",                       linestyle = :dash),
    (impl_tag = "_SmagorinskyLilly",  color = :seagreen,   label = "LES (Smagorinsky-Lilly)",           linestyle = :dot),
    (impl_tag = "_Cd0.002",           color = :purple,     label = "LES (Cd = 0.002)",                  linestyle = :dashdot),
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
# shared initial-condition profile, which is common to all four implementations.
function reference_variant(year)
    for vd in variant_data
        haskey(vd, year) && return vd[year]
    end
    return nothing
end

# ── Model-vs-observation error stats (every plotted implementation) ───────────
# Linear interpolation of a model profile (zC ascending, bottom → surface) onto
# an arbitrary target depth (negative z), clamped to the profile's own ends —
# same clamping convention as interp_to_common() in the mooring processing script.
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

# Depth-weighted (trapezoidal-integral) mean error (model − obs) and RMSE of a
# model profile against the raw observed sensor-depth temperatures (dep_obs
# positive-down; Tobs may contain NaN for dropped sensors, which are skipped).
# Plain point-wise averaging biases the stats toward whatever depth range
# happens to have more sensors (usually near the surface); weighting each
# error by half the distance to its neighboring sensors instead approximates
#   ME   = ∫ err(z)   dz / ∫ dz
#   RMSE = √(∫ err(z)² dz / ∫ dz)
# over the sensor span, so widely- and tightly-spaced sensors count equally
# per unit depth. Falls back to the single point when only one sensor
# overlaps (no interval to weight by). Returns (ME = NaN, RMSE = NaN) if no
# valid sensor overlaps.
function model_obs_stats(Tmodel, dep_obs, Tobs)
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

#####
##### Mean forcing per year, cumulative from day 0 through day 60
#####
function mean_forcing_at(year, day)
    forcing  = load_glen_forcing(year; ndays = day)
    dk_ref   = reference_variant(year)
    Tbar_fts = isnothing(dk_ref) ? nothing : dk_ref["Tbar"]

    if isnothing(forcing) || isnothing(Tbar_fts)
        @warn "No GLEN forcing for winter $year (missing date or NetCDF)"
        return (Qh_Wm2 = NaN, QU_m2s2 = NaN)
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
    mask = (t_forc .> 0.0) .& (t_forc .<= day * 86400.0)
    any(mask) || return (Qh_Wm2 = NaN, QU_m2s2 = NaN)
    ts = t_forc[mask]
    Qh = nanmean(Q_pre[mask] .+ lw_up_at.(ts))
    QU = nanmean(tau_kin[mask])
    return (Qh_Wm2 = Qh, QU_m2s2 = QU)
end

mean_forcing = Dict(year => mean_forcing_at(year, DAY) for year in winter_years)

#####
##### Plotting: 2 x 3 grid — one panel per winter year, day-60 snapshot only,
##### legend filling the last (unused) panel slot.
#####
const NROWS         = 2
const NCOLS         = 3
const T_XLIMS       = (0.0, 5.0)
const T_XTICKS      = 0:1:5
const T_XMINORTICKS = IntervalsBetween(2)   # unlabelled minor tick every 0.5 °C

# (year, row, col) for every data panel, filled row-major; the legend takes the
# next slot after the last winter year.
const PANELS = [(year = y, row = ((i - 1) ÷ NCOLS) + 1, col = ((i - 1) % NCOLS) + 1)
                for (i, y) in enumerate(winter_years)]
const LEGEND_ROW, LEGEND_COL = let i = length(winter_years) + 1
    ((i - 1) ÷ NCOLS) + 1, ((i - 1) % NCOLS) + 1
end

function plot_LES_T_profiles(filename)
    fig = Figure(size = (220 * NCOLS + 40, 230 * NROWS + 70),
                 fontsize = 11, figure_padding = (6, 10, 6, 4))

    axes = Axis[]
    for p in PANELS
        year, row, col = p.year, p.row, p.col
        ax = CairoMakie.Axis(fig[row, col];
                 title              = "Winter $year",
                 xlabel             = L"T \; (^\circ\mathrm{C})",
                 ylabel             = col == 1 ? L"z\;(\mathrm{m})" : "",
                 ylabelrotation     = π/2,
                 yticklabelsvisible = col == 1,
                 xticks             = T_XTICKS,
                 xminorticks        = T_XMINORTICKS,
                 xminorticksvisible = true,
                 xminortickalign    = 0,
                 xgridvisible       = true,
                 ygridvisible       = true,
                 xticksize          = 4,
                 xminorticksize     = 2.5,
                 yticksize          = 4)
        push!(axes, ax)

        dk_ref = reference_variant(year)
        if isnothing(dk_ref)
            text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                  align = (:center, :center), fontsize = 9, color = :gray)
            ylims!(ax, (-Lz_EM - 5, 5))
            xlims!(ax, T_XLIMS)
            continue
        end

        # Shared initial profile — single grey dashed line.
        lines!(ax, panel_profile(dk_ref, 0), zC;
               color = :gray40, linewidth = 1.5, linestyle = :dash)

        stats_entries  = NamedTuple[]   # (color, ME, RMSE) — every implementation plotted in this panel
        v1_profile     = nothing        # default implementation's profile, used only for text-side placement

        for (vi, v) in enumerate(VARIANTS)
            dk = get(variant_data[vi], year, nothing)
            (isnothing(dk) || !day_available(dk, DAY)) && continue
            profile = panel_profile(dk, DAY)
            vi == 1 && (v1_profile = profile)
            lines!(ax, profile, zC;
                   color = v.color, linewidth = 2.0, linestyle = v.linestyle)
            push!(stats_entries, (color = v.color, profile = profile))
        end

        ylims!(ax, (-Lz_EM - 5, 5))
        xlims!(ax, T_XLIMS)

        # Observations — day 60 only.
        wy_idx = nothing
        if !isnothing(obs_em)
            wy_idx = findfirst(==(year), obs_em.wys)
            if !isnothing(wy_idx)
                scatter!(ax, obs_em.T2_raw[wy_idx], -obs_em.dep_raw[wy_idx];
                         color = :firebrick, marker = :circle, markersize = 8)
            end
        end

        # ME (RMSE) (model vs. obs) — stacked along the top under a single "ME
        # (RMSE)" header, one value line per plotted implementation in that
        # implementation's own line color. Anchored left or right depending on
        # where the surface (z ≈ 0) default implementation's temperature sits in
        # the T range: profiles converge near the surface, so a *small* surface
        # value means the lines cluster on the left, leaving the open space on
        # the right (and vice versa).
        computed_stats = NamedTuple[]
        if !isnothing(wy_idx)
            dep_o, T_o = obs_em.dep_raw[wy_idx], obs_em.T2_raw[wy_idx]
            for e in stats_entries
                s = model_obs_stats(e.profile, dep_o, T_o)
                isnan(s.ME) || push!(computed_stats, (color = e.color, ME = s.ME, RMSE = s.RMSE))
            end
        end

        mid_T     = sum(T_XLIMS) / 2
        surface_T = !isnothing(v1_profile) ? v1_profile[end] :
                    !isempty(stats_entries) ? stats_entries[1].profile[end] : nothing
        anchor_x, halign = isnothing(surface_T) || surface_T > mid_T ? (0.03, :left) : (0.97, :right)
        if !isempty(computed_stats)
            text!(ax, anchor_x, 0.97;
                  text = "ME (RMSE)", space = :relative, align = (halign, :top), fontsize = 10)
            for (i, e) in enumerate(computed_stats)
                text!(ax, anchor_x, 0.97 - i * 0.08;
                      text  = latexstring(@sprintf("%+.2f", e.ME), "\\ (", @sprintf("%.2f", e.RMSE),
                                          ")\\;^\\circ\\mathrm{C}"),
                      space = :relative, align = (halign, :top), color = e.color, fontsize = 10)
            end
        end

        # Q̄_h / Q̄_U annotation — cumulative mean forcing from day 0 to day 60.
        mf = get(mean_forcing, year, nothing)
        if !isnothing(mf) && !isnan(mf.Qh_Wm2)
            text!(ax, 0.03, 0.11;
                  text  = latexstring("\\bar{Q}_h = ", round(Int, mf.Qh_Wm2), "\\;\\mathrm{W\\,m^{-2}}"),
                  space = :relative, align = (:left, :bottom), fontsize = 10)
            text!(ax, 0.03, 0.03;
                  text  = latexstring("\\bar{Q}_U = ", sci_latex(abs(mf.QU_m2s2)), "\\;\\mathrm{m^2\\,s^{-2}}"),
                  space = :relative, align = (:left, :bottom), fontsize = 10)
        end
    end

    length(axes) > 1 && linkyaxes!(axes...)
    length(axes) > 1 && linkxaxes!(axes...)

    # ── Legend: initial profile, obs, variant colour key ─────────────────
    legend_elems  = Any[LineElement(color = :gray40, linewidth = 1.5, linestyle = :dash),
                        MarkerElement(color = :firebrick, marker = :circle, markersize = 8)]
    legend_labels = Any["t = 0 (initial)", "Observations"]
    for v in VARIANTS
        push!(legend_elems,  LineElement(color = v.color, linewidth = 2.0, linestyle = v.linestyle))
        push!(legend_labels, v.label)
    end
    Legend(fig[LEGEND_ROW, LEGEND_COL], legend_elems, legend_labels;
           labelsize = 11, framevisible = false, patchsize = (20, 12), tellwidth = false)

    Label(fig[0, 1:NCOLS],
          "Eastern Mooring — LES UVstress implementations, day 60 (GLEN-forced, $(FORCING_SOURCE))";
          fontsize = 11, font = :bold, justification = :center)

    for col in 1:NCOLS
        colsize!(fig.layout, col, Relative(1 / NCOLS))
    end

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 6)

    outfile = joinpath(FIGURE_DIR, filename)
    save(outfile, fig)
    @info "Saved → $outfile"
end

#####
##### Generate figure
#####
plot_LES_T_profiles("LES_eastern_mooring_UVstress_implementations_T_profiles_$(FORCING_SOURCE).pdf")
