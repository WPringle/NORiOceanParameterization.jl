#####
##### GLEN-forced Lake Superior Moorings — bottom temperature T_bot(t) time series
#####
# 2 x 3 grid: the 4 Eastern Mooring winters with bottom-temperature observations
# (2009, 2011, 2014, 2015 — 2010's EM deployment never reached the lake bottom,
# see process_lake_superior_southern_eastern_moorings.jl) plus the 2 Southern
# Mooring winters (2010, 2011 — the only two with a TURB run). Each panel overlays:
#   • LES        — bottom-cell Tbar, UVstress                              (black)
#                   (Eastern Mooring only; no LES has been run for the Southern
#                   Mooring site, so those two panels omit this line)
#   • k-ε        — bottom-cell Tbar, UVstress + TEOS-10 EOS (the default)  (blue)
#   • Simple model (dashed, purple) — a diagnostic argument for T_bot, evaluated
#     hourly (not daily) directly from the raw GLEN forcing, with both the
#     Monin–Obukhov length L_MO *and* the upwelling-longwave correction in Q_T
#     built from the *observed* surface temperature rather than the model's
#     simulated surface temperature — i.e. the same formula the experiment
#     script uses for its (daily) L_MO diagnostic,
#         L_MO = -|Q|^(3/2) / (κ_vonK α_sfc g Q_T),   α_sfc = gsw_alpha(S, CT_sfc, 0)
#     but with CT_sfc from the observed surface sensor at hourly resolution, so
#     short wind gusts aren't averaged away before the mixing gate sees them.
#     Heat lost (or gained) at the surface reaches the bottom in hours the column
#     is judged fully mixed: either L_MO < 0 *and* Q_T < 0 (convectively unstable
#     — warming back toward T_MD while below it increases density there,
#     regardless of |L_MO|) or L_MO > H (stable, but wind-shear mixing alone is
#     strong enough to reach the bottom). Ordinary cooling while above T_MD
#     (also L_MO < 0, but with Q_T > 0) is deliberately left to the L_MO > H
#     pathway instead. Otherwise T_bot is left unchanged that hour:
#       T_bot(t) = T_start - (1/H) ∫₀ᵗ Q_T(t') 𝟙[mixed to bottom](t') dt'
#     with Q_T the kinematic heat flux (K·m/s), H the site's water depth, and
#     T_start = 4 °C the isothermal winter-start bottom temperature (not T_MD).
#   • Observations — continuous hourly record at the deepest sensor           (red)
#
# Inputs  : data/TURB_outputs/{eastern,southern}_mooring_GLEN_forced/winter<YEAR>/
#               kepsilon[_coare_wind]_UVstress_winter<YEAR>.jld2      (Tbar)
#           data/LES_outputs/eastern_mooring_GLEN/
#               LES_GLEN_winter<YEAR>_<forcing>_UVstress_Lxy256_Lz212_Nxy128_Nz106/
#                   hourly_averaged_timeseries.jld2   (Tbar)
#           figure_data/lake_superior_{eastern,southern}_mooring/
#               lake_superior_{eastern,southern}_mooring_winter_start_dates.csv  (isothermal dates)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
#           Austin2023 hourly .mat files (raw surface/bottom-sensor observations,
#               EM*.mat for the Eastern Mooring, SM*.mat for the Southern Mooring)
# Outputs : figures/lake_superior_mooring_Tbot_timeseries_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_eastern_mooring_Tbot_timeseries.jl               # direct (default)
#   julia plot_eastern_mooring_Tbot_timeseries.jl coare_wind
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

const S_lake    = 0.05    # g/kg — Lake Superior absolute salinity (matches other figure scripts)
const ρ₀        = 999.8
const g_grav    = 9.80665
const κ_vonK    = 0.41    # von Kármán constant (matches the experiment script's L_MO callback)
const cₚ        = 4182.0  # J/(kg·K) — freshwater specific heat (matches experiment)
const albedo_sw = 0.08
const ε_water   = 0.98

# Simple model's initial bottom temperature — the isothermal winter-start state
# (matches the ~4 °C isothermal criterion used to pick t_iso itself), not T_MD.
const T_START = 4.0

# Forcing source: "direct" (measured EC fluxes) or "coare_wind".
const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"

const LES_STEM = "Lxy256_Lz212_Nxy128_Nz106"
const OBS_DIR  = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/Austin2023"
FIGURE_DIR      = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# ── Mooring site definitions ───────────────────────────────────────────────────
# Southern Mooring has no LES run (les_dir = nothing) — those panels simply omit
# the LES line rather than warning about a "missing" file that was never run.
struct MooringSite
    name       :: String   # for panel titles
    turb_dir   :: String
    les_dir    :: Union{String, Nothing}
    csv_file   :: String
    obs_prefix :: String    # "EM" or "SM" — Austin2023 raw .mat filename prefix
end

const EM_SITE = MooringSite(
    "Eastern",
    joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced"),
    joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN"),
    joinpath(@__DIR__, "..", "figure_data", "lake_superior_eastern_mooring",
             "lake_superior_eastern_mooring_winter_start_dates.csv"),
    "EM")

const SM_SITE = MooringSite(
    "Southern",
    joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced"),
    nothing,
    joinpath(@__DIR__, "..", "figure_data", "lake_superior_southern_mooring",
             "lake_superior_southern_mooring_winter_start_dates.csv"),
    "SM")

# (site, year) panel list — 2 x 3 grid, row-major.
const PANELS = [(site = EM_SITE, year = 2009), (site = EM_SITE, year = 2011), (site = EM_SITE, year = 2014),
                (site = EM_SITE, year = 2015), (site = SM_SITE, year = 2010), (site = SM_SITE, year = 2011)]

turb_file(site, year) = joinpath(site.turb_dir, "winter$(year)", "kepsilon$(SRC_TAG)_UVstress_winter$(year).jld2")
les_file(site, year)  = isnothing(site.les_dir) ? nothing :
    joinpath(site.les_dir, "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_UVstress_$(LES_STEM)",
             "hourly_averaged_timeseries.jld2")

const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"

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

# A few short (few-point) NaN gaps remain in the "gap-filled" GLEN NetCDF (e.g.
# downwelling_shortwave_flux/downwelling_longwave_flux both have a 5-point gap
# in the winter-2009 window). Left unfilled, a single NaN sample poisons every
# later cumtrapz value forever, silently truncating the whole rest of the
# integral — so linearly fill any remaining gaps (clamping at the ends) before
# they're used, same spirit as the experiment script's own fill_missing_linear.
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

# ── Raw GLEN forcing at native half-hourly resolution, [t_iso, t_iso + ndays] ──
# Reproduces the experiment script's own Q_precomp_Wm2 / Qu_vals / Qv_vals exactly
# (see lake_superior_eastern_mooring_GLEN_forced_UVstress.jl), just without the
# LW_up(T_sfc) term — that's added back in separately using observed T_sfc. Same
# GLEN forcing file is used for both mooring sites.
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
        wdir = getv("wind_direction")   # shared between direct/coare_wind

        t_days = Float64[Dates.value(times[i] - t_iso) / (1000 * 86400.0) for i in idx]
        Q_precomp_Wm2 = @. shf + lhf - (1.0 - albedo_sw) * sw - ε_water * lw

        # Directional kinematic momentum flux: êₓ = sinθ, êᵧ = cosθ (θ = wind_direction).
        ex = sind.(wdir)
        ey = cosd.(wdir)
        τ_mag_kin = mom ./ ρ₀
        Qu_vals = τ_mag_kin .* ex
        Qv_vals = τ_mag_kin .* ey

        return t_days, fill_missing_linear(Q_precomp_Wm2),
               fill_missing_linear(Qu_vals), fill_missing_linear(Qv_vals)
    end
end

# ── Model bottom-cell temperature time series ─────────────────────────────────
# Bottom-cell in-situ temperature (index 1 = deepest cell, matching how the
# surface-cell index "end" is used elsewhere in these scripts).
function bottom_Tinsitu_series(fts)
    zC     = znodes(fts.grid, Center())
    p_dbar = ρ₀ * g_grav * abs(zC[1]) / 1e4
    Θ_bot  = [Float64(interior(fts[i])[1, 1, 1]) for i in eachindex(fts.times)]
    T_bot  = [gsw_t_from_ct(S_lake, Θ, p_dbar) for Θ in Θ_bot]
    return Float64.(fts.times) ./ 86400.0, T_bot
end

# ── Observed sensor time series (continuous hourly record) ───────────────────
# sensor = :bottom (argmax depth) or :surface (argmin depth).
t2dt(t::Real) = DateTime(2000, 1, 1) + Millisecond(round(Int64, (t - 730486.0) * 86_400_000.0))

function observed_sensor_series(site, year, t_iso, sensor::Symbol; ndays = 60)
    isnothing(t_iso) && return (Float64[], Float64[])
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

            dep   = vec(Float64.(reshape(data["dep"], :)))
            T_raw = data["T"]
            Ndep, Nt = length(dep), length(t_raw)
            T = size(T_raw) == (Ndep, Nt) ? Float64.(T_raw) : Float64.(T_raw')
            sensor_idx = sensor === :bottom ? argmax(dep) : argmin(dep)

            idx  = findall(dt -> t_iso <= dt <= t_iso + Day(ndays), dts)
            days = [Dates.value(dts[i] - t_iso) / (1000 * 86400.0) for i in idx]
            return days, T[sensor_idx, idx]
        end
    end
    return Float64[], Float64[]
end

# Linear interpolation of a (days, values) record onto arbitrary query days.
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

# ── Simple diagnostic T_bot model, L_MO recomputed from *observed* surface T ──
# Mixing is judged to reach the bottom either when L_MO < 0 *and* Q_T < 0 (the
# column is warming while below T_MD, α < 0 — destabilizing regardless of
# |L_MO|, since warming toward T_MD increases density there) or when L_MO > H
# (stable, but wind-shear mixing alone is strong enough to reach the bottom
# anyway). It is judged not to reach the bottom otherwise — including the
# L_MO < 0 & Q_T > 0 case (ordinary cooling while above T_MD), which is left to
# the L_MO > H wind-driven pathway instead.
mixed_to_bottom(L_MO, QT, H) = (L_MO < 0 && QT < 0) || (L_MO > H)

function cumtrapz(t, y)
    out = zeros(length(t))
    for i in 2:length(t)
        # Defensive fallback: a stray NaN sample must not poison the running sum
        # for the rest of the record — treat it as a zero-contribution interval.
        step = 0.5 * (y[i] + y[i-1]) * (t[i] - t[i-1])
        out[i] = out[i-1] + (isnan(step) ? 0.0 : step)
    end
    return out
end

function simple_Tbot_model(t_iso, H, obs_sfc_days, obs_sfc_T; ndays = 60)
    t_glen_days, Q_precomp_Wm2, Qu, Qv = load_glen_forcing_raw(t_iso; ndays)
    (isempty(t_glen_days) || isempty(obs_sfc_days)) && return (Float64[], Float64[])

    # Evaluate hourly (not at the GLEN record's native half-hourly cadence) —
    # matches the model/obs Tbar cadence and is plenty to resolve wind gusts
    # without the noise of the raw half-hourly record.
    t_days = collect(0.0:(1 / 24):ndays)
    t_sec  = t_days .* 86400.0

    Q_precomp_Wm2_h = interp_at(t_glen_days, Q_precomp_Wm2, t_days)
    Qu_h            = interp_at(t_glen_days, Qu, t_days)
    Qv_h            = interp_at(t_glen_days, Qv, t_days)
    T_obs_sfc       = interp_at(obs_sfc_days, obs_sfc_T, t_days)

    # Q_T = precomputed forcing + LW_up(T_sfc), both in kinematic units (K·m/s) —
    # using *observed* T_sfc for LW_up too, not just for α, so the whole diagnostic
    # is built from observations rather than mixing in the model's simulated T_sfc.
    CT_obs_sfc = [gsw_ct_from_t(S_lake, T, 0.0) for T in T_obs_sfc]
    α_obs_sfc  = [gsw_alpha(S_lake, CT, 0.0) for CT in CT_obs_sfc]
    LW_up_kin  = @. ε_water * 5.67e-8 * (T_obs_sfc + 273.15)^4 / (ρ₀ * cₚ)
    QT         = Q_precomp_Wm2_h ./ (ρ₀ * cₚ) .+ LW_up_kin

    Qmag = sqrt.(Qu_h .^ 2 .+ Qv_h .^ 2)
    L_MO = map(eachindex(t_days)) do i
        abs(QT[i]) < 1e-10 && return -Inf   # no heat flux ⇒ never mixes the column
        -Qmag[i]^1.5 / (κ_vonK * α_obs_sfc[i] * g_grav * QT[i])
    end

    integrand = QT .* [mixed_to_bottom(L_MO[i], QT[i], H) ? 1.0 : 0.0 for i in eachindex(L_MO)]
    T_bot     = T_START .- cumtrapz(t_sec, integrand) ./ H
    return t_days, T_bot
end

#####
##### Load everything
#####
panel_data = Dict{Tuple{String,Int}, NamedTuple}()
for (site, year) in PANELS
    key   = (site.obs_prefix, year)
    t_iso = read_isothermal_date(site.csv_file, year)

    turb_f = turb_file(site, year)
    les_f  = les_file(site, year)
    les_days, les_T = if !isnothing(les_f) && isfile(les_f)
        bottom_Tinsitu_series(FieldTimeSeries(les_f, "Tbar"))
    else
        isnothing(les_f) || @warn "Missing LES output: $les_f"
        (Float64[], Float64[])
    end

    if !isfile(turb_f)
        @warn "Missing k-ε output: $turb_f"
        continue
    end
    turb_fts          = FieldTimeSeries(turb_f, "Tbar")
    turb_days, turb_T = bottom_Tinsitu_series(turb_fts)
    H                 = turb_fts.grid.Lz

    obs_days, obs_T         = observed_sensor_series(site, year, t_iso, :bottom)
    obs_sfc_days, obs_sfc_T = observed_sensor_series(site, year, t_iso, :surface)
    model_days, model_T     = simple_Tbot_model(t_iso, H, obs_sfc_days, obs_sfc_T)

    panel_data[key] = (les_days = les_days, les_T = les_T,
                       turb_days = turb_days, turb_T = turb_T,
                       model_days = model_days, model_T = model_T,
                       obs_days = obs_days, obs_T = obs_T)
end
isempty(panel_data) && error("No k-ε TURB outputs found for FORCING_SOURCE=$(FORCING_SOURCE)")

#####
##### Plotting: 2 x 3 grid, one subplot per (site, winter) panel
#####
function plot_Tbot_timeseries(filename)
    fig = Figure(size = (1090, 620), fontsize = 11, figure_padding = (6, 10, 6, 4))

    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    axes      = Axis[]
    for (i, (site, year)) in enumerate(PANELS)
        row, col = positions[i]
        ax = CairoMakie.Axis(fig[row, col];
                 title  = "$(site.name) Mooring, Winter $year",
                 xlabel = L"\mathrm{Days\ since\ start}",
                 ylabel = L"T_{\mathrm{bot}} \; (^\circ\mathrm{C})")
        push!(axes, ax)

        d = get(panel_data, (site.obs_prefix, year), nothing)
        if isnothing(d)
            text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                  align = (:center, :center), fontsize = 11, color = :gray)
            continue
        end

        !isempty(d.obs_days)   && lines!(ax, d.obs_days, d.obs_T; color = :firebrick, linewidth = 1.5)
        !isempty(d.les_days)   && lines!(ax, d.les_days, d.les_T; color = :black, linewidth = 2.0)
        !isempty(d.turb_days)  && lines!(ax, d.turb_days, d.turb_T; color = :steelblue4, linewidth = 2.0)
        !isempty(d.model_days) && lines!(ax, d.model_days, d.model_T;
                                          color = :purple, linewidth = 2.0, linestyle = :dash)

        xlims!(ax, (0.0, 60.0))
    end

    linkyaxes!(axes...)
    linkxaxes!(axes...)

    legend_elems = [LineElement(color = :firebrick, linewidth = 1.5),
                    LineElement(color = :black, linewidth = 2.0),
                    LineElement(color = :steelblue4, linewidth = 2.0),
                    LineElement(color = :purple, linewidth = 2.0, linestyle = :dash)]
    legend_labels = ["Observations", "LES", "k-ε", "Simple model"]
    Legend(fig[3, 1:3], legend_elems, legend_labels;
           orientation = :horizontal, tellwidth = false, labelsize = 11,
           framevisible = false, padding = (0, 0, 0, 0))

    Label(fig[0, 1:3],
          "Lake Superior Moorings — bottom temperature evolution (GLEN-forced, $(FORCING_SOURCE))";
          fontsize = 11, font = :bold, justification = :center)

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)

    outfile = joinpath(FIGURE_DIR, filename)
    save(outfile, fig)
    @info "Saved → $outfile"
end

plot_Tbot_timeseries("lake_superior_mooring_Tbot_timeseries_$(FORCING_SOURCE).pdf")
