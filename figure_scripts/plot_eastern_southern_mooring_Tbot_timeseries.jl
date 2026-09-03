#####
##### GLEN-forced Lake Superior Moorings — bottom temperature T_bot(t) time series
#####
# 2 x 4 grid (legend in the last slot), matching both the panel arrangement and
# year/mooring list of plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl:
# row 1 is the Eastern Mooring winters 2009/2010/2011/2014; row 2 is Eastern
# Mooring winter 2015, then the 2 Southern Mooring winters (2010, 2011 — the only
# two with a TURB run), then the legend. Note that EM winter 2010's deployment
# never reached the true lake bottom (its deepest sensor is well short of the
# model's 212 m column) — see process_lake_superior_southern_eastern_moorings.jl
# and the per-panel recorded site water depth Z / deepest-sensor-depth values
# @info-logged near the bottom of this script (for adding to the figure
# caption), since both vary by deployment/year/site. Each panel's xlabel spells
# out its own calendar isothermal-start date ("Days since 2008-12-16") rather
# than a generic "Days since start", since that date differs by site/year (see
# the EM-vs-SM Q̄_h/Q̄_U discussion earlier in this conversation). Each panel
# overlays:
#   • LES        — Tbar sampled at the observed deepest sensor's own depth      (black)
#                   (not the grid's bottom cell / Lz — several deployments'
#                   deepest sensor sits well short of the true lake bottom, so
#                   sampling at Lz would compare the model against a different
#                   depth than the sensor actually measures; see
#                   bottom_Tinsitu_series below)
#                   (Eastern Mooring only; no LES has been run for the Southern
#                   Mooring site, so those two panels omit this line)
#   • k-ε        — Tbar sampled at the same observed sensor depth as LES        (blue)
#                   (UVstress + TEOS-10 EOS, the default)
#   • Analytical model (dashed, purple) — a diagnostic argument for T_bot, evaluated
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
#       T_bot(t) = T_bot(0) - (1/H) ∫₀ᵗ Q_T(t') 𝟙[mixed to bottom](t') dt'
#     with Q_T the kinematic heat flux (K·m/s); H the deployment's own recorded
#     water depth Z (not the model grid's Lz — falls back to Lz only if Z is
#     unavailable; see the per-panel Z/Lz comparison @info-logged below); and
#     T_bot(0) the *observed* bottom temperature at t = 0 (not a fixed 4 °C —
#     actual starting values vary by site/year, e.g. SM 2010/2011 start at
#     3.94/3.83 °C; falls back to 4 °C only if no observation is available).
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

# Analytical model's fallback initial bottom temperature, used only if no
# observed T_bot(0) is available to start from (see simple_Tbot_model below) —
# the nominal ~4 °C isothermal criterion used to pick t_iso itself, not T_MD.
# Actual observed starting values vary by site/year and can differ meaningfully
# from 4 °C (e.g. SM winter 2010 starts at 3.94 °C, SM winter 2011 at 3.83 °C).
const T_START_FALLBACK = 4.0

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

const EM_CSV_FILE = joinpath(@__DIR__, "..", "figure_data", "lake_superior_eastern_mooring",
                              "lake_superior_eastern_mooring_winter_start_dates.csv")

const EM_SITE = MooringSite(
    "Eastern",
    joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced"),
    joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN"),
    EM_CSV_FILE,
    "EM")

# The southern mooring experiment script now also reads t_iso from the eastern
# mooring's isothermal-date CSV rather than its own (SM runs are re-initialized
# to start at the same calendar date as the EM runs) — see the equivalent note
# in plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl — so this
# site definition points at EM_CSV_FILE too, not its own
# lake_superior_southern_mooring_winter_start_dates.csv.
const SM_SITE = MooringSite(
    "Southern",
    joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced"),
    nothing,
    EM_CSV_FILE,
    "SM")

# (site, year) panel list — matches plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl's
# EM_ROW1_YEARS/EM_ROW2_YEAR/winter_years_sm — 4 x 2 grid, row-major, legend in the last slot.
const PANELS = [(site = EM_SITE, year = 2009), (site = EM_SITE, year = 2010),
                (site = EM_SITE, year = 2011), (site = EM_SITE, year = 2014),
                (site = EM_SITE, year = 2015), (site = SM_SITE, year = 2010),
                (site = SM_SITE, year = 2011)]

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

# ── Model temperature time series at a given depth ────────────────────────────
# Sampled at `target_depth` (m, positive down) — the *observed* deepest sensor's
# own mounting depth, not the model grid's own bottom cell / the site's recorded
# water depth Z. Several deployments' deepest sensor sits well short of the true
# lake bottom (e.g. EM winter 2010 — see the note by observed_sensor_series
# below), so comparing the model's actual bottom cell (near Lz ≈ Z) against that
# sensor would be comparing two different depths, not a model-vs-obs difference.
# Linearly interpolated between the two nearest cell centers; `target_depth` is
# clamped to the grid's own depth range first (Center() nodes run bottom→top, so
# zC[1] is the deepest cell — matches how the surface-cell index "end" is used
# elsewhere in these scripts).
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

# SMS09h.mat (Southern Mooring, winter 2010) records every one of its 15 sensor
# depths a uniform ~18 m deeper than the equivalent sensor in the following
# year's deployment (SMS10h.mat), including a deepest sensor at 396 m versus a
# recorded site water depth of only 380 m. The data producer suspects the
# deepest value is a typo, but since the offset is uniform across the whole
# chain rather than isolated to one sensor, every depth in this one file is
# shifted up by 20 m (matches process_lake_superior_southern_eastern_moorings.jl)
# — deepest sensor becomes 376 m. Revert once the data producer confirms/fixes
# the source file.
function apply_known_depth_corrections!(dep, filename)
    filename == "SMS09h.mat" && (dep .-= 20.0)
    return dep
end

# ── Observed sensor time series (continuous hourly record) ───────────────────
# sensor = :bottom (argmax depth) or :surface (argmin depth). Also returns Z
# (the deployment's own recorded nominal total water depth, for reporting/
# captioning against the model's column depth H) and dep_sensor, the *actual*
# mounting depth of the chosen sensor — used to sample the model's own T field
# at the same depth (see bottom_Tinsitu_series), since Z and dep_sensor can
# differ substantially: e.g. SMS09h.mat's deepest sensor reads deeper than its
# own file's Z, and EM winter 2010's deployment never sent a sensor anywhere
# near the true bottom.
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

function simple_Tbot_model(t_iso, H, T_bot0, obs_sfc_days, obs_sfc_T; ndays = 60)
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
    T_bot     = T_bot0 .- cumtrapz(t_sec, integrand) ./ H
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

    if !isfile(turb_f)
        @warn "Missing k-ε output: $turb_f"
        continue
    end
    turb_fts = FieldTimeSeries(turb_f, "Tbar")
    H        = turb_fts.grid.Lz

    obs_days, obs_T, obs_Z, obs_bottom_dep = observed_sensor_series(site, year, t_iso, :bottom)
    obs_sfc_days, obs_sfc_T, _, _          = observed_sensor_series(site, year, t_iso, :surface)

    # Sample the model fields at the *observed* deepest sensor's own depth, not
    # the model grid's bottom cell (≈ Lz) or the site's recorded Z — falls back
    # to H (grid Lz) only if no bottom-sensor observation was found at all.
    sample_depth = isnan(obs_bottom_dep) ? H : obs_bottom_dep

    turb_days, turb_T = bottom_Tinsitu_series(turb_fts, sample_depth)
    les_days, les_T = if !isnothing(les_f) && isfile(les_f)
        bottom_Tinsitu_series(FieldTimeSeries(les_f, "Tbar"), sample_depth)
    else
        isnothing(les_f) || @warn "Missing LES output: $les_f"
        (Float64[], Float64[])
    end

    # Start the analytical model from the *observed* T_bot(0), not a fixed 4 °C —
    # actual starting values vary by site/year (e.g. SM 2010/2011 start at 3.94/3.83
    # °C, not 4.0), which matters most for the deep Southern Mooring column.
    T_bot0 = isempty(obs_days) ? T_START_FALLBACK : interp_at(obs_days, obs_T, [0.0])[1]

    # Use the deployment's own recorded water depth Z for the analytical model's
    # H (not the model grid's Lz) — falls back to Lz only if Z is unavailable.
    # Unlike sample_depth above, this is a physical heat-budget depth for the
    # lumped 0-D model, not a sampling location, so it stays keyed to Z.
    H_analytical = isnan(obs_Z) ? H : obs_Z

    model_days, model_T = simple_Tbot_model(t_iso, H_analytical, T_bot0, obs_sfc_days, obs_sfc_T)

    panel_data[key] = (les_days = les_days, les_T = les_T,
                       turb_days = turb_days, turb_T = turb_T,
                       model_days = model_days, model_T = model_T,
                       obs_days = obs_days, obs_T = obs_T, obs_Z = obs_Z,
                       obs_bottom_dep = obs_bottom_dep, H = H, t_iso = t_iso)
end
isempty(panel_data) && error("No k-ε TURB outputs found for FORCING_SOURCE=$(FORCING_SOURCE)")

# Report the deployment's own recorded water depth Z and the deepest sensor's
# own mounting depth per panel (alongside the model's water-column depth H, for
# comparison) so it can be added to the figure caption — and to make it obvious
# when a panel's model/LES lines are being sampled well short of the true H.
@info "Recorded site water depth Z / deepest-sensor depth per panel (model water-column depth H for comparison):"
for (site, year) in PANELS
    d = get(panel_data, (site.obs_prefix, year), nothing)
    isnothing(d) && continue
    @info "  $(site.name) Winter $year: Z = $(round(d.obs_Z, digits=1)) m, " *
          "deepest sensor = $(round(d.obs_bottom_dep, digits=1)) m  (model H = $(round(d.H, digits=1)) m)"
end

#####
##### Plotting: 2 x 4 grid, one subplot per (site, winter) panel, legend last
##### (matches plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl's
##### panel arrangement: EM 2009/2010/2011/2014 on row 1, EM 2015 + SM 2010/2011
##### + legend on row 2)
#####
const TBOT_XTICKS = 0:10:60   # day-since-start ticks

function plot_Tbot_timeseries(filename)
    fig = Figure(size = (195 * 4 + 40, 210 * 2 + 70), fontsize = 11, figure_padding = (6, 10, 6, 4))

    positions = [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3)]
    axes      = Axis[]
    for (i, (site, year)) in enumerate(PANELS)
        row, col = positions[i]
        d = get(panel_data, (site.obs_prefix, year), nothing)

        # The isothermal start date differs by panel (site/year), so it's spelled
        # out in the xlabel itself — "Days since 2008-12-16" — rather than a
        # generic "Days since start" alongside plain day-count tick labels.
        xlabel_str = isnothing(d) ? "Days since start" :
                     "Days since $(Dates.format(d.t_iso, "yyyy-mm-dd"))"

        ax = CairoMakie.Axis(fig[row, col];
                 title              = "$(site.name) Mooring, Winter $year",
                 xlabel             = xlabel_str,
                 ylabel             = col == 1 ? L"T_{\mathrm{bot}} \; (^\circ\mathrm{C})" : "",
                 yticklabelsvisible = col == 1,
                 yticksvisible      = col == 1,
                 xticks             = collect(TBOT_XTICKS))
        push!(axes, ax)

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
    legend_labels = ["Observations", "LES", "k-ε", "Analytical model"]
    Legend(fig[2, 4], legend_elems, legend_labels;
           tellwidth = false, labelsize = 11, framevisible = false, padding = (0, 0, 0, 0))

    Label(fig[0, 1:4],
          "Lake Superior Moorings — bottom temperature evolution (GLEN-forced, $(FORCING_SOURCE))";
          fontsize = 11, font = :bold, justification = :center)

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)

    outfile = joinpath(FIGURE_DIR, filename)
    save(outfile, fig)
    @info "Saved → $outfile"
end

plot_Tbot_timeseries("lake_superior_mooring_Tbot_timeseries_$(FORCING_SOURCE).pdf")
