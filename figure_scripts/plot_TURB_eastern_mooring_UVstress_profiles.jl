#####
##### GLEN-forced Eastern Mooring — TURB k-ε profile plots (original vs UV stress)
#####
# Same layout as the LES profile figures: rows = winter year, columns = variable
# (T, |u|, N²).  Two k-ε variants are overlaid on every panel:
#   • original   — all wind momentum placed on the u-stress            (blue)
#   • UV stress  — momentum split between u and v by wind direction    (orange)
# Each shows snapshot profiles at days 0, 7, 15, 22, 30 (month 1) or
# 30, 37, 45, 52, 60 (month 2); the first column is labelled with the mean
# Q_T (W/m²) and Q_U (m²/s²) over the 30-day window.  One figure per (month, forcing).
#
# Inputs  : data/TURB_outputs/eastern_mooring_GLEN_forced/winter<YEAR>/
#               kepsilon[_coare_wind][_UVstress]_winter<YEAR>.jld2   (Tbar, ubar, vbar, N²bar)
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv  (isothermal dates)
#               observed_mld.jld2                                     (obs)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
# Outputs : figures/TURB_eastern_mooring_UVstress_profiles_{month1,month2}_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_TURB_eastern_mooring_UVstress_profiles.jl               # direct (default)
#   julia plot_TURB_eastern_mooring_UVstress_profiles.jl coare_wind
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

# Plain-text scientific notation with Unicode superscripts (for text! annotations)
function sci_plain(x; dec = 1)
    s    = @sprintf("%.*e", dec, x)
    m, e = split(s, 'e')
    exp  = parse(Int, e)
    sup  = Dict('0'=>'⁰','1'=>'¹','2'=>'²','3'=>'³','4'=>'⁴',
                '5'=>'⁵','6'=>'⁶','7'=>'⁷','8'=>'⁸','9'=>'⁹','-'=>'⁻')
    return "$(m)×10" * join(get(sup, c, c) for c in string(exp))
end

const S_lake    = 0.05
const ρ₀        = 999.8
const ρ_air     = 1.225
const g_grav    = 9.80665
const C_D       = 1.2e-3   # neutral drag coefficient
const albedo_sw = 0.08     # broadband shortwave albedo (matches experiment)
const ε_water   = 0.98     # longwave emissivity of water (matches experiment)

# Forcing source: "direct" (measured EC fluxes) or "coare_wind".
const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"   # JLD2 / NetCDF flux suffix
const UV_TAG = "_UVstress"

const TURB_DIR = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
FIGURE_DIR     = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# k-ε output file for a given winter and variant ("" = original, "_EOScabbeling").
turb_file(year, eos_tag) = joinpath(TURB_DIR, "winter$(year)",
    "kepsilon$(SRC_TAG)$(UV_TAG)$(eos_tag)_winter$(year).jld2")

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

# Mean forcing (W/m², m²/s²) over [t_iso, t_iso + ndays], read directly from the
# gap-filled GLEN NetCDF and reduced exactly as the experiment does.
function load_glen_forcing(year; ndays = 2 * 30)
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
        (wys      = Int.(f["winter_years"]),
         dep_raw  = f["dep_raw_obs"],
         T1_raw   = f["T1_raw_obs"],
         T2_raw   = f["T2_raw_obs"],
         spd_bins = f["spd_bins_obs"],
         spd1     = f["spd1_raw_obs"],
         spd2     = f["spd2_raw_obs"])
    end
else
    @warn "EM obs not found ($OBS_EM_FILE) — run process_lake_superior_southern_eastern_moorings.jl first"
    nothing
end

winter_years = [2009, 2010, 2011, 2014, 2015]

const month_days = 30   # length of each averaging / plotting window

#####
##### Load data — the two k-ε variants
#####
# variant tag => display info; each maps year => Dict of FieldTimeSeries.
const VARIANTS = [
    (tag = "",          color = :steelblue4, label = "k-ε (TEOS-10)"),
    (tag = "_EOScabbeling", color = :darkorange, label = "k-ε (Cabbeling)"),
]

variant_data = [Dict{Int, Dict{String, Any}}() for _ in VARIANTS]
for (vi, v) in enumerate(VARIANTS)
    for year in winter_years
        f = turb_file(year, v.tag)
        isfile(f) || (@warn "Missing: $f"; continue)
        variant_data[vi][year] = Dict(
            "Tbar"  => FieldTimeSeries(f, "Tbar"),
            "ubar"  => FieldTimeSeries(f, "ubar"),
            "vbar"  => FieldTimeSeries(f, "vbar"),
            "N²bar" => FieldTimeSeries(f, "N²bar"),
        )
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
zC        = znodes(ref.grid, Center())   # Nz points   (T, u)
zF        = znodes(ref.grid, Face())     # Nz+1 points (N²)
const Lz_EM = ref.grid.Lz

# ── Mean forcing from gap-filled GLEN NetCDF + T_sfc (LW_up) correction ───────
mean_forcing = Dict{Int, NamedTuple}()
for year in winter_years
    QT1 = NaN; QU1 = NaN; U101 = NaN
    QT2 = NaN; QU2 = NaN; U102 = NaN

    # Surface temperature from the first variant that has this winter
    Tbar_fts = nothing
    for vd in variant_data
        haskey(vd, year) && (Tbar_fts = vd[year]["Tbar"]; break)
    end

    forcing = load_glen_forcing(year)
    if isnothing(forcing) || isnothing(Tbar_fts)
        @warn "No GLEN forcing for winter $year (missing date or NetCDF)"
    else
        t_forc  = forcing.t_forc
        Q_pre   = forcing.Q_pre
        tau_kin = forcing.tau_kin
        ε_w     = ε_water

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
            return ε_w * 5.67e-8 * (T_s + 273.15)^4
        end

        nanmean(v) = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))
        function window_means(t_lo, t_hi)
            mask = (t_forc .> t_lo * 86400.0) .& (t_forc .<= t_hi * 86400.0)
            any(mask) || return (NaN, NaN, NaN)
            ts   = t_forc[mask]
            QT   = nanmean(Q_pre[mask] .+ lw_up_at.(ts))
            QU   = nanmean(tau_kin[mask])
            U10  = sqrt(max(-QU * ρ₀ / (ρ_air * C_D), 0.0))
            return (QT, QU, U10)
        end

        QT1, QU1, U101 = window_means(0,          month_days)
        QT2, QU2, U102 = window_means(month_days, 2 * month_days)
    end

    mean_forcing[year] = (
        month1 = (QT_Wm2 = QT1, QU_m2s2 = QU1, U10_ms = U101),
        month2 = (QT_Wm2 = QT2, QU_m2s2 = QU2, U10_ms = U102),
    )
end

# ── Daily-averaging helpers (times taken from each field's own record) ─────────
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

# Daily mean of hourly current speed √(u²+v²) (matches the ADCP obs).
function daily_avg_speed(ufts, vfts, target_day)
    td = ufts.times ./ 86400.0
    spd(i) = sqrt.(Float64.(interior(ufts[i], 1, 1, :)) .^ 2 .+
                   Float64.(interior(vfts[i], 1, 1, :)) .^ 2)
    target_day == 0 && return spd(1)
    idxs = findall(t -> target_day <= t < target_day + 1.0, td)
    if isempty(idxs)
        _, i = findmin(abs.(td .- target_day))
        return spd(i)
    end
    return mean([spd(i) for i in idxs])
end

function day_available(dk, d)
    d == 0 && return true
    td = dk["Tbar"].times ./ 86400.0
    return any(t -> d <= t < d + 1.0, td)
end

# Conservative Temperature → in-situ temperature (matches the LES / TURB figures)
p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
Θ_to_Tinsitu(Θ_prof) = [gsw_t_from_ct(S_lake, Θ_prof[k], p_dbar[k]) for k in eachindex(Θ_prof)]

#####
##### Column (variable) configuration
#####
# kind ∈ (:temperature, :speed, :n2).  N² is stored directly (N²bar) at faces.
columns = [
    (key = :temperature, label = L"T \; (^\circ\mathrm{C})",              xlims = (0.0, 5.0), z = zC),
    (key = :speed,       label = L"|\mathbf{u}| \; (\mathrm{m\,s^{-1}})", xlims = (0.0, 0.2), z = zC),
    (key = :n2,          label = L"N^2 \; (\mathrm{s^{-2}})",             xlims = nothing,    z = zF),
]

function panel_profile(dk, kind, d)
    if kind === :temperature
        return daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu)
    elseif kind === :speed
        return daily_avg_speed(dk["ubar"], dk["vbar"], d)
    else
        return daily_avg_profile(dk["N²bar"], d, identity)
    end
end

#####
##### Plotting: one figure (rows = winter year, cols = variable) per month window
#####
function plot_TURB_profiles(filename; snapshot_days = [0, 7, 15, 22, 30],
                                      forcing_key = :month1)
    n_snaps = length(snapshot_days)
    alphas  = range(0.3, 1.0; length = n_snaps)
    obs_day = forcing_key == :month1 ? 30 : 60

    with_theme(theme_latexfonts()) do
        Nyears = length(winter_years)
        Ncols  = length(columns)

        fig = Figure(size = (230 * Ncols + 120, 175 * Nyears + 110),
                     fontsize = 10, figure_padding = (6, 10, 6, 4))

        # Column headers
        for (col, c) in enumerate(columns)
            Label(fig[1, col], c.label; fontsize = 11, font = :bold,
                  tellwidth = false, halign = :center)
        end

        axes = Matrix{Any}(undef, Nyears, Ncols)

        for (row, year) in enumerate(winter_years)
            mf = get(mean_forcing, year, nothing)
            for (col, c) in enumerate(columns)
                ax = CairoMakie.Axis(fig[row + 1, col];
                         xlabel             = row == Nyears ? c.label : "",
                         ylabel             = col == 1 ? L"z\;(\mathrm{m})" : "",
                         ylabelrotation     = π/2,
                         xticklabelsvisible = row == Nyears,
                         yticklabelsvisible = col == 1,
                         xgridvisible       = true,
                         ygridvisible       = true,
                         xticksize          = 4,
                         yticksize          = 4)
                axes[row, col] = ax

                if !any(haskey(vd, year) for vd in variant_data)
                    text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                          align = (:center, :center), fontsize = 9, color = :gray)
                    continue
                end

                # Both k-ε variants, each with the time-alpha gradient
                for (vi, v) in enumerate(VARIANTS)
                    haskey(variant_data[vi], year) || continue
                    dk = variant_data[vi][year]
                    for (si, d) in enumerate(snapshot_days)
                        day_available(dk, d) || continue
                        prof = panel_profile(dk, c.key, d)
                        lines!(ax, prof, c.z;
                               color     = (v.color, alphas[si]),
                               linewidth = si == n_snaps ? 2.5 : 1.2,
                               linestyle = si == 1 ? :dash : :solid)
                    end
                end

                ylims!(ax, (-Lz_EM - 5, 5))
                !isnothing(c.xlims) && xlims!(ax, c.xlims)

                # Forcing annotation in first column
                if col == 1
                    ann = if isnothing(mf) || isnan(mf[forcing_key].QT_Wm2)
                        "Winter $year\n(no forcing data)"
                    else
                        f = mf[forcing_key]
                        "Winter $year\n" *
                        "Q̄_T = $(round(Int, f.QT_Wm2)) W m⁻²\n" *
                        "Q̄_U = $(sci_plain(f.QU_m2s2)) m² s⁻²\n" *
                        "(U₁₀ ≈ $(round(f.U10_ms; digits=1)) m s⁻¹)"
                    end
                    text!(ax, 0.03, 0.03; text = ann, space = :relative,
                          align = (:left, :bottom), fontsize = 8)
                end
            end
        end

        # Link y-axes everywhere; x-axes within each column (variable)
        all_axes = filter(x -> x isa Axis, vec(axes))
        length(all_axes) > 1 && linkyaxes!(all_axes...)
        for col in 1:Ncols
            col_axes = filter(x -> x isa Axis, axes[:, col])
            length(col_axes) > 1 && linkxaxes!(col_axes...)
        end

        # ── Observation overlays (markers only) ──────────────────────────────
        has_obs = false
        if !isnothing(obs_em)
            obs_T   = forcing_key == :month1 ? obs_em.T1_raw : obs_em.T2_raw
            obs_spd = forcing_key == :month1 ? obs_em.spd1   : obs_em.spd2
            for (row, year) in enumerate(winter_years)
                wy_idx = findfirst(==(year), obs_em.wys)
                isnothing(wy_idx) && continue

                axT = axes[row, 1]
                if axT isa Axis
                    scatter!(axT, obs_T[wy_idx], -obs_em.dep_raw[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                    has_obs = true
                end

                axU = axes[row, 2]
                if axU isa Axis && !isnothing(obs_em.spd_bins[wy_idx]) && !isnothing(obs_spd[wy_idx])
                    scatter!(axU, obs_spd[wy_idx], -obs_em.spd_bins[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                    has_obs = true
                end
            end
        end

        # Legend: time progression (grey) + obs + variant colour key
        time_labels = ["t = $(d) days" for d in snapshot_days]
        time_labels[1] = "t = 0 (initial)"
        time_elems = Any[LineElement(color     = (:gray30, alphas[i]),
                                     linewidth = i == n_snaps ? 2.5 : 1.2,
                                     linestyle = i == 1 ? :dash : :solid)
                         for i in 1:n_snaps]
        if has_obs
            push!(time_elems,  MarkerElement(color = :firebrick, marker = :circle, markersize = 8))
            push!(time_labels, "Obs. (day $(obs_day))")
        end
        for v in VARIANTS
            push!(time_elems,  LineElement(color = v.color, linewidth = 2.0))
            push!(time_labels, v.label)
        end
        Legend(fig[Nyears + 2, 1:Ncols], time_elems, time_labels;
               orientation = :horizontal, tellwidth = false, labelsize = 9,
               framevisible = false, padding = (0, 0, 0, 0), nbanks = 1)

        Label(fig[0, 1:Ncols],
              "Eastern Mooring — TURB k-ε, GLEN-forced ($(FORCING_SOURCE); " *
              "TEOS-10 vs Cabbeling EOS; $(month_days)-day mean forcing per row)";
              fontsize = 11, font = :bold, justification = :center)

        colgap!(fig.layout, 18)   # columns have different x-scales; avoid tick collisions
        rowgap!(fig.layout, 4)

        outfile = joinpath(FIGURE_DIR, filename)
        save(outfile, fig)
        @info "Saved → $outfile"
    end
end

#####
##### Generate figures
#####
for (snap_days, fkey, suffix) in [
    ([0, 7, 15, 22, 30],   :month1, "month1"),
    ([30, 37, 45, 52, 60], :month2, "month2"),
]
    plot_TURB_profiles("TURB_eastern_mooring_EOS_profiles_$(suffix)_$(FORCING_SOURCE).pdf";
                       snapshot_days = snap_days, forcing_key = fkey)
end
