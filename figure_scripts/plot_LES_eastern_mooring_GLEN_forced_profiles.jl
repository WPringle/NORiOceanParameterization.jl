#####
##### GLEN-forced Eastern Mooring — LES profile plots
#####
# One figure per (month-window, forcing source).  Layout: rows = winter year,
# columns = variable (T, |u|, N²).  Each panel shows profile snapshots at days
# 0, 7, 15, 22, 30 (month 1) or 30, 37, 45, 52, 60 (month 2).  First-column rows
# are labelled with the mean Q_T (W/m²) and Q_U (m²/s²) over the 30-day window.
#
# Unlike the TURB comparison (which had a closure dimension), the LES is a single
# run per winter+forcing, so the closure columns are replaced by the three
# variables.  Model speed is the daily mean of hourly √(ubar²+vbar²) and N² is
# the daily mean of hourly ∂bbar/∂z — both matching how the observations / TURB
# figures are formed.
#
# Inputs  : data/LES_outputs/eastern_mooring_GLEN/
#               LES_GLEN_winter<YEAR>_<forcing>_Lxy256_Lz212_Nxy128_Nz106/
#                   hourly_averaged_timeseries.jld2      (Tbar, ubar, vbar, bbar)
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates_LES.csv  (old MLD dates)
#               observed_mld_LES.jld2                    (obs at the old dates)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
# Outputs : figures/LES_eastern_mooring_GLEN_forced_profiles_{month1,month2}_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_LES_eastern_mooring_GLEN_forced_profiles.jl               # direct (default)
#   julia plot_LES_eastern_mooring_GLEN_forced_profiles.jl coare_wind
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
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"   # GLEN NetCDF flux suffix

const LES_DIR    = joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN")
const LES_STEM   = "Lxy256_Lz212_Nxy128_Nz106"
FIGURE_DIR       = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

les_file(year) = joinpath(LES_DIR,
    "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_$(LES_STEM)", "hourly_averaged_timeseries.jld2")

# ── GLEN forcing file + old MLD (winter start) dates ──────────────────────────
const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"
const CSV_FILE  = joinpath(@__DIR__, "..", "figure_data",
                          "lake_superior_eastern_mooring",
                          "lake_superior_eastern_mooring_winter_start_dates_LES.csv")

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

# ── Observed eastern mooring profiles at the OLD (LES) start dates ─────────────
const OBS_EM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_eastern_mooring", "observed_mld_LES.jld2")
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
    @warn "LES obs not found ($OBS_EM_FILE) — run process_lake_superior_southern_eastern_moorings.jl first"
    nothing
end

winter_years = [2009, 2011, 2014, 2015]

const month_days = 30   # length of each averaging / plotting window

#####
##### Load data
#####

data         = Dict{Int, Dict{String, Any}}()
mean_forcing = Dict{Int, NamedTuple}()

for year in winter_years
    f = les_file(year)
    isfile(f) || (@warn "Missing LES output: $f"; continue)
    data[year] = Dict(
        "Tbar" => FieldTimeSeries(f, "Tbar"),
        "ubar" => FieldTimeSeries(f, "ubar"),
        "vbar" => FieldTimeSeries(f, "vbar"),
        "bbar" => FieldTimeSeries(f, "bbar"),
    )
end
isempty(data) && error("No LES outputs found for FORCING_SOURCE=$(FORCING_SOURCE)")

# Optional second LES variant overlaid (different colour) on the same panels.
# The variant depends on the forcing source:
#   coare_wind → Smagorinsky–Lilly closure (dir suffix "SmagorinskyLilly")
#   direct     → UV-decomposed wind stress, i.e. momentum split between u and v
#                by wind direction rather than all placed on u (dir suffix "UVstress")
const OVERLAY_SUFFIX, OVERLAY_LABEL, OVERLAY_COLOR =
    FORCING_SOURCE == "coare_wind" ? ("SmagorinskyLilly", "Smag.–Lilly", :seagreen)   :
    FORCING_SOURCE == "direct"     ? ("UVstress",         "UV stress",   :darkorange) :
    ("", "", :seagreen)

overlay = Dict{Int, Dict{String, Any}}()
if OVERLAY_SUFFIX != ""
    for year in winter_years
        f = joinpath(LES_DIR,
            "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_$(OVERLAY_SUFFIX)_$(LES_STEM)",
            "hourly_averaged_timeseries.jld2")
        isfile(f) || continue
        overlay[year] = Dict(
            "Tbar" => FieldTimeSeries(f, "Tbar"),
            "ubar" => FieldTimeSeries(f, "ubar"),
            "vbar" => FieldTimeSeries(f, "vbar"),
            "bbar" => FieldTimeSeries(f, "bbar"),
        )
    end
end
!isempty(overlay) && @info "$(OVERLAY_LABEL) runs found for winters: $(sort(collect(keys(overlay))))"

# ── Grid geometry ──────────────────────────────────────────────────────────────
ref        = first(values(data))["Tbar"]
zC         = znodes(ref.grid, Center())   # Nz points  (T, u, and N² at centers)
zF         = znodes(ref.grid, Face())     # Nz+1 points (bbar location)
times_days = ref.times ./ 86400.0
const Lz_LES = ref.grid.Lz

# ── Mean forcing from gap-filled GLEN NetCDF + T_sfc (LW_up) correction ───────
for year in winter_years
    QT1 = NaN; QU1 = NaN; U101 = NaN
    QT2 = NaN; QU2 = NaN; U102 = NaN

    forcing = load_glen_forcing(year)
    if isnothing(forcing) || !haskey(data, year)
        @warn "No GLEN forcing for winter $year (missing date or NetCDF)"
    else
        t_forc  = forcing.t_forc
        Q_pre   = forcing.Q_pre
        tau_kin = forcing.tau_kin
        ε_w     = ε_water

        Tbar_fts    = data[year]["Tbar"]
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

# ── Daily-averaging helpers ──────────────────────────────────────────────────
# Times are taken from each field's own record, so the same helpers work for the
# base LES and the shorter Smagorinsky–Lilly runs.
# For t = 0: the first saved profile.  For t = d > 0: average all hourly profiles
# in [d, d+1) days (nearest snapshot if that window is empty).
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

# Daily mean of hourly current speed √(u²+v²) (matches the ADCP obs, which
# average instantaneous speed rather than the speed of the daily-mean velocity).
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

# Daily mean of hourly N² = ∂bbar/∂z.  bbar is stored at faces (Nz+1); the
# gradient lands at cell centers (Nz).  The top/bottom boundary faces hold
# placeholder zeros, so those two centered values are NaN'd.
function daily_avg_N2(bfts, target_day)
    td = bfts.times ./ 86400.0
    n2(i) = begin
        b = Float64.(interior(bfts[i], 1, 1, :))
        v = diff(b) ./ diff(zF)
        v[1] = NaN; v[end] = NaN
        v
    end
    target_day == 0 && return n2(1)
    idxs = findall(t -> target_day <= t < target_day + 1.0, td)
    if isempty(idxs)
        _, i = findmin(abs.(td .- target_day))
        return n2(i)
    end
    return mean([n2(i) for i in idxs])
end

# Whether snapshot day `d` is actually covered by a run (avoid extrapolating the
# shorter Smag runs by snapping to their last available time).
function day_available(dk, d)
    d == 0 && return true
    td = dk["Tbar"].times ./ 86400.0
    return any(t -> d <= t < d + 1.0, td)
end

# Conservative Temperature → in-situ temperature (matches the TURB EM figure)
p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
Θ_to_Tinsitu(Θ_prof) = [gsw_t_from_ct(S_lake, Θ_prof[k], p_dbar[k]) for k in eachindex(Θ_prof)]

#####
##### Column (variable) configuration
#####
# kind ∈ (:temperature, :speed, :n2); each panel shows the snapshot profiles.
columns = [
    (key = :temperature, label = L"T \; (^\circ\mathrm{C})",          xlims = (0.0, 5.0)),
    (key = :speed,       label = L"|\mathbf{u}| \; (\mathrm{m\,s^{-1}})", xlims = (0.0, 0.2)),
    (key = :n2,          label = L"N^2 \; (\mathrm{s^{-2}})",         xlims = nothing),
]

function panel_profile(dk, kind, d)
    if kind === :temperature
        return daily_avg_profile(dk["Tbar"], d, Θ_to_Tinsitu)
    elseif kind === :speed
        return daily_avg_speed(dk["ubar"], dk["vbar"], d)
    else
        return daily_avg_N2(dk["bbar"], d)
    end
end

#####
##### Plotting: one figure (rows = winter year, cols = variable) per month window
#####
function plot_LES_profiles(filename; snapshot_days = [0, 7, 15, 22, 30],
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

                if !haskey(data, year)
                    text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                          align = (:center, :center), fontsize = 9, color = :gray)
                    continue
                end

                for (si, d) in enumerate(snapshot_days)
                    prof = panel_profile(data[year], c.key, d)
                    lines!(ax, prof, zC;
                           color     = (:steelblue4, alphas[si]),
                           linewidth = si == n_snaps ? 2.5 : 1.2,
                           linestyle = si == 1 ? :dash : :solid)
                end

                # Second LES variant overlay (Smag.–Lilly or UV stress), available days only
                if haskey(overlay, year)
                    for (si, d) in enumerate(snapshot_days)
                        day_available(overlay[year], d) || continue
                        prof = panel_profile(overlay[year], c.key, d)
                        lines!(ax, prof, zC;
                               color     = (OVERLAY_COLOR, alphas[si]),
                               linewidth = si == n_snaps ? 2.5 : 1.2,
                               linestyle = si == 1 ? :dash : :solid)
                    end
                end

                ylims!(ax, (-Lz_LES - 5, 5))
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

                # Temperature markers (column 1)
                axT = axes[row, 1]
                if axT isa Axis
                    T_raw = obs_T[wy_idx]; z_raw = -obs_em.dep_raw[wy_idx]
                    scatter!(axT, T_raw, z_raw; color = :firebrick, marker = :circle, markersize = 8)
                    has_obs = true
                end

                # Speed markers (column 2)
                axU = axes[row, 2]
                if axU isa Axis && !isnothing(obs_em.spd_bins[wy_idx]) && !isnothing(obs_spd[wy_idx])
                    scatter!(axU, obs_spd[wy_idx], -obs_em.spd_bins[wy_idx];
                             color = :firebrick, marker = :circle, markersize = 8)
                    has_obs = true
                end
            end
        end

        # Legend
        time_labels = ["t = $(d) days" for d in snapshot_days]
        time_labels[1] = "t = 0 (initial)"
        time_elems = Any[LineElement(color     = (:steelblue4, alphas[i]),
                                     linewidth = i == n_snaps ? 2.5 : 1.2,
                                     linestyle = i == 1 ? :dash : :solid)
                         for i in 1:n_snaps]
        if has_obs
            push!(time_elems,  MarkerElement(color = :firebrick, marker = :circle, markersize = 8))
            push!(time_labels, "Obs. (day $(obs_day))")
        end
        if !isempty(overlay)
            # Colour key for the two LES variants (time is encoded by line alpha above)
            push!(time_elems,  LineElement(color = :steelblue4, linewidth = 2.0))
            push!(time_labels, "LES")
            push!(time_elems,  LineElement(color = OVERLAY_COLOR, linewidth = 2.0))
            push!(time_labels, OVERLAY_LABEL)
        end
        Legend(fig[Nyears + 2, 1:Ncols], time_elems, time_labels;
               orientation = :horizontal, tellwidth = false, labelsize = 9,
               framevisible = false, padding = (0, 0, 0, 0))

        Label(fig[0, 1:Ncols],
              "Eastern Mooring — LES, GLEN-forced ($(FORCING_SOURCE); " *
              "$(month_days)-day mean forcing per row)";
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
    plot_LES_profiles("LES_eastern_mooring_GLEN_forced_profiles_$(suffix)_$(FORCING_SOURCE).pdf";
                      snapshot_days = snap_days, forcing_key = fkey)
end
