#####
##### GLEN-forced Eastern Mooring — profile plots
#####
# Rows = winter year; columns = turbulence closure.
# Each panel shows snapshots at days 0, 7, 14, 21, 28.
# Row labels include mean Q_T (W/m²) and Q_U (m²/s²) averaged over the first
# 28 days.  Q_T uses the half-hourly forcing file (Q_precomp_Wm2 + LW_up),
# where LW_up = ε·σ·T_sfc^4 is interpolated from the hourly Tbar output.
#
# Inputs  : data/TURB_outputs/eastern_mooring_GLEN_forced/winter<YEAR>/
#               <closure>[_coare_wind]_winter<YEAR>.jld2
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv   (isothermal date)
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc  (forcing)
# Outputs : figures/TURB_eastern_mooring_GLEN_forced_{T,u,N2}_profiles.pdf
#
# The mean Q_T / Q_U row labels are recomputed directly from the gap-filled GLEN
# NetCDF (sliced from each winter's isothermal date), matching the experiment:
#   Q_precomp = SHF + LHF - (1-albedo)·SW↓ - ε·LW↓   [+ LW_up(T_sfc) added below]
#   τ_kin     = -momentum_flux / ρ₀
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

# Format a number in LaTeX scientific notation with `dec` decimal places
function sci_latex(x; dec = 1)
    s    = @sprintf("%.*e", dec, x)
    m, e = split(s, 'e')
    exp  = parse(Int, e)
    return "$(m)\\!\\times\\!10^{$(exp)}"
end

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
const cₚ        = 4182.0
const ρ_air     = 1.225
const g_grav    = 9.80665
const C_D       = 1.2e-3   # neutral drag coefficient (same as idealized forcing cases)
const albedo_sw = 0.08     # broadband shortwave albedo (matches experiment)
const ε_water   = 0.98     # longwave emissivity of water (matches experiment)

# Forcing source: "direct" (measured EC fluxes) or "coare_wind".
# Pass as the first command-line argument, e.g.:
#   julia plot_TURB_eastern_mooring_GLEN_forced_profiles.jl coare_wind
# Defaults to "direct" if omitted.
const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"   # JLD2 / figure filename suffix

TURB_DIR   = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

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

# Mean forcing arrays (W/m², m²/s²) over [t_iso, t_iso + ndays], read directly from
# the gap-filled GLEN NetCDF and reduced exactly as the experiment does.
#   t_forc : seconds since t_iso ;  Q_pre : Q_precomp_Wm2 (LW_up added later)
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

# ── Observed eastern mooring T profiles ────────────────────────────────────────
const OBS_EM_FILE = joinpath(@__DIR__, "..", "figure_data",
                             "lake_superior_eastern_mooring", "observed_mld.jld2")
obs_em = if isfile(OBS_EM_FILE)
    jldopen(OBS_EM_FILE) do f
        (wys     = Int.(f["winter_years"]),
         dep     = Float64.(f["dep_obs"]),
         T1      = Float64.(f["T1_profiles"]),
         T2      = Float64.(f["T2_profiles"]),
         dep_raw = f["dep_raw_obs"],
         T1_raw  = f["T1_raw_obs"],
         T2_raw  = f["T2_raw_obs"],
         spd_bins = haskey(f, "spd_bins_obs") ? f["spd_bins_obs"] : nothing,
         spd1     = haskey(f, "spd1_raw_obs") ? f["spd1_raw_obs"] : nothing,
         spd2     = haskey(f, "spd2_raw_obs") ? f["spd2_raw_obs"] : nothing)
    end
else
    @warn "Eastern mooring obs not found — run process_lake_superior_southern_eastern_moorings.jl first"
    nothing
end

winter_years = [2009, 2010, 2011, 2014, 2015]

closure_models = [
    ("kepsilon",         L"k\text{-}\varepsilon"),
    ("kepsilon_Rist035", L"k\text{-}\varepsilon\;(Ri_{st}{=}0.35)"),
    ("CATKE",            "CATKE"),
    ("CATKE_highRi",     L"\mathrm{CATKE}\;(\mathrm{highRi})"),
]

const month_days = 30   # length of each averaging / plotting window

#####
##### Load data
#####

data         = Dict{Tuple{Int,String}, Dict{String, Any}}()
mean_forcing = Dict{Int, NamedTuple}()

# ── Loop 1: load profile data ─────────────────────────────────────────────────
for year in winter_years
    year_dir = joinpath(TURB_DIR, "winter$(year)")
    for (cname, _) in closure_models
        ts_file = joinpath(year_dir, "$(cname)$(SRC_TAG)_winter$(year).jld2")
        isfile(ts_file) || (@warn "Missing: $ts_file"; continue)
        Tbar  = FieldTimeSeries(ts_file, "Tbar")
        ubar  = FieldTimeSeries(ts_file, "ubar")
        vbar  = FieldTimeSeries(ts_file, "vbar")
        N²bar = FieldTimeSeries(ts_file, "N²bar")
        data[(year, cname)] = Dict("Tbar" => Tbar, "ubar" => ubar, "vbar" => vbar, "N²bar" => N²bar)
    end
end

# ── Grid geometry ──────────────────────────────────────────────────────────────
ref        = first(values(data))["Tbar"]
zC         = znodes(ref.grid, Center())   # Nz points  (for T, u)
zF         = znodes(ref.grid, Face())     # Nz+1 points (for N²)
times_days = ref.times ./ 86400.0
Nz_grid    = length(zC)

# ── Loop 2: mean forcing from gap-filled GLEN NetCDF + T_sfc correction ───────
# Q_net = Q_precomp_Wm2 (SHF + LHF - SW_net - LW_down) + LW_up (ε·σ·T_sfc^4)
# T_sfc is interpolated from the hourly Tbar output at the surface cell.
for year in winter_years
    QT1 = NaN; QU1 = NaN; U101 = NaN
    QT2 = NaN; QU2 = NaN; U102 = NaN

    forcing = load_glen_forcing(year)
    if isnothing(forcing)
        @warn "No GLEN forcing for winter $year (missing isothermal date or NetCDF)"
    else
        t_forc  = forcing.t_forc
        Q_pre   = forcing.Q_pre
        tau_kin = forcing.tau_kin
        ε_w     = ε_water

        # Surface temperature from first available closure's hourly Tbar
        T_sfc_times = nothing; T_sfc_vals = nothing
        for (cname, _) in closure_models
            haskey(data, (year, cname)) || continue
            Tbar_fts    = data[(year, cname)]["Tbar"]
            T_sfc_times = Float64.(Tbar_fts.times)
            T_sfc_vals  = [Float64(interior(Tbar_fts[i])[1, 1, end])
                           for i in eachindex(Tbar_fts.times)]
            break
        end

        function lw_up_at(t_s)
            T_s = if isnothing(T_sfc_vals)
                4.0
            else
                i = searchsortedlast(T_sfc_times, t_s)
                if i == 0
                    T_sfc_vals[1]
                elseif i >= length(T_sfc_vals)
                    T_sfc_vals[end]
                else
                    α = (t_s - T_sfc_times[i]) / (T_sfc_times[i+1] - T_sfc_times[i])
                    T_sfc_vals[i] + α * (T_sfc_vals[i+1] - T_sfc_vals[i])
                end
            end
            return ε_w * 5.67e-8 * (T_s + 273.15)^4
        end

        nanmean(v) = (w = filter(!isnan, v); isempty(w) ? NaN : mean(w))
        function window_means(t_lo, t_hi)
            mask = (t_forc .> t_lo * 86400.0) .& (t_forc .<= t_hi * 86400.0)
            any(mask) || return (NaN, NaN, NaN)
            ts   = t_forc[mask]
            QT   = nanmean(Q_pre[mask] .+ lw_up_at.(ts))   # skip any residual GLEN gaps
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

# For t = 0: return the first saved profile.
# For t = d > 0: average all hourly profiles in [d, d+1) days.
function daily_avg_profile(fts, target_day, transform)
    if target_day == 0
        return transform(Float64.(interior(fts[1], 1, 1, :)))
    end
    idxs = findall(t -> target_day <= t < target_day + 1.0, times_days)
    if isempty(idxs)
        _, i = findmin(abs.(times_days .- target_day))
        return transform(Float64.(interior(fts[i], 1, 1, :)))
    end
    return mean([transform(Float64.(interior(fts[i], 1, 1, :))) for i in idxs])
end

# Daily mean of hourly current speed √(u²+v²).  Computed as the average of the
# hourly speed (not the speed of the daily-mean velocity) so that it matches the
# ADCP observations, which average instantaneous speed; the two differ whenever
# the current rotates/oscillates within the day (e.g. near-inertial motions).
function daily_avg_speed(ufts, vfts, target_day)
    spd(i) = sqrt.(Float64.(interior(ufts[i], 1, 1, :)) .^ 2 .+
                   Float64.(interior(vfts[i], 1, 1, :)) .^ 2)
    target_day == 0 && return spd(1)
    idxs = findall(t -> target_day <= t < target_day + 1.0, times_days)
    if isempty(idxs)
        _, i = findmin(abs.(times_days .- target_day))
        return spd(i)
    end
    return mean([spd(i) for i in idxs])
end

# Conservative Temperature → in-situ temperature
p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
Θ_to_Tinsitu(Θ_prof) = [gsw_t_from_ct(S_lake, Θ_prof[k], p_dbar[k]) for k in eachindex(Θ_prof)]

#####
##### Plotting function
#####

function plot_profiles_GLEN(field_name, xlabel_str, filename;
                            snapshot_days     = [0, 7, 15, 22, 30],
                            forcing_key       = :month1,
                            profile_transform = identity,
                            xlims_range       = nothing,
                            z_nodes           = zC)

    n_snaps = length(snapshot_days)
    alphas  = range(0.3, 1.0; length = n_snaps)

    with_theme(theme_latexfonts()) do
        Nyears    = length(winter_years)
        Nclosures = length(closure_models)

        fig = Figure(size = (210 * Nclosures + 140, 175 * Nyears + 100),
                     fontsize = 10, figure_padding = (6, 10, 6, 4))

        # Column headers
        for (col, (_, clabel)) in enumerate(closure_models)
            Label(fig[1, col], clabel;
                  fontsize = 10, font = :bold, tellwidth = false, halign = :center)
        end

        axes = Matrix{Any}(undef, Nyears, Nclosures)

        for (row, year) in enumerate(winter_years)
            mf = mean_forcing[year][forcing_key]

            for (col, (cname, _)) in enumerate(closure_models)
                ax = CairoMakie.Axis(fig[row + 1, col];
                         xlabel             = row == Nyears ? xlabel_str : "",
                         ylabel             = col == 1 ? L"z\;(\mathrm{m})" : "",
                         ylabelrotation     = π/2,
                         xticklabelsvisible = row == Nyears,
                         yticklabelsvisible = col == 1,
                         xgridvisible       = true,
                         ygridvisible       = true,
                         xticksize          = 4,
                         yticksize          = 4)
                axes[row, col] = ax

                key = (year, cname)
                if !haskey(data, key)
                    text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                          align = (:center, :center), fontsize = 9, color = :gray)
                    continue
                end

                for (si, d) in enumerate(snapshot_days)
                    prof = field_name == "ubar" ?
                        daily_avg_speed(data[key]["ubar"], data[key]["vbar"], d) :
                        daily_avg_profile(data[key][field_name], d, profile_transform)
                    lines!(ax, prof, z_nodes;
                           color     = (:steelblue4, alphas[si]),
                           linewidth = si == n_snaps ? 2.5 : 1.2,
                           linestyle = si == 1 ? :dash : :solid)
                end

                ylims!(ax, (-215, 5))
                !isnothing(xlims_range) && xlims!(ax, xlims_range)
                col > 1 && hideydecorations!(ax; ticks = false, grid = false)
                row < Nyears && hidexdecorations!(ax; ticks = false, grid = false)

                # Forcing annotation in lower-left of first-column panels
                if col == 1
                    ann = if isnan(mf.QT_Wm2)
                        "Winter $year\n(no forcing data)"
                    else
                        "Winter $year\n" *
                        "Q̄_T = $(round(Int, mf.QT_Wm2)) W m⁻²\n" *
                        "Q̄_U = $(sci_plain(mf.QU_m2s2)) m² s⁻²\n" *
                        "(U₁₀ ≈ $(round(mf.U10_ms; digits=1)) m s⁻¹)"
                    end
                    text!(ax, 0.03, 0.03; text = ann, space = :relative,
                          align = (:left, :bottom), fontsize = 8)
                end
            end
        end

        # Link all axes globally so ticks and gridlines are consistent across rows and columns
        all_axes = filter(x -> x isa Axis, vec(axes))
        length(all_axes) > 1 && linkxaxes!(all_axes...)
        length(all_axes) > 1 && linkyaxes!(all_axes...)

        # Observed T profile overlay (day 30 for month1, day 60 for month2)
        has_obs_T = field_name == "Tbar" && !isnothing(obs_em)
        if has_obs_T
            obs_T     = forcing_key == :month1 ? obs_em.T1     : obs_em.T2
            obs_T_raw = forcing_key == :month1 ? obs_em.T1_raw : obs_em.T2_raw
            z_obs_em  = -obs_em.dep
            for (row, year) in enumerate(winter_years)
                wy_idx = findfirst(==(year), obs_em.wys)
                isnothing(wy_idx) && continue
                T_raw = obs_T_raw[wy_idx]
                z_raw = -obs_em.dep_raw[wy_idx]
                for col in 1:Nclosures
                    ax = axes[row, col]
                    ax isa Axis || continue
                    scatter!(ax, T_raw, z_raw;
                             color = :firebrick, marker = :circle, markersize = 8)
                end
            end
        end

        # Observed total current speed overlay from ADCP
        has_obs_u = field_name == "ubar" && !isnothing(obs_em) &&
                    !isnothing(obs_em.spd_bins) && !isnothing(obs_em.spd1)
        if has_obs_u
            obs_spd = forcing_key == :month1 ? obs_em.spd1 : obs_em.spd2
            for (row, year) in enumerate(winter_years)
                wy_idx = findfirst(==(year), obs_em.wys)
                isnothing(wy_idx) && continue
                spd_bins = obs_em.spd_bins[wy_idx]
                spd_data = obs_spd[wy_idx]
                (isnothing(spd_bins) || isnothing(spd_data)) && continue
                z_bins = -spd_bins
                for col in 1:Nclosures
                    ax = axes[row, col]
                    ax isa Axis || continue
                    scatter!(ax, spd_data, z_bins;
                             color = :firebrick, marker = :circle, markersize = 8)
                end
            end
        end

        has_obs = has_obs_T || has_obs_u

        # Legend
        time_labels = ["t = $(d) days" for d in snapshot_days]
        time_labels[1] = "t = 0 (initial)"
        time_elems = Any[LineElement(color     = (:steelblue4, alphas[i]),
                                     linewidth = i == n_snaps ? 2.5 : 1.2,
                                     linestyle = i == 1 ? :dash : :solid)
                         for i in 1:n_snaps]
        if has_obs
            obs_day = forcing_key == :month1 ? 30 : 60
            push!(time_elems,  MarkerElement(color = :firebrick, marker = :circle, markersize = 8))
            push!(time_labels, "Obs. (day $(obs_day))")
        end
        Legend(fig[Nyears + 2, 1:Nclosures], time_elems, time_labels;
               orientation = :horizontal, tellwidth = false, labelsize = 9,
               framevisible = false, padding = (0, 0, 0, 0))

        Label(fig[0, 1:Nclosures],
              "Eastern Mooring — GLEN-forced ($(month_days)-day mean forcing shown per row)";
              fontsize = 11, font = :bold, justification = :center)

        colgap!(fig.layout, 6)
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
    plot_profiles_GLEN("Tbar",
                       L"T \; (^\circ\mathrm{C})",
                       "TURB_eastern_mooring_GLEN_forced_T_profiles_$(suffix)_$(FORCING_SOURCE).pdf";
                       snapshot_days = snap_days, forcing_key = fkey,
                       profile_transform = Θ_to_Tinsitu,
                       xlims_range = (0.0, 5.0),
                       z_nodes = zC)

    plot_profiles_GLEN("ubar",
                       L"|\mathbf{u}| \; (\mathrm{m\,s^{-1}})",
                       "TURB_eastern_mooring_GLEN_forced_speed_profiles_$(suffix)_$(FORCING_SOURCE).pdf";
                       snapshot_days = snap_days, forcing_key = fkey,
                       xlims_range = (0.0, 0.2),
                       z_nodes = zC)

    plot_profiles_GLEN("N²bar",
                       L"N^2 \; (\mathrm{s^{-2}})",
                       "TURB_eastern_mooring_GLEN_forced_N2_profiles_$(suffix)_$(FORCING_SOURCE).pdf";
                       snapshot_days = snap_days, forcing_key = fkey,
                       z_nodes = zF)
end
