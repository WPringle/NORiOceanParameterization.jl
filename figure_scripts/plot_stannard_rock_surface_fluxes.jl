#####
##### Stannard Rock surface flux timeseries during fall transition periods.
#####
# For each winter season identified in the Eastern mooring analysis,
# the fall transition window is defined as:
#   [isothermal 4°C date  →  isothermal date + 60 days]
# The script plots latent heat flux, sensible heat flux, downwelling shortwave,
# downwelling longwave, and wind speed from the Stannard Rock GLEN netCDF file.
# One figure panel row per season; columns = flux variables.
#
# Inputs  : figure_data/lake_superior_eastern_mooring/observed_mld.jld2
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc.nc
# Outputs : figures/stannard_rock_fluxes_eastern_mooring_winters.pdf
#####

using NCDatasets
using JLD2
using Dates
using CairoMakie
using LaTeXStrings
using Statistics

const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc.nc"

const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# ── Load Stannard Rock data ────────────────────────────────────────────────────
glen_times, lhf, shf, sw, lw, wspd = NCDataset(GLEN_FILE) do ds
    # NCDatasets decodes CF time to DateTime automatically
    times = DateTime.(ds["time"][:])

    # Replace missing sentinels with missing
    load_var(name) = begin
        v = ds[name][:]
        [ismissing(x) ? missing : Float64(x) for x in v]
    end

    lhf  = load_var("latent_heat_flux")
    shf  = load_var("sensible_heat_flux")
    sw   = load_var("downwelling_shortwave_flux")
    lw   = load_var("downwelling_longwave_flux")
    wspd = load_var("wind_speed")

    times, lhf, shf, sw, lw, wspd
end
@info "Loaded Stannard Rock: $(glen_times[1]) to $(glen_times[end]) ($(length(glen_times)) records)"

# ── Define flux panels ─────────────────────────────────────────────────────────
flux_panels = [
    ("Latent heat flux",            lhf,  L"LHF \; (W\,m^{-2})"),
    ("Sensible heat flux",          shf,  L"SHF \; (W\,m^{-2})"),
    ("Downwelling shortwave",       sw,   L"SW_\downarrow \; (W\,m^{-2})"),
    ("Downwelling longwave",        lw,   L"LW_\downarrow \; (W\,m^{-2})"),
    ("Wind speed",                  wspd, L"U \; (m\,s^{-1})"),
]
Nvar = length(flux_panels)

# ── Helper: indices of glen_times within [t_start, t_end] ─────────────────────
function window_indices(t_start, t_end)
    findall(t -> t_start <= t <= t_end, glen_times)
end

# ── Daily-average helper ───────────────────────────────────────────────────────
function daily_average(times_sub, data_sub)
    # Group by Date, return (dates, means)
    day_map = Dict{Date, Vector{Float64}}()
    for (t, v) in zip(times_sub, data_sub)
        ismissing(v) && continue
        d = Date(t)
        push!(get!(day_map, d, Float64[]), v)
    end
    days  = sort(collect(keys(day_map)))
    means = [mean(day_map[d]) for d in days]
    return days, means
end

# ── Make one figure for a set of winter seasons ───────────────────────────────
function plot_flux_figure(winter_years, start_dates_str, fig_stem, mooring_name)
    Nseasons = length(winter_years)
    start_dates = DateTime.(start_dates_str)

    fig = Figure(size = (220 * Nvar, 180 * Nseasons + 60),
                 fontsize = 10, figure_padding = (6, 10, 6, 4))

    # Column headers (flux variable names)
    for (col, (label, _, _)) in enumerate(flux_panels)
        Label(fig[1, col], label; fontsize = 10, font = :bold,
              tellwidth = false, halign = :center)
    end

    row_axes = Vector{Vector{Axis}}(undef, Nseasons)

    for (row, (wy, t0)) in enumerate(zip(winter_years, start_dates))
        t_end = t0 + Day(60)
        idx   = window_indices(t0, t_end)

        iso_str = Dates.format(Date(t0), "d u Y")
        row_axes[row] = Axis[]
        for (col, (_, data, ylabel)) in enumerate(flux_panels)
            ax = CairoMakie.Axis(fig[row + 1, col];
                     ylabel    = col == 1 ? "Winter $(wy)\n(iso. $(iso_str))\n" * ylabel : ylabel,
                     xlabel    = row == Nseasons ? "Date" : "",
                     xticklabelsvisible = row == Nseasons,
                     yticklabelsvisible = true)
            push!(row_axes[row], ax)

            if isempty(idx)
                text!(ax, 0.5, 0.5; text = "no data", space = :relative,
                      align = (:center, :center), fontsize = 9)
                continue
            end

            t_sub = glen_times[idx]
            d_sub = data[idx]

            # Half-hourly scatter (faint)
            t_float = Dates.value.(t_sub)   # milliseconds since epoch (Int64)
            t_days  = (t_float .- Dates.value(t0)) ./ (1000.0 * 86400.0)
            valid   = [!ismissing(v) for v in d_sub]
            if any(valid)
                scatter!(ax, t_days[valid], Float64.(d_sub[valid]),
                         color = (:steelblue, 0.2), markersize = 2)
            end

            # Daily means
            days_sub, means_sub = daily_average(t_sub, d_sub)
            if !isempty(days_sub)
                d_days = Dates.value.(days_sub .- Date(t0)) ./ 1.0
                lines!(ax, d_days, means_sub,
                       color = :steelblue, linewidth = 1.8)
            end

            # Mark t0 and t0+30d
            vlines!(ax, [0.0];  color = (:black, 0.6), linewidth = 1.2, linestyle = :dash)
            vlines!(ax, [30.0]; color = (:red,   0.6), linewidth = 1.2, linestyle = :dot)
            vlines!(ax, [60.0]; color = (:red,   0.6), linewidth = 1.2, linestyle = :dot)

            ax.xlabel = row == Nseasons ? "Days from isothermal date" : ""
        end

        # Link x-axes across columns for same row
        linkxaxes!(row_axes[row]...)
    end

    # Global y-axis links across rows (same variable)
    for col in 1:Nvar
        col_axes = [row_axes[row][col] for row in 1:Nseasons
                    if length(row_axes[row]) >= col]
        length(col_axes) > 1 && linkyaxes!(col_axes...)
    end

    Label(fig[0, :],
          "Stannard Rock — surface fluxes during fall transition\n($(mooring_name) isothermal dates)",
          fontsize = 12, font = :bold, justification = :center)

    # Legend
    legend_elems = [
        MarkerElement(color = (:steelblue, 0.4), marker = :circle, markersize = 6),
        LineElement(color = :steelblue, linewidth = 1.8),
        LineElement(color = (:black, 0.6), linewidth = 1.2, linestyle = :dash),
        LineElement(color = (:red,   0.6), linewidth = 1.2, linestyle = :dot),
    ]
    legend_labels = ["Half-hourly", "Daily mean", "t = 0 (isothermal)", "+30 / +60 days"]
    Legend(fig[Nseasons + 2, :], legend_elems, legend_labels,
           orientation = :horizontal, tellwidth = false, labelsize = 9,
           framevisible = false, padding = (0, 0, 0, 0))

    colgap!(fig.layout, 6)
    rowgap!(fig.layout, 4)

    outfig = joinpath(FIGURE_DIR, "stannard_rock_fluxes_$(fig_stem)_winters.pdf")
    save(outfig, fig)
    @info "Saved → $outfig"
end

# ── Load mooring results and make figure ────────────────────────────────────────
let (jld2_subdir, mooring_name, fig_stem) = ("lake_superior_eastern_mooring", "Eastern Mooring", "eastern_mooring")
    jld2_file = joinpath(@__DIR__, "..", "figure_data", jld2_subdir, "observed_mld.jld2")
    if !isfile(jld2_file)
        @warn "JLD2 not found: $jld2_file — run process_lake_superior_southern_eastern_moorings.jl first"
    else
        wys, sdates = jldopen(jld2_file) do f
            f["winter_years"], f["start_dates"]
        end
        @info "$(mooring_name): $(length(wys)) winter seasons: $wys"
        plot_flux_figure(wys, sdates, fig_stem, mooring_name)
    end
end
