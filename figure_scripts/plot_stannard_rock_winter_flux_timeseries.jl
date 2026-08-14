#####
##### Stannard Rock — winter surface-flux diagnostics
#####
# For each winter season (Dec 01 → Mar 15), plots half-hourly time series of:
#   wind speed, wind direction, momentum flux, sensible heat flux, latent heat flux
# from the Stannard Rock GLEN netCDF. The three flux panels overlay both the
# "direct" (measured eddy-covariance) and "coare_wind" (COARE bulk algorithm
# driven by wind speed) estimates. Vertical dotted lines mark the isothermal
# winter-start dates of the Eastern mooring (2009, 2010, 2011, 2014, 2015) and,
# where they exist, the Southern mooring (2010, 2011 only). One figure per
# winter season.
#
# Inputs  : /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc
#           figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv
#           figure_data/lake_superior_southern_mooring/
#               lake_superior_southern_mooring_winter_start_dates.csv
# Outputs : figures/stannard_rock_winter_flux_timeseries_winter<YEAR>.pdf
#####

using NCDatasets
using Dates
using CairoMakie
using LaTeXStrings

const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"

const CSV_EASTERN  = joinpath(@__DIR__, "..", "figure_data",
                               "lake_superior_eastern_mooring",
                               "lake_superior_eastern_mooring_winter_start_dates.csv")
const CSV_SOUTHERN = joinpath(@__DIR__, "..", "figure_data",
                               "lake_superior_southern_mooring",
                               "lake_superior_southern_mooring_winter_start_dates.csv")

const FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# Winter seasons to plot: labeled by the calendar year in which they end
# (season = [Dec 1, year-1] → [Mar 15, year]) — matches the Eastern mooring's
# isothermal winter years.
const winter_years = [2009, 2010, 2011, 2014, 2015]

# ── Isothermal (winter-start) dates ────────────────────────────────────────────
function read_isothermal_dates(csv_file)
    dates = Dict{Int, DateTime}()
    isfile(csv_file) || return dates
    open(csv_file) do f
        readline(f)   # skip header
        for line in eachline(f)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) >= 2 || continue
            wy = tryparse(Int, strip(parts[1]))
            isnothing(wy) && continue
            dates[wy] = DateTime(Date(strip(parts[2])))
        end
    end
    return dates
end

const eastern_iso  = read_isothermal_dates(CSV_EASTERN)
const southern_iso = read_isothermal_dates(CSV_SOUTHERN)

# ── Load Stannard Rock data (once) ─────────────────────────────────────────────
glen_times, wspd, wdir, mom, mom_c, shf, shf_c, lhf, lhf_c = NCDataset(GLEN_FILE) do ds
    times = DateTime.(ds["time"][:])
    getv(name) = Float64[ismissing(x) ? NaN : Float64(x) for x in ds[name][:]]
    (times,
     getv("wind_speed"), getv("wind_direction"),
     getv("momentum_flux"),      getv("momentum_flux_coare_wind"),
     getv("sensible_heat_flux"), getv("sensible_heat_flux_coare_wind"),
     getv("latent_heat_flux"),   getv("latent_heat_flux_coare_wind"))
end
@info "Loaded Stannard Rock: $(glen_times[1]) to $(glen_times[end]) ($(length(glen_times)) records)"

# ── Panel definitions ──────────────────────────────────────────────────────────
# (label, ylabel, direct series, coare_wind series or `nothing`)
panels = [
    ("Wind speed",       L"U \; (\mathrm{m\,s^{-1}})",   wspd, nothing),
    ("Wind direction",   L"\theta \; (^\circ)",          wdir, nothing),
    ("Momentum flux",    L"\tau \; (\mathrm{N\,m^{-2}})", mom,  mom_c),
    ("Sensible heat flux", L"SHF \; (\mathrm{W\,m^{-2}})", shf,  shf_c),
    ("Latent heat flux",   L"LHF \; (\mathrm{W\,m^{-2}})", lhf,  lhf_c),
]
Nvar = length(panels)

const DIRECT_COLOR = :steelblue4
const COARE_COLOR  = :firebrick

to_days(t, t0) = Dates.value(t - t0) / (1000.0 * 86400.0)

function plot_winter_season(year)
    t0 = DateTime(year - 1, 12, 1)
    t1 = DateTime(year, 3, 15, 23, 59, 59)

    idx = findall(t -> t0 <= t <= t1, glen_times)
    t_days = [to_days(t, t0) for t in glen_times[idx]]

    fig = Figure(size = (900, 900), fontsize = 11, figure_padding = (8, 12, 6, 6))

    month_starts = [DateTime(year - 1, 12, 1), DateTime(year, 1, 1),
                    DateTime(year, 2, 1),      DateTime(year, 3, 1)]
    month_labels = ["Dec 1", "Jan 1", "Feb 1", "Mar 1"]
    xtick_pos = [to_days(t, t0) for t in month_starts]

    axes = Axis[]
    for (row, (label, ylabel, direct, coare)) in enumerate(panels)
        ax = CairoMakie.Axis(fig[row, 1];
                 ylabel             = ylabel,
                 xlabel             = row == Nvar ? "Date" : "",
                 xticks             = (xtick_pos, month_labels),
                 xticklabelsvisible = row == Nvar,
                 title              = row == 1 ? "Winter $(year)" : "")
        push!(axes, ax)

        d_direct = direct[idx]
        if label == "Wind direction"
            valid = .!isnan.(d_direct)
            scatter!(ax, t_days[valid], d_direct[valid];
                     color = (DIRECT_COLOR, 0.4), markersize = 2)
            ylims!(ax, (0, 360))
            ax.yticks = ([0, 90, 180, 270, 360], ["0", "90", "180", "270", "360"])
        else
            lines!(ax, t_days, d_direct; color = DIRECT_COLOR, linewidth = 1.0)
            if !isnothing(coare)
                d_coare = coare[idx]
                lines!(ax, t_days, d_coare; color = (COARE_COLOR, 0.8), linewidth = 1.0,
                       linestyle = :dash)
            end
        end

        # Eastern mooring isothermal date (dotted vertical line)
        if haskey(eastern_iso, year)
            vlines!(ax, [to_days(eastern_iso[year], t0)];
                    color = :black, linewidth = 1.4, linestyle = :dot)
        end

        # Southern mooring isothermal date (only exists for 2010, 2011)
        if haskey(southern_iso, year)
            vlines!(ax, [to_days(southern_iso[year], t0)];
                    color = :darkorange, linewidth = 1.4, linestyle = :dot)
        end

        xlims!(ax, (to_days(t0, t0), to_days(t1, t0)))
    end
    linkxaxes!(axes...)

    # Legend
    legend_elems = Any[
        LineElement(color = DIRECT_COLOR, linewidth = 1.5),
        LineElement(color = (COARE_COLOR, 0.8), linewidth = 1.5, linestyle = :dash),
        LineElement(color = :black, linewidth = 1.4, linestyle = :dot),
    ]
    legend_labels = ["Direct (EC)", "COARE (wind-driven)", "Eastern iso. date"]
    if haskey(southern_iso, year)
        push!(legend_elems, LineElement(color = :darkorange, linewidth = 1.4, linestyle = :dot))
        push!(legend_labels, "Southern iso. date")
    end
    Legend(fig[Nvar + 1, 1], legend_elems, legend_labels;
           orientation = :horizontal, tellwidth = false, labelsize = 9,
           framevisible = false, padding = (0, 0, 0, 0))

    rowgap!(fig.layout, 6)

    outfig = joinpath(FIGURE_DIR, "stannard_rock_winter_flux_timeseries_winter$(year).pdf")
    save(outfig, fig)
    @info "Saved → $outfig"
end

for year in winter_years
    plot_winter_season(year)
end
