using Oceananigans
using CairoMakie
using JLD2
using Statistics
using LaTeXStrings
using GibbsSeaWater

const S_lake = 0.05   # g/kg absolute salinity
const ρ₀     = 999.8
const g_grav = 9.80665

#####
##### Configuration
#####

LES_DIR    = joinpath(@__DIR__, "..", "data", "LES_output", "eastern_mooring")
FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

ts_file = joinpath(LES_DIR,
    "LES_lakesuperior_U10.0ms_QT250_Lxy256_Lz212_Nxy128_Nz106",
    "instantaneous_timeseries.jld2")

#####
##### Load fields
#####

@info "Loading $ts_file"
Tbar_fts = FieldTimeSeries(ts_file, "Tbar")
ubar_fts = FieldTimeSeries(ts_file, "ubar")
wT_fts   = FieldTimeSeries(ts_file, "wT")
uw_fts   = FieldTimeSeries(ts_file, "uw")

zC = znodes(Tbar_fts.grid, Center())
zF = znodes(Tbar_fts.grid, Face())
times_s    = Tbar_fts.times
times_days = times_s ./ 86400
Nt = length(times_s)

# Hourly averaging: output is every 10 min → 6 samples per hour
samples_per_hour = 6
Nt_hourly = div(Nt, samples_per_hour)

# Convert conservative temperature to in-situ at each depth
p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
Θ_to_Tinsitu(Θ_profile) = [gsw_t_from_ct(S_lake, Θ_profile[k], p_dbar[k]) for k in eachindex(Θ_profile)]

#####
##### Build hourly-averaged 2D arrays (time × depth)
#####

function hourly_avg_center(fts; transform = identity)
    Nz_c = length(zC)
    arr = zeros(Nt_hourly, Nz_c)
    for ih in 1:Nt_hourly
        i0 = (ih - 1) * samples_per_hour + 1
        i1 = ih * samples_per_hour
        profiles = [transform(interior(fts[i], 1, 1, :)) for i in i0:i1]
        arr[ih, :] = mean(profiles)
    end
    return arr
end

function hourly_avg_face(fts)
    Nz_f = length(zF)
    arr = zeros(Nt_hourly, Nz_f)
    for ih in 1:Nt_hourly
        i0 = (ih - 1) * samples_per_hour + 1
        i1 = ih * samples_per_hour
        profiles = [interior(fts[i], 1, 1, :) for i in i0:i1]
        arr[ih, :] = mean(profiles)
    end
    return arr
end

@info "Computing hourly averages..."
T_hourly  = hourly_avg_center(Tbar_fts, transform = Θ_to_Tinsitu)
u_hourly  = hourly_avg_center(ubar_fts)
wT_hourly = hourly_avg_face(wT_fts)
uw_hourly = hourly_avg_face(uw_fts)

t_hours = [(ih - 0.5) for ih in 1:Nt_hourly] ./ 24  # centre of each hour bin, in days

#####
##### Plot
#####

with_theme(theme_latexfonts()) do

    fig = Figure(size = (900, 900), fontsize = 13)

    panels = [
        (T_hourly,  zC, L"T \; (^\circ\mathrm{C})",              "T_hourly"),
        (u_hourly,  zC, L"\bar{u} \; (\mathrm{m/s})",            "u_hourly"),
        (wT_hourly, zF, L"\overline{w'T'} \; (^\circ\mathrm{C\,m/s})", "wT_hourly"),
        (uw_hourly, zF, L"\overline{u'w'} \; (\mathrm{m}^2/\mathrm{s}^2)", "uw_hourly"),
    ]

    for (idx, (arr, z, cbar_label, _)) in enumerate(panels)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1

        ax = CairoMakie.Axis(fig[row, col],
                 xlabel = row == 2 ? "Time (days)" : "",
                 ylabel = col == 1 ? L"z \; \mathrm{(m)}" : "")

        hm = heatmap!(ax, t_hours, z, arr, colormap = :balance)

        ylims!(ax, (-215, 5))
        xlims!(ax, (0, times_days[end]))
        col > 1 && hideydecorations!(ax, ticks = false, grid = false)
        row < 2 && hidexdecorations!(ax, ticks = false, grid = false)

        Colorbar(fig[row, col][1, 2], hm, label = cbar_label, width = 12)
    end

    Label(fig[0, :], "LES Eastern Mooring — U = 10 m/s, QT = 250 W/m² (hourly averaged)",
          fontsize = 15, font = :bold)

    colgap!(fig.layout, 15)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "LES_eastern_mooring_U10_QT250_hovmoller.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end
