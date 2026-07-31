using Oceananigans
using CairoMakie
using JLD2
using LaTeXStrings
using GibbsSeaWater

const S_lake = 0.05  # g/kg absolute salinity
const ρ₀     = 999.8
const g_grav = 9.80665

#####
##### Configuration
#####

TURB_DIR   = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring")
FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

U_cases  = [6.0, 10.0, 14.0]
QT_cases = [150, 250, 350]

wind_tags = Dict(6.0 => "U6ms", 10.0 => "U10ms", 14.0 => "U14ms")
QT_tags   = Dict(QT => "QTmean$(QT)Wm2" for QT in QT_cases)

snapshot_days = [0, 7, 14, 21, 28]

closure_models = [
    ("kepsilon",         L"k-\varepsilon"),
    ("kepsilon_Rist035", L"k-\varepsilon \; (Ri_{st}=0.35)"),
    ("CATKE",            "CATKE"),
    ("CATKE_highRi",     L"\mathrm{CATKE} \; (\mathrm{highRi})"),
]

#####
##### Load data for all closures
#####

data = Dict{Tuple{String,Float64,Int}, Dict{String, Any}}()

for (cname, _) in closure_models, U in U_cases, QT in QT_cases
    ts_file = joinpath(TURB_DIR, "$(cname)_$(wind_tags[U])_$(QT_tags[QT]).jld2")
    @info "Loading $ts_file"

    Tbar = FieldTimeSeries(ts_file, "Tbar")
    ubar = FieldTimeSeries(ts_file, "ubar")

    data[(cname, U, QT)] = Dict("Tbar" => Tbar, "ubar" => ubar)
end

ref = first(values(data))["Tbar"]
zC  = znodes(ref.grid, Center())
times_days = ref.times ./ 86400

function nearest_time_index(t_day)
    _, idx = findmin(abs.(times_days .- t_day))
    return idx
end

snap_indices = [nearest_time_index(d) for d in snapshot_days]
n_snaps = length(snap_indices)
alphas  = range(0.3, 1.0, length = n_snaps)

# Convert conservative temperature Θ to in-situ temperature T at each depth
p_dbar = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]
Θ_to_Tinsitu(Θ_profile) = [gsw_t_from_ct(S_lake, Θ_profile[k], p_dbar[k]) for k in eachindex(Θ_profile)]

#####
##### Plotting helper
#####

function plot_profiles(closure_name, closure_label, field_name, xlabel, filename;
                       profile_transform = identity,
                       xlims_range = nothing)

    with_theme(theme_latexfonts()) do

        fig = Figure(size = (720, 720), fontsize = 12)
        axes = Matrix{Any}(undef, length(QT_cases), length(U_cases))

        for (row, QT) in enumerate(QT_cases)
            for (col, U) in enumerate(U_cases)
                ax = CairoMakie.Axis(fig[row, col],
                         xlabel = row == length(QT_cases) ? xlabel : "",
                         ylabel = col == 1 ? L"z \; \mathrm{(m)}" : "",
                         title  = row == 1 ? "U = $(U) m/s" : "")
                axes[row, col] = ax

                fts = data[(closure_name, U, QT)][field_name]

                for (si, t_idx) in enumerate(snap_indices)
                    prof = profile_transform(interior(fts[t_idx], 1, 1, :))
                    lw = si == n_snaps ? 2.5 : 1.5
                    ls = si == 1 ? :dash : :solid
                    lines!(ax, prof, zC,
                           color     = (:black, alphas[si]),
                           linewidth = lw,
                           linestyle = ls)
                end

                ylims!(ax, (-215, 5))
                !isnothing(xlims_range) && xlims!(ax, xlims_range)
                col > 1 && hideydecorations!(ax, ticks = false, grid = false)
                row < length(QT_cases) && hidexdecorations!(ax, ticks = false, grid = false)

                if col == length(U_cases)
                    Label(fig[row, col + 1],
                          "QT = $(QT) W/m²",
                          rotation = -π/2, tellheight = false, fontsize = 12)
                end
            end
        end

        linkyaxes!(axes...)
        linkxaxes!(axes...)

        time_labels = ["t = $(d) days" for d in snapshot_days]
        time_labels[1] = "t = 0 days (initial)"
        time_elems = [LineElement(color     = (:black, alphas[i]),
                                  linewidth = i == n_snaps ? 2.5 : 1.5,
                                  linestyle = i == 1 ? :dash : :solid)
                      for i in 1:n_snaps]

        Legend(fig[end+1, 1:3], time_elems, time_labels,
               "Snapshot", orientation = :horizontal, tellwidth = false, labelsize = 10)

        Label(fig[0, :], closure_label, fontsize = 15, font = :bold)

        colgap!(fig.layout, 8)
        rowgap!(fig.layout, 10)

        outfile = joinpath(FIGURE_DIR, filename)
        save(outfile, fig)
        @info "Saved → $outfile"
        display(fig)
    end
end

#####
##### Generate figures for each closure model
#####

for (cname, clabel) in closure_models

    # Temperature profiles
    plot_profiles(cname, clabel, "Tbar",
                  L"T \; (^\circ\mathrm{C})",
                  "TURB_eastern_mooring_$(cname)_T_profiles.pdf",
                  profile_transform = Θ_to_Tinsitu,
                  xlims_range = (0.0, 5.0))

    # u velocity profiles
    plot_profiles(cname, clabel, "ubar",
                  L"\bar{u} \; (\mathrm{m/s})",
                  "TURB_eastern_mooring_$(cname)_u_profiles.pdf")
end
