using Oceananigans
using CairoMakie
using JLD2

#####
##### Configuration
#####

DATA_DIR   = joinpath(@__DIR__, "..", "figure_data", "lake_superior_western_mooring")
FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

const H      = 184.0    # m, lake depth (divisible by dz=8 m)
const ρ₀     = 999.8    # kg/m³
const cₚ     = 4182.0   # J/(kg·K), freshwater specific heat
const C_D    = 1.2e-3
const ρ_air  = 1.225
const g_grav = 9.80665  # m/s²
const S_lake = 0.05 / 1000  # kg/kg absolute salinity (= 0.05 g/kg)
# T of maximum density at surface (p=0), SA=0.05 g/kg, TEOS-10: T_MD ≈ 3.9839 - 0.2137*SA
const T_MD   = 3.9839 - 0.2137 * (S_lake * 1000) + 273.15  # K  (≈ 277.12)

wind_stress(U_ms) = -ρ_air * C_D * U_ms^2 / ρ₀

closures = [
    ("kepsilon", "k-ε",     Makie.wong_colors()[6]),
    ("CATKE",    "CATKE",   Makie.wong_colors()[2]),
    ("NN",       "NORi NN", :black),
]

wind_cases = [
    ("U4ms", "U = 4 m/s", 4.0),
    ("U6ms", "U = 6 m/s", 6.0),
    ("U8ms", "U = 8 m/s", 8.0),
]

QT_mean_cases = [150.0, 200.0, 250.0]   # W/m²

# Insert NaN wherever L_MO changes sign so the line is discontinuous there
function break_at_sign_change(t, L)
    t_out = Float64[]
    L_out = Float64[]
    for i in eachindex(L)
        if i > 1 && sign(L[i]) != sign(L[i-1])
            push!(t_out, NaN)
            push!(L_out, NaN)
        end
        push!(t_out, t[i])
        push!(L_out, L[i])
    end
    return t_out, L_out
end

#####
##### Load data
#####

QT_tags = ["QTmean$(round(Int, Q_mean))Wm2" for Q_mean in QT_mean_cases]

# FieldTimeSeries for T profiles
Tbar_data = Dict(
    (wind, cname, QT_tag) =>
        FieldTimeSeries(joinpath(DATA_DIR, "$(cname)_$(wind)_$(QT_tag).jld2"), "Tbar")
    for (wind, _, _) in wind_cases
    for (cname, _, _) in closures
    for QT_tag in QT_tags
)

# LMO timeseries
LMO_data = Dict(
    (wind, cname, QT_tag) => jldopen(joinpath(DATA_DIR, "$(cname)_$(wind)_$(QT_tag)_LMO.jld2")) do f
        (t = f["t"] ./ 86400, L_MO = f["L_MO"])   # t in days
    end
    for (wind, _, _) in wind_cases
    for (cname, _, _) in closures
    for QT_tag in QT_tags
)

# Shared grid info (same for all runs)
ref_fts = first(values(Tbar_data))
zC    = znodes(ref_fts.grid, Center())        # cell centres [m]
Nz    = length(zC)
times = ref_fts.times ./ 86400               # [days]
Nt    = length(times)

# Profile snapshot indices: every 15 days, skipping t=0 (blue line handles that)
profile_inds = 16:15:Nt   # indices 16,31,...  →  15,30,45,60 days

#####
##### Figure 1 — Temperature profiles (9 panels: 3 QT × 3 wind)
#####

with_theme(theme_latexfonts()) do

    fig = Figure(size = (680, 680), fontsize = 12)

    n_snaps = length(profile_inds)
    alphas  = range(0.25, 1.0, length = n_snaps)

    ax_T = Matrix{Any}(undef, length(QT_mean_cases), length(wind_cases))

    for (row, Q_mean) in enumerate(QT_mean_cases)
        QT_tag = QT_tags[row]
        for (col, (wind, wind_label, _)) in enumerate(wind_cases)
            ax = CairoMakie.Axis(fig[row, col],
                     xlabel = row == length(QT_mean_cases) ? L"\Theta \; (^\circ\mathrm{C})" : "",
                     ylabel = col == 1 ? L"z \; \mathrm{(m)}" : "",
                     title  = row == 1 ? wind_label : "",
                     xticks = 0:1:5)
            ax_T[row, col] = ax

            for (cname, clabel, color) in closures
                fts = Tbar_data[(wind, cname, QT_tag)]
                for (snap_i, t_idx) in enumerate(profile_inds)
                    T_profile = interior(fts[t_idx], 1, 1, :)
                    lw = snap_i == n_snaps ? 2.5 : 1.5
                    lines!(ax, T_profile, zC,
                           color     = (color, alphas[snap_i]),
                           linewidth = lw,
                           label     = snap_i == n_snaps ? clabel : nothing)
                end
            end

            # Initial condition (dashed blue) — plotted last so it's on top
            T0 = interior(Tbar_data[(wind, first(closures)[1], QT_tag)][1], 1, 1, :)
            lines!(ax, T0, zC,
                   color = :blue, linewidth = 2.0, linestyle = :dash)

            xlims!(ax, (0.0, 4.3))
            ylims!(ax, (-H - 5, 5))
            col > 1 && hideydecorations!(ax, ticks = false, grid = false)
            row < length(QT_mean_cases) && hidexdecorations!(ax, ticks = false, grid = false)

            # Row label: QT_mean value
            if col == length(wind_cases)
                Label(fig[row, col + 1],
                      "Q̄ᵀ = $(round(Int, Q_mean)) W/m²",
                      rotation = -π/2, tellheight = false, fontsize = 12)
            end
        end
    end

    # Link axes within columns (same wind speed, varying QT)
    for col in 1:length(wind_cases)
        linkyaxes!([ax_T[row, col] for row in 1:length(QT_mean_cases)]...)
    end

    axislegend(ax_T[2, 2], position = :lb, labelsize = 11)

    time_labels = ["t = $(round(Int, times[i])) days" for i in profile_inds]
    time_elems  = [LineElement(color = (:grey30, alphas[i]), linewidth = 2) for i in 1:n_snaps]
    init_elem   = LineElement(color = :blue, linewidth = 2.0, linestyle = :dash)

    Legend(fig[end+1, 1:3], [init_elem; time_elems],
           ["t = 0 days (initial)"; time_labels],
           "Time snapshot", orientation = :horizontal, tellwidth = false)

    Label(fig[0, :], "Lake Superior, Western Mooring — Temperature profiles",
          fontsize = 15, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_T_profiles.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 2 — L_MO timeseries (9 panels: 3 QT × 3 wind)
#####

with_theme(theme_latexfonts()) do

    fig = Figure(size = (680, 680), fontsize = 12)

    ax_LMO = Matrix{Any}(undef, length(QT_mean_cases), length(wind_cases))

    # Symlog transform: log scale outside ±H, linear inside
    symlog_fwd(x) = sign(x) * (abs(x) <= H ? abs(x) : H * (1 + log(abs(x) / H)))
    symlog_inv(y) = sign(y) * (abs(y) <= H ? abs(y) : H * exp(abs(y) / H - 1))

    # Tick positions in data space for the symlog axis
    lmo_ticks_data  = [-5000.0, -1000.0, -500.0, -185.0, 0.0, 185.0, 500.0, 1000.0, 5000.0]
    lmo_ticks_trans = symlog_fwd.(lmo_ticks_data)
    lmo_tick_labels = [string(round(Int, v)) for v in lmo_ticks_data]

    for (row, Q_mean) in enumerate(QT_mean_cases)
        QT_tag = QT_tags[row]
        for (col, (wind, wind_label, _)) in enumerate(wind_cases)
            ax = CairoMakie.Axis(fig[row, col],
                     xlabel  = row == length(QT_mean_cases) ? "Time (days)" : "",
                     ylabel  = col == 1 ? L"L_{MO} \, \mathrm{(m)}" : "",
                     title   = row == 1 ? wind_label : "",
                     xticks  = LinearTicks(5),
                     yticks  = (lmo_ticks_trans, lmo_tick_labels),
                     ygridvisible = false)
            ax_LMO[row, col] = ax

            hlines!(ax, [symlog_fwd(0.0)],  color = :grey70, linewidth = 1,   linestyle = :dash)
            hlines!(ax, [symlog_fwd(H)],   color = :grey40, linewidth = 1.5, linestyle = :dot)
            hlines!(ax, [symlog_fwd(-H)],  color = :grey40, linewidth = 1.5, linestyle = :dot)

            for (cname, clabel, color) in closures
                d = LMO_data[(wind, cname, QT_tag)]
                t_plot, L_plot = break_at_sign_change(d.t, d.L_MO)
                lines!(ax, t_plot, symlog_fwd.(L_plot),
                       color = color, linewidth = 2, label = clabel)
            end

            xlims!(ax, (0, 60))
            ylims!(ax, symlog_fwd(-5000.0), symlog_fwd(5000.0))
            col > 1 && hideydecorations!(ax, ticks = false, grid = false)
            row < length(QT_mean_cases) && hidexdecorations!(ax, ticks = false, grid = false)

            if col == length(wind_cases)
                Label(fig[row, col + 1],
                      "Q̄ᵀ = $(round(Int, Q_mean)) W/m²",
                      rotation = -π/2, tellheight = false, fontsize = 12)
            end
        end
    end

    for col in 1:length(wind_cases)
        linkyaxes!([ax_LMO[row, col] for row in 1:length(QT_mean_cases)]...)
    end

    axislegend(ax_LMO[2, 2], position = :rb, labelsize = 11)

    Label(fig[0, :], "Lake Superior, Western Mooring — Monin-Obukhov length timeseries",
          fontsize = 15, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_LMO_timeseries.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 3 — MLD scaling (2 panels: 1 month and 2 months)
#####

# Mixed layer depth via temperature threshold from surface
function mixed_layer_depth(T_profile, zC; ΔT = 0.1)
    T_sfc = T_profile[end]
    for k in length(T_profile):-1:1
        if abs(T_profile[k] - T_sfc) > ΔT
            return abs(zC[k])
        end
    end
    return abs(zC[1])   # fully mixed
end

t_1month_idx = argmin(abs.(times .- 30.0))
t_2month_idx = argmin(abs.(times .- 60.0))

time_panels = [
    (t_1month_idx, "After 1 month"),
    (t_2month_idx, "After 2 months"),
]

const x_scale = 1e3   # scale M_sfc for readable tick labels (T_MD in K)

with_theme(theme_latexfonts()) do

    fig = Figure(size = (800, 370), fontsize = 14)

    ax_MLD = Axis[]

    for (panel, (t_idx, panel_title)) in enumerate(time_panels)
        ax = CairoMakie.Axis(fig[1, panel],
                 xlabel = L"M_\mathrm{sfc} \times 10^{3} = Q_u^{3/2} \, T_\mathrm{MD} \, / \, (g \, \bar{Q}^T \, H) \times 10^3",
                 ylabel = panel == 1 ? "MLD (m)" : "",
                 title  = panel_title)
        push!(ax_MLD, ax)

        for (cname, clabel, color) in closures
            x_vals = Float64[]
            y_vals = Float64[]

            for (qi, Q_mean) in enumerate(QT_mean_cases)
                QT_tag = QT_tags[qi]
                for (wind, _, U_ms) in wind_cases
                    Qᵁ     = abs(wind_stress(U_ms))
                    QT_kin = Q_mean / (ρ₀ * cₚ)
                    x      = Qᵁ^(3/2) * T_MD / (g_grav * QT_kin * H) * x_scale
                    fts    = Tbar_data[(wind, cname, QT_tag)]
                    T_prof = interior(fts[t_idx], 1, 1, :)
                    push!(x_vals, x)
                    push!(y_vals, mixed_layer_depth(T_prof, zC))
                end
            end

            perm = sortperm(x_vals)
            lines!(ax, x_vals[perm], y_vals[perm], color = color, linewidth = 2, label = clabel)
            scatter!(ax, x_vals, y_vals, color = color, markersize = 10)
        end

        panel > 1 && hideydecorations!(ax, ticks = false, grid = false)
    end

    linkaxes!(ax_MLD...)

    axislegend(ax_MLD[2], position = :lt, labelsize = 11)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_surface_T_scaling.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end
