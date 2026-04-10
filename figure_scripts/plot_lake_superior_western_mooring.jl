using Oceananigans
using CairoMakie
using JLD2
using Statistics
using GibbsSeaWater

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

# Default closures only — used for T_profiles, Ri_profiles, and LMO_timeseries figures
closures = [
    ("kepsilon",  "k-ε",    Makie.wong_colors()[6], :solid),
    ("CATKE",    "CATKE",   Makie.wong_colors()[2], :solid),
    ("NN",       "NORi NN", :steelblue,             :solid),
]

# Alternative (highRi) closures — used for alt T_profiles and Ri_profiles figures
alt_closures = [
    ("kepsilon_Rist035", L"k-\varepsilon\;(Ri_{st}=0.35)", Makie.wong_colors()[6], :dash),
    ("CATKE_highRi",     L"CATKE\;(Ri_{st}=0.23,\;\Gamma^0=0.4,\;\Gamma^\infty=0.2)", Makie.wong_colors()[2], :dash),
    ("NN_highRi",        L"NORi NN\;(Ri^{sh}=0.25,\;Ri_c=1.27)", :steelblue,      :dash),
]

# All closures including variants — used for MLD scaling figure only
closures_scaling = [
    ("kepsilon",         L"k-\varepsilon \; (Ri_{st}=0.25,\;C_{\varepsilon}^{b}=-0.65)", Makie.wong_colors()[6], :solid),
    ("kepsilon_Rist035", L"k-\varepsilon \; (Ri_{st}=0.35,\;C_{\varepsilon}^{b}=-0.26)",  Makie.wong_colors()[6], :dash),
    ("CATKE",            L"\mathrm{CATKE }\; (Ri_{st}=0.18,\;C_D^0=1.60)",  Makie.wong_colors()[2], :solid),
    ("CATKE_highRi",     L"\mathrm{CATKE }\; (Ri_{st}=0.23,\;\Gamma^0=0.4,\;\Gamma^\infty=0.2)", Makie.wong_colors()[2], :dash),
    ("NN",               L"\mathrm{NORi NN }\; (Ri_c=0.44)",                             :steelblue, :solid),
    ("NN_highRi",        L"\mathrm{NORi NN }\; (Ri^{sh}=0.25,\;Ri_c=1.27)",            :steelblue, :dash),
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

# FieldTimeSeries for T profiles — load all closures (default + variants) for scaling plot
Tbar_data = Dict(
    (wind, cname, QT_tag) =>
        FieldTimeSeries(joinpath(DATA_DIR, "$(cname)_$(wind)_$(QT_tag).jld2"), "Tbar")
    for (wind, _, _) in wind_cases
    for (cname, _, _, _) in closures_scaling
    for QT_tag in QT_tags
)

# Ribar profiles — default and alt closures
Ribar_data = Dict(
    (wind, cname, QT_tag) =>
        FieldTimeSeries(joinpath(DATA_DIR, "$(cname)_$(wind)_$(QT_tag).jld2"), "Ribar")
    for (wind, _, _) in wind_cases
    for (cname, _, _, _) in [closures; alt_closures]
    for QT_tag in QT_tags
)

# LMO timeseries — default closures only
LMO_data = Dict(
    (wind, cname, QT_tag) => jldopen(joinpath(DATA_DIR, "$(cname)_$(wind)_$(QT_tag)_LMO.jld2")) do f
        (t = f["t"] ./ 86400, L_MO = f["L_MO"])   # t in days
    end
    for (wind, _, _) in wind_cases
    for (cname, _, _, _) in closures
    for QT_tag in QT_tags
)

# Shared grid info (same for all runs)
ref_fts = first(values(Tbar_data))
zC    = znodes(ref_fts.grid, Center())        # cell centres [m]
zF    = znodes(ref_fts.grid, Face())[2:end-1] # interior face nodes [m] (Ri is at Face, drop boundaries)
Nz    = length(zC)
times = ref_fts.times ./ 86400               # [days]
Nt    = length(times)

# Profile snapshot indices: every 15 days, skipping t=0 (blue line handles that)
profile_inds = 16:15:Nt   # indices 16,31,...  →  15,30,45,60 days

# Ri plot axis limits and reference value (used across Figures 2 and 2b)
const Ri_lo       = -1.0
const Ri_hi       =  2.0
const Riᶜ_default = 0.4366901962987793   # calibrated default Riᶜ

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

            for (cname, clabel, color, _) in closures
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
##### Figure 1b — Temperature profiles — alternative (highRi) closures
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

            for (cname, clabel, color, lstyle) in alt_closures
                fts = Tbar_data[(wind, cname, QT_tag)]
                for (snap_i, t_idx) in enumerate(profile_inds)
                    T_profile = interior(fts[t_idx], 1, 1, :)
                    lw = snap_i == n_snaps ? 2.5 : 1.5
                    lines!(ax, T_profile, zC,
                           color     = (color, alphas[snap_i]),
                           linewidth = lw,
                           linestyle = lstyle,
                           label     = snap_i == n_snaps ? clabel : nothing)
                end
            end

            T0 = interior(Tbar_data[(wind, first(alt_closures)[1], QT_tag)][1], 1, 1, :)
            lines!(ax, T0, zC, color = :blue, linewidth = 2.0, linestyle = :dash)

            xlims!(ax, (0.0, 4.3))
            ylims!(ax, (-H - 5, 5))
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
        linkyaxes!([ax_T[row, col] for row in 1:length(QT_mean_cases)]...)
    end

    axislegend(ax_T[2, 2], position = :lb, labelsize = 11)

    time_labels = ["t = $(round(Int, times[i])) days" for i in profile_inds]
    time_elems  = [LineElement(color = (:grey30, alphas[i]), linewidth = 2) for i in 1:n_snaps]
    init_elem   = LineElement(color = :blue, linewidth = 2.0, linestyle = :dash)

    Legend(fig[end+1, 1:3], [init_elem; time_elems],
           ["t = 0 days (initial)"; time_labels],
           "Time snapshot", orientation = :horizontal, tellwidth = false)

    Label(fig[0, :], "Lake Superior, Western Mooring — Temperature profiles (alternative closures)",
          fontsize = 15, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_T_profiles_alt.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 2 — Richardson number profiles — default closures (9 panels: 3 QT × 3 wind)
#####
# Ri clamped to [-2, 2] for display; convective (Ri<0) on left, stable (Ri>0) on right.
# Vertical dashed lines mark Ri=0 and Ri=Riᶜ (≈0.44).

with_theme(theme_latexfonts()) do

    fig = Figure(size = (680, 680), fontsize = 12)

    n_snaps = length(profile_inds)
    alphas  = range(0.25, 1.0, length = n_snaps)

    ax_Ri = Matrix{Any}(undef, length(QT_mean_cases), length(wind_cases))

    for (row, Q_mean) in enumerate(QT_mean_cases)
        QT_tag = QT_tags[row]
        for (col, (wind, wind_label, _)) in enumerate(wind_cases)
            ax = CairoMakie.Axis(fig[row, col],
                     xlabel = row == length(QT_mean_cases) ? L"Ri" : "",
                     ylabel = col == 1 ? L"z \; \mathrm{(m)}" : "",
                     title  = row == 1 ? wind_label : "")
            ax_Ri[row, col] = ax

            vlines!(ax, [0.0],          color = :grey60, linewidth = 1,   linestyle = :dash)
            vlines!(ax, [Riᶜ_default], color = :grey40, linewidth = 1.5, linestyle = :dot)

            for (cname, clabel, color, _) in closures
                fts = Ribar_data[(wind, cname, QT_tag)]
                for (snap_i, t_idx) in enumerate(profile_inds)
                    Ri_prof = clamp.(interior(fts[t_idx], 1, 1, 2:Nz), Ri_lo, Ri_hi)
                    lw = snap_i == n_snaps ? 2.5 : 1.5
                    lines!(ax, Ri_prof, zF,
                           color     = (color, alphas[snap_i]),
                           linewidth = lw,
                           label     = snap_i == n_snaps ? clabel : nothing)
                end
            end

            xlims!(ax, (Ri_lo, Ri_hi))
            ylims!(ax, (-H - 5, 5))
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
        linkyaxes!([ax_Ri[row, col] for row in 1:length(QT_mean_cases)]...)
    end

    axislegend(ax_Ri[2, 2], position = :rb, labelsize = 11)

    time_labels = ["t = $(round(Int, times[i])) days" for i in profile_inds]
    time_elems  = [LineElement(color = (:grey30, alphas[i]), linewidth = 2) for i in 1:n_snaps]
    ri0_elem    = LineElement(color = :grey60, linewidth = 1,   linestyle = :dash)
    ric_elem    = LineElement(color = :grey40, linewidth = 1.5, linestyle = :dot)

    Legend(fig[end+1, 1:3],
           [time_elems; [ri0_elem, ric_elem]],
           [time_labels; ["Ri = 0", "Ri = Riᶜ (≈0.44)"]],
           "Time snapshot", orientation = :horizontal, nbanks = 2, tellwidth = false)

    Label(fig[0, :], "Lake Superior, Western Mooring — Richardson number profiles",
          fontsize = 15, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_Ri_profiles.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 2b — Richardson number profiles — alternative (highRi) closures
#####

with_theme(theme_latexfonts()) do

    fig = Figure(size = (680, 680), fontsize = 12)

    n_snaps = length(profile_inds)
    alphas  = range(0.25, 1.0, length = n_snaps)

    ax_Ri = Matrix{Any}(undef, length(QT_mean_cases), length(wind_cases))

    for (row, Q_mean) in enumerate(QT_mean_cases)
        QT_tag = QT_tags[row]
        for (col, (wind, wind_label, _)) in enumerate(wind_cases)
            ax = CairoMakie.Axis(fig[row, col],
                     xlabel = row == length(QT_mean_cases) ? L"Ri" : "",
                     ylabel = col == 1 ? L"z \; \mathrm{(m)}" : "",
                     title  = row == 1 ? wind_label : "")
            ax_Ri[row, col] = ax

            vlines!(ax, [0.0],          color = :grey60, linewidth = 1,   linestyle = :dash)
            vlines!(ax, [Riᶜ_default], color = :grey40, linewidth = 1.5, linestyle = :dot)

            for (cname, clabel, color, lstyle) in alt_closures
                fts = Ribar_data[(wind, cname, QT_tag)]
                for (snap_i, t_idx) in enumerate(profile_inds)
                    Ri_prof = clamp.(interior(fts[t_idx], 1, 1, 2:Nz), Ri_lo, Ri_hi)
                    lw = snap_i == n_snaps ? 2.5 : 1.5
                    lines!(ax, Ri_prof, zF,
                           color     = (color, alphas[snap_i]),
                           linewidth = lw,
                           linestyle = lstyle,
                           label     = snap_i == n_snaps ? clabel : nothing)
                end
            end

            xlims!(ax, (Ri_lo, Ri_hi))
            ylims!(ax, (-H - 5, 5))
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
        linkyaxes!([ax_Ri[row, col] for row in 1:length(QT_mean_cases)]...)
    end

    axislegend(ax_Ri[2, 2], position = :rb, labelsize = 11)

    time_labels = ["t = $(round(Int, times[i])) days" for i in profile_inds]
    time_elems  = [LineElement(color = (:grey30, alphas[i]), linewidth = 2) for i in 1:n_snaps]
    ri0_elem    = LineElement(color = :grey60, linewidth = 1,   linestyle = :dash)
    ric_elem    = LineElement(color = :grey40, linewidth = 1.5, linestyle = :dot)

    Legend(fig[end+1, 1:3],
           [time_elems; [ri0_elem, ric_elem]],
           [time_labels; ["Ri = 0", "Ri = Riᶜ (≈0.44)"]],
           "Time snapshot", orientation = :horizontal, nbanks = 2, tellwidth = false)

    Label(fig[0, :], "Lake Superior, Western Mooring — Richardson number profiles (alternative closures)",
          fontsize = 15, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 10)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_Ri_profiles_alt.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 3 — L_MO timeseries (9 panels: 3 QT × 3 wind)
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

            for (cname, clabel, color, _) in closures
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

# ── Observed MLD ──────────────────────────────────────────────────────────────
obs_mld_file = joinpath(DATA_DIR, "western_mooring_observed_mld.jld2")
obs_mld = if isfile(obs_mld_file)
    jldopen(obs_mld_file) do f
        (mld1        = f["mld1_obs"],
         mld2        = f["mld2_obs"],
         Tbot1       = f["Tbot1_obs"],
         Tbot2       = f["Tbot2_obs"],
         dep         = f["dep_obs"],
         T0_profiles = f["T0_profiles"],
         T1_profiles = f["T1_profiles"],
         T2_profiles = f["T2_profiles"])
    end
else
    @warn "Observed MLD file not found; run figure_scripts/process_lake_superior_observations.jl first."
    nothing
end

time_panels = [
    (t_1month_idx, "After 1 month"),
    (t_2month_idx, "After 2 months"),
]

const x_scale = 1e3   # scale M_sfc for readable tick labels (T_MD in K)

# Pre-compute x values (same for all closures) to position the violin
x_vals_scaling = Float64[]
for Q_mean in QT_mean_cases
    for (_, _, U_ms) in wind_cases
        Qᵁ     = abs(wind_stress(U_ms))
        QT_kin = Q_mean / (ρ₀ * cₚ)
        push!(x_vals_scaling, Qᵁ^(3/2) * T_MD / (g_grav * QT_kin * H) * x_scale)
    end
end
x_min_sc    = minimum(x_vals_scaling)
x_max_sc    = maximum(x_vals_scaling)
violin_x    = (x_min_sc + x_max_sc) / 2          # centre of the x range
violin_width = 0.12 * (x_max_sc - x_min_sc)      # narrow relative to axis

with_theme(theme_latexfonts()) do

    fig = Figure(size = (800, 480), fontsize = 14)

    ax_MLD = Axis[]

    for (panel, (t_idx, panel_title)) in enumerate(time_panels)
        ax = CairoMakie.Axis(fig[1, panel],
                 xlabel = L"M_\mathrm{sfc} \times 10^{3} = Q_u^{3/2} \, T_\mathrm{MD} \, / \, (g \, \bar{Q}^T \, H) \times 10^3",
                 ylabel = panel == 1 ? "MLD (m)" : "",
                 title  = panel_title)
        push!(ax_MLD, ax)

        for (cname, clabel, color, lstyle) in closures_scaling
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
            lines!(ax, x_vals[perm], y_vals[perm],
                   color = color, linewidth = 2, linestyle = lstyle, label = clabel)
            lstyle == :solid && scatter!(ax, x_vals, y_vals, color = color, markersize = 10)
        end

        # Observed MLD: violin showing distribution across years
        if !isnothing(obs_mld)
            mld_obs_panel = panel == 1 ? obs_mld.mld1 : obs_mld.mld2
            q1  = quantile(mld_obs_panel, 0.25)
            q3  = quantile(mld_obs_panel, 0.75)
            med = median(mld_obs_panel)
            lo  = minimum(mld_obs_panel)
            hi  = maximum(mld_obs_panel)
            hw  = violin_width / 2   # half box width
            cap = hw * 0.5           # whisker cap half-width
            # IQR box
            poly!(ax, Point2f[(violin_x - hw, q1), (violin_x + hw, q1),
                              (violin_x + hw, q3), (violin_x - hw, q3)],
                  color = (:grey50, 0.4), strokecolor = :grey30, strokewidth = 1.5)
            # Median line
            lines!(ax, [violin_x - hw, violin_x + hw], [med, med],
                   color = :grey10, linewidth = 2)
            # Whiskers
            lines!(ax, [violin_x, violin_x], [q3, hi],
                   color = :grey30, linewidth = 1.5)
            lines!(ax, [violin_x, violin_x], [q1, lo],
                   color = :grey30, linewidth = 1.5)
            # Whisker caps
            lines!(ax, [violin_x - cap, violin_x + cap], [hi, hi],
                   color = :grey30, linewidth = 1.5)
            lines!(ax, [violin_x - cap, violin_x + cap], [lo, lo],
                   color = :grey30, linewidth = 1.5)
        end

        panel > 1 && hideydecorations!(ax, ticks = false, grid = false)
    end

    linkaxes!(ax_MLD...)

    # Legend below panels
    sim_elems  = Any[LineElement(color = color, linewidth = 2, linestyle = lstyle)
                     for (_, _, color, lstyle) in closures_scaling]
    sim_labels = [clabel for (_, clabel, _, _) in closures_scaling]
    if !isnothing(obs_mld)
        push!(sim_elems,  [PolyElement(color = (:grey50, 0.4), strokecolor = :grey30, strokewidth = 1.5),
                            LineElement(color = :grey10, linewidth = 1.5)])
        push!(sim_labels, "WM observations")
    end
    Legend(fig[2, 1:2], sim_elems, sim_labels,
           orientation = :horizontal, nbanks = 2,
           tellwidth = false, labelsize = 10, framevisible = false)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 8)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_MLD_scaling.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 4 — Bottom temperature scaling (2 panels: 1 month and 2 months)
#####
# Model: conservative temperature at deepest cell → in-situ T via gsw_t_from_ct
# Observations: in-situ temperature at deepest DEP_COMMON level (180 m)
#
# gsw_t_from_ct(SA, Θ, p): SA in g/kg, Θ in °C, p in dbar
# Pressure at z: p_dbar = ρ₀ · g · |z| / 1e4

# Deepest model cell centre and corresponding pressure
const z_bot   = zC[1]                            # e.g. -180 m
const p_bot   = ρ₀ * g_grav * abs(z_bot) / 1e4  # dbar
Θ_to_Tinsitu(Θ) = gsw_t_from_ct(S_lake * 1e3, Θ, p_bot)  # SA in g/kg

with_theme(theme_latexfonts()) do

    fig = Figure(size = (800, 480), fontsize = 14)

    ax_Tbot = Axis[]

    for (panel, (t_idx, panel_title)) in enumerate(time_panels)
        ax = CairoMakie.Axis(fig[1, panel],
                 xlabel = L"M_\mathrm{sfc} \times 10^{3} = Q_u^{3/2} \, T_\mathrm{MD} \, / \, (g \, \bar{Q}^T \, H) \times 10^3",
                 ylabel = panel == 1 ? L"T_\mathrm{bot} \; (^\circ\mathrm{C})" : "",
                 title  = panel_title)
        push!(ax_Tbot, ax)

        for (cname, clabel, color, lstyle) in closures_scaling
            x_vals = Float64[]
            y_vals = Float64[]

            for (qi, Q_mean) in enumerate(QT_mean_cases)
                QT_tag = QT_tags[qi]
                for (wind, _, U_ms) in wind_cases
                    Qᵁ     = abs(wind_stress(U_ms))
                    QT_kin = Q_mean / (ρ₀ * cₚ)
                    x      = Qᵁ^(3/2) * T_MD / (g_grav * QT_kin * H) * x_scale
                    fts    = Tbar_data[(wind, cname, QT_tag)]
                    Θ_bot  = interior(fts[t_idx], 1, 1, 1)[1]   # conservative T at deepest cell
                    push!(x_vals, x)
                    push!(y_vals, Θ_to_Tinsitu(Θ_bot))
                end
            end

            perm = sortperm(x_vals)
            lines!(ax, x_vals[perm], y_vals[perm],
                   color = color, linewidth = 2, linestyle = lstyle, label = clabel)
            lstyle == :solid && scatter!(ax, x_vals, y_vals, color = color, markersize = 10)
        end

        # Observed bottom temperature: box plot (same style as MLD)
        if !isnothing(obs_mld)
            Tbot_obs_panel = panel == 1 ? obs_mld.Tbot1 : obs_mld.Tbot2
            q1  = quantile(Tbot_obs_panel, 0.25)
            q3  = quantile(Tbot_obs_panel, 0.75)
            med = median(Tbot_obs_panel)
            lo  = minimum(Tbot_obs_panel)
            hi  = maximum(Tbot_obs_panel)
            hw  = violin_width / 2
            cap = hw * 0.5
            poly!(ax, Point2f[(violin_x - hw, q1), (violin_x + hw, q1),
                              (violin_x + hw, q3), (violin_x - hw, q3)],
                  color = (:grey50, 0.4), strokecolor = :grey30, strokewidth = 1.5)
            lines!(ax, [violin_x - hw, violin_x + hw], [med, med],
                   color = :grey10, linewidth = 2)
            lines!(ax, [violin_x, violin_x], [q3, hi],
                   color = :grey30, linewidth = 1.5)
            lines!(ax, [violin_x, violin_x], [q1, lo],
                   color = :grey30, linewidth = 1.5)
            lines!(ax, [violin_x - cap, violin_x + cap], [hi, hi],
                   color = :grey30, linewidth = 1.5)
            lines!(ax, [violin_x - cap, violin_x + cap], [lo, lo],
                   color = :grey30, linewidth = 1.5)
        end

        panel > 1 && hideydecorations!(ax, ticks = false, grid = false)
    end

    linkaxes!(ax_Tbot...)

    # Legend below panels
    sim_elems  = Any[LineElement(color = color, linewidth = 2, linestyle = lstyle)
                     for (_, _, color, lstyle) in closures_scaling]
    sim_labels = [clabel for (_, clabel, _, _) in closures_scaling]
    if !isnothing(obs_mld)
        push!(sim_elems,  [PolyElement(color = (:grey50, 0.4), strokecolor = :grey30, strokewidth = 1.5),
                            LineElement(color = :grey10, linewidth = 1.5)])
        push!(sim_labels, "WM observations")
    end
    Legend(fig[2, 1:2], sim_elems, sim_labels,
           orientation = :horizontal, nbanks = 2,
           tellwidth = false, labelsize = 10, framevisible = false)

    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 8)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_bottom_T_scaling.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 5 — T profile comparison after 1 month: 3 model closures + observations (4 panels)
#####

# Convert a conservative temperature profile (at model cell centres zC) to in-situ temperature.
# gsw_t_from_ct(SA [g/kg], Θ [°C], p [dbar])
const p_levels = [ρ₀ * g_grav * abs(z) / 1e4 for z in zC]   # pressure at each cell centre
Θ_to_Tinsitu_profile(Θ_prof) = [gsw_t_from_ct(S_lake * 1e3, Θ_prof[k], p_levels[k])
                                 for k in eachindex(Θ_prof)]

with_theme(theme_latexfonts()) do

    fig = Figure(size = (900, 420), fontsize = 12)

    # One panel per default closure + one for observations
    panel_specs = [
        (closures[1], Makie.wong_colors()[6]),   # kepsilon
        (closures[2], Makie.wong_colors()[2]),   # CATKE
        (closures[3], :steelblue),               # NN
    ]
    panel_titles = [clabel for (_, clabel, _, _) in closures]
    push!(panel_titles, "Observations")

    ax_all = Axis[]

    # ── Model panels (panels 1–3) ─────────────────────────────────────────────
    for (panel, ((cname, clabel, color, _), color2)) in enumerate(panel_specs)

        ax = CairoMakie.Axis(fig[1, panel],
                 xlabel = L"T \; (^\circ\mathrm{C})",
                 ylabel = panel == 1 ? L"z \; \mathrm{(m)}" : "",
                 title  = panel_titles[panel],
                 xticks = 0:1:5)
        push!(ax_all, ax)

        # Collect all forcing combinations at t = 1 month, converting Θ → T_insitu
        all_profiles = Vector{Vector{Float64}}()
        for (qi, Q_mean) in enumerate(QT_mean_cases)
            QT_tag = QT_tags[qi]
            for (wind, _, _) in wind_cases
                fts    = Tbar_data[(wind, cname, QT_tag)]
                Θ_prof = Float64.(interior(fts[t_1month_idx], 1, 1, :))
                T_prof = Θ_to_Tinsitu_profile(Θ_prof)
                push!(all_profiles, T_prof)
                lines!(ax, T_prof, zC, color = (color, 0.3), linewidth = 1.0)
            end
        end

        # Median across all forcing combinations
        T_mat = hcat(all_profiles...)   # Nz × N_combos
        T_med = median(T_mat, dims = 2)[:, 1]
        lines!(ax, T_med, zC, color = color, linewidth = 2.5)

        xlims!(ax, (0.0, 4.5))
        ylims!(ax, (-H - 5, 5))
        panel > 1 && hideydecorations!(ax, ticks = false, grid = false)
    end

    # ── Observations panel (panel 4) ─────────────────────────────────────────
    if !isnothing(obs_mld)
        ax_obs = CairoMakie.Axis(fig[1, 4],
                     xlabel = L"T \; (^\circ\mathrm{C})",
                     title  = "Observations",
                     xticks = 0:1:5)
        push!(ax_all, ax_obs)

        z_obs    = -obs_mld.dep   # positive-downward → negative z
        Nseasons = size(obs_mld.T1_profiles, 2)

        for s in 1:Nseasons
            lines!(ax_obs, obs_mld.T1_profiles[:, s], z_obs,
                   color = (:steelblue, 0.3), linewidth = 1.0)
        end
        T1_med = median(obs_mld.T1_profiles, dims = 2)[:, 1]
        lines!(ax_obs, T1_med, z_obs, color = :steelblue, linewidth = 2.5)

        xlims!(ax_obs, (0.0, 4.5))
        ylims!(ax_obs, (-H - 5, 5))
        hideydecorations!(ax_obs, ticks = false, grid = false)
    end

    linkaxes!(ax_all...)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elems = [
        LineElement(color = :grey40,           linewidth = 2.5),
        LineElement(color = (:grey40, 0.3),    linewidth = 1.0),
    ]
    legend_labels = ["Median", "Individual cases / years"]
    Legend(fig[2, 1:4], legend_elems, legend_labels,
           orientation = :horizontal, tellwidth = false,
           labelsize = 11, framevisible = false)

    Label(fig[0, :],
          "Lake Superior, Western Mooring — Temperature profiles after 1 month",
          fontsize = 13, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 6)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_T_profiles_1month_comparison.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

#####
##### Figure 6 — T profile comparison after 1 month: alternative (highRi) closures + observations
#####

with_theme(theme_latexfonts()) do

    fig = Figure(size = (900, 420), fontsize = 12)

    # Alternative closure variants (highRi)
    alt_panel_specs = [
        ("kepsilon_Rist035", L"k-\varepsilon\;(Ri_{st}=0.35)",  Makie.wong_colors()[6]),
        ("CATKE_highRi",     L"CATKE\;(Ri_{st}=0.23,\;\Gamma^0=0.4,\;\Gamma^\infty=0.2)", Makie.wong_colors()[2]),
        ("NN_highRi",        L"NORi NN\;(Ri^{sh}=0.25,\;Ri_c=1.27)", :steelblue),
    ]

    ax_all6 = Axis[]

    for (panel, (cname, clabel, color)) in enumerate(alt_panel_specs)

        ax = CairoMakie.Axis(fig[1, panel],
                 xlabel = L"T \; (^\circ\mathrm{C})",
                 ylabel = panel == 1 ? L"z \; \mathrm{(m)}" : "",
                 title  = clabel,
                 xticks = 0:1:5)
        push!(ax_all6, ax)

        all_profiles = Vector{Vector{Float64}}()
        for (qi, Q_mean) in enumerate(QT_mean_cases)
            QT_tag = QT_tags[qi]
            for (wind, _, _) in wind_cases
                fts    = Tbar_data[(wind, cname, QT_tag)]
                Θ_prof = Float64.(interior(fts[t_1month_idx], 1, 1, :))
                T_prof = Θ_to_Tinsitu_profile(Θ_prof)
                push!(all_profiles, T_prof)
                lines!(ax, T_prof, zC, color = (color, 0.3), linewidth = 1.0)
            end
        end

        T_mat = hcat(all_profiles...)
        T_med = median(T_mat, dims = 2)[:, 1]
        lines!(ax, T_med, zC, color = color, linewidth = 2.5)

        xlims!(ax, (0.0, 4.5))
        ylims!(ax, (-H - 5, 5))
        panel > 1 && hideydecorations!(ax, ticks = false, grid = false)
    end

    # ── Observations panel ────────────────────────────────────────────────────
    if !isnothing(obs_mld)
        ax_obs = CairoMakie.Axis(fig[1, 4],
                     xlabel = L"T \; (^\circ\mathrm{C})",
                     title  = "Observations",
                     xticks = 0:1:5)
        push!(ax_all6, ax_obs)

        z_obs    = -obs_mld.dep
        Nseasons = size(obs_mld.T1_profiles, 2)
        for s in 1:Nseasons
            lines!(ax_obs, obs_mld.T1_profiles[:, s], z_obs,
                   color = (:steelblue, 0.3), linewidth = 1.0)
        end
        T1_med = median(obs_mld.T1_profiles, dims = 2)[:, 1]
        lines!(ax_obs, T1_med, z_obs, color = :steelblue, linewidth = 2.5)

        xlims!(ax_obs, (0.0, 4.5))
        ylims!(ax_obs, (-H - 5, 5))
        hideydecorations!(ax_obs, ticks = false, grid = false)
    end

    linkaxes!(ax_all6...)

    legend_elems = [
        LineElement(color = :grey40,        linewidth = 2.5),
        LineElement(color = (:grey40, 0.3), linewidth = 1.0),
    ]
    Legend(fig[2, 1:4], legend_elems, ["Median", "Individual cases / years"],
           orientation = :horizontal, tellwidth = false,
           labelsize = 11, framevisible = false)

    Label(fig[0, :],
          "Lake Superior, Western Mooring — Temperature profiles after 1 month (alternative closures)",
          fontsize = 13, font = :bold)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 6)

    outfile = joinpath(FIGURE_DIR, "lake_superior_western_mooring_T_profiles_1month_altclosures.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
    display(fig)
end

