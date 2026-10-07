#####
##### Thermal expansion coefficient α(T) between T_MD and 0 °C (TEOS-10, Lake Superior salinity)
#####
# α = -(1/ρ)(∂ρ/∂T)_{S,p}, evaluated at the surface (p = 0 dbar) with the exact
# TEOS-10 equation of state (gsw_alpha_wrt_t_exact, w.r.t. in-situ temperature).
# α = 0 exactly at the temperature of maximum density T_MD; α < 0 below it
# (freshwater's anomalous expansion on further cooling) and α > 0 above it.
#
# Inputs  : none (S_lake is the same constant used throughout these figure scripts)
# Outputs : figures/thermal_expansion_coefficient_TEOS10.pdf
#####

using CairoMakie
using GibbsSeaWater
using LaTeXStrings
using Printf

# LaTeX scientific notation, e.g. sci_latex(4.56e-5) -> "4.6\times10^{-5}".
function sci_latex(x; dec = 2)
    x == 0 && return "0"
    s     = @sprintf("%.*e", dec, x)
    m, e  = split(s, 'e')
    expo  = parse(Int, e)
    return "$(m)\\times10^{$(expo)}"
end

const S_lake = 0.05   # g/kg — Lake Superior absolute salinity (matches other figure scripts)
const P_SFC  = 0.0    # dbar — surface pressure

# Eastern/Southern Mooring site depths and latitudes (Austin2023; matches
# plot_lake_superior_map.jl and the LES/TURB domain depths Lz_EM/Lz_SM used
# throughout the other mooring figure scripts) — used to get the pressure
# equivalent to each mooring's bottom depth via the TEOS-10 p(z) relation.
const H_EM   = 212.0    # m — Eastern Mooring water depth
const H_SM   = 384.0    # m — Southern Mooring water depth
const LAT_EM = 47.537   # °N
const LAT_SM = 47.03333 # °N

const P_EM = gsw_p_from_z(-H_EM, LAT_EM)   # dbar
const P_SM = gsw_p_from_z(-H_SM, LAT_SM)   # dbar

FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# In-situ temperature of maximum density at the surface (same construction as
# the M_sfc scripts, but keeping it in Celsius here since this is a T-axis plot).
# T_MD itself is pressure-dependent, so each curve's zero-crossing sits at its
# own T_MD(p) rather than all three sharing the surface value.
const T_MD    = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, P_SFC), P_SFC)
const T_MD_EM = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, P_EM), P_EM)
const T_MD_SM = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, P_SM), P_SM)

T_range = range(0.0, T_MD; length = 400)
α       = [gsw_alpha_wrt_t_exact(S_lake, T, P_SFC) for T in T_range]
α_EM    = [gsw_alpha_wrt_t_exact(S_lake, T, P_EM) for T in T_range]
α_SM    = [gsw_alpha_wrt_t_exact(S_lake, T, P_SM) for T in T_range]
const α0 = gsw_alpha_wrt_t_exact(S_lake, 0.0, P_SFC)   # α at T = 0 °C, called out on the y-axis

with_theme(theme_latexfonts()) do
    fig = Figure(size = (340, 280), fontsize = 12, figure_padding = (6, 36, 6, 6))

    # Round background ticks plus the exact α(0 °C) value called out explicitly.
    ytick_vals = sort(unique([α0, -5e-5, -2.5e-5, 0.0]))
    ytick_labs = [v == 0.0 ? L"0" : latexstring(sci_latex(v)) for v in ytick_vals]

    # Round background ticks plus each curve's own T_MD (α = 0) value called out explicitly.
    md_symbols = Dict(T_MD => "T_{\\mathrm{MD}}^{\\mathrm{surf}}", T_MD_EM => "T_{\\mathrm{MD}}^{\\mathrm{EM}}",
                       T_MD_SM => "T_{\\mathrm{MD}}^{\\mathrm{SM}}")
    xtick_vals = sort(unique(vcat(0:1:floor(Int, T_MD) - 1, T_MD, T_MD_EM, T_MD_SM)))
    xtick_labs = [haskey(md_symbols, v) ? latexstring(md_symbols[v], " = ", round(v; digits = 2)) :
                                           latexstring(round(Int, v))
                  for v in xtick_vals]

    ax = Axis(fig[1, 1];
              xlabel = L"T \; (^\circ\mathrm{C})",
              ylabel = L"\alpha \; (\mathrm{K^{-1}})",
              xticks = (xtick_vals, xtick_labs),
              xticklabelsize = 8,
              xticklabelrotation = pi / 6,
              yticks = (ytick_vals, ytick_labs))

    hlines!(ax, [0.0]; color = :gray60, linewidth = 1.0, linestyle = :dash)
    l_sfc = lines!(ax, T_range, α; color = :steelblue4, linewidth = 2.5)
    l_em  = lines!(ax, T_range, α_EM; color = :darkorange2, linewidth = 2.0, linestyle = :dash)
    l_sm  = lines!(ax, T_range, α_SM; color = :seagreen, linewidth = 2.0, linestyle = :dot)

    # Highlight each curve's own T_MD (where α = 0), since T_MD shifts with pressure.
    vlines!(ax, [T_MD, T_MD_EM, T_MD_SM];
            color = [(:steelblue4, 0.4), (:darkorange2, 0.4), (:seagreen, 0.4)],
            linewidth = 1.0, linestyle = :dot)
    scatter!(ax, [T_MD, T_MD_EM, T_MD_SM], [0.0, 0.0, 0.0];
             color = [:steelblue4, :darkorange2, :seagreen],
             marker = :circle, markersize = 10, strokecolor = :black, strokewidth = 1)

    xlims!(ax, (0.0, T_MD * 1.15))

    axislegend(ax,
        [l_sfc, l_em, l_sm],
        [L"p = 0 \; \mathrm{dbar} \; (\mathrm{surface})",
         latexstring(@sprintf("p = %.0f", P_EM), "\\;\\mathrm{dbar}\\;(H=212\\,\\mathrm{m},\\,\\mathrm{EM})"),
         latexstring(@sprintf("p = %.0f", P_SM), "\\;\\mathrm{dbar}\\;(H=384\\,\\mathrm{m},\\,\\mathrm{SM})")];
        position = :rb, labelsize = 9, framevisible = false, patchsize = (16, 8))

    outfile = joinpath(FIGURE_DIR, "thermal_expansion_coefficient_TEOS10.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
end
