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

FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

# In-situ temperature of maximum density at the surface (same construction as
# the M_sfc scripts, but keeping it in Celsius here since this is a T-axis plot).
const T_MD = gsw_t_from_ct(S_lake, gsw_ct_maxdensity(S_lake, P_SFC), P_SFC)

T_range = range(0.0, T_MD; length = 400)
α       = [gsw_alpha_wrt_t_exact(S_lake, T, P_SFC) for T in T_range]
const α0 = gsw_alpha_wrt_t_exact(S_lake, 0.0, P_SFC)   # α at T = 0 °C, called out on the y-axis

with_theme(theme_latexfonts()) do
    fig = Figure(size = (340, 280), fontsize = 12, figure_padding = (6, 36, 6, 6))

    # Round background ticks plus the exact α(0 °C) value called out explicitly.
    ytick_vals = sort(unique([α0, -5e-5, -2.5e-5, 0.0]))
    ytick_labs = [v == 0.0 ? L"0" : latexstring(sci_latex(v)) for v in ytick_vals]

    # Round background ticks plus the exact T_MD value called out explicitly.
    xtick_vals = sort(unique(vcat(0:1:floor(Int, T_MD), T_MD)))
    xtick_labs = [v == T_MD ? latexstring("T_{\\mathrm{MD}} = ", round(T_MD; digits = 2)) :
                              latexstring(round(Int, v))
                  for v in xtick_vals]

    ax = Axis(fig[1, 1];
              xlabel = L"T \; (^\circ\mathrm{C})",
              ylabel = L"\alpha \; (\mathrm{K^{-1}})",
              xticks = (xtick_vals, xtick_labs),
              yticks = (ytick_vals, ytick_labs))

    hlines!(ax, [0.0]; color = :gray60, linewidth = 1.0, linestyle = :dash)
    lines!(ax, T_range, α; color = :steelblue4, linewidth = 2.5)
    scatter!(ax, [T_MD], [0.0]; color = :firebrick, marker = :circle, markersize = 10)

    xlims!(ax, (0.0, T_MD * 1.15))

    outfile = joinpath(FIGURE_DIR, "thermal_expansion_coefficient_TEOS10.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
end
