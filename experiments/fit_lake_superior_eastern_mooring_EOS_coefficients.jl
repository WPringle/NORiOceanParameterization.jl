#####
##### Fit Roquet-form (SecondOrderSeawaterPolynomial) EOS coefficients to GSW/TEOS-10
##### density for Lake Superior's near-freshwater regime
#####
# The published Roquet et al. (2015) coefficient sets (:Cabbeling,
# :CabbelingThermobaricity, ... in SeawaterPolynomials.SecondOrderSeawaterPolynomials)
# were optimized against horizontal density gradients over the full range of *oceanic*
# temperature/salinity encountered by TEOS-10. Lake Superior is nearly fresh
# (S_A ≈ 0.05 g/kg, held constant) and stays within 0-4.5 °C, straddling the freshwater
# temperature of maximum density (T_MD ≈ 3.98 °C at the surface, decreasing with
# pressure) -- exactly the regime cabbeling/thermobaricity are meant to capture, so the
# literature coefficients are the wrong shape here.
#
# This script re-fits the same polynomial *forms* directly against `gsw_rho` (TEOS-10)
# over the lake's actual T/S/Z domain:
#
#   Cabbeling (no Z term): ordinary least squares at the surface only (Z = 0), since
#     thermobaricity is deliberately excluded from this form.
#
#   CabbelingThermobaricity (adds a Θ·Z term): R₁₀₀, R₀₁₀, R₀₂₀ are *not* re-fit here —
#     they're carried over unchanged from the Cabbeling fit above. Only R₀₁₁ is fit,
#     against the depth-dependence of the temperature of maximum density (T_MD) itself.
#
#     This is deliberate, not a shortcut: an ordinary least-squares fit of *absolute*
#     density over the full (Θ, Z) grid was tried first and produced unusable
#     coefficients (implied T_MD of -18.5 °C at the surface — no density maximum at all
#     within 0-4.5 °C). The reason is that pure pressure compression changes density by
#     ~2 kg/m³ over 0-400 m *independent of Θ* (checked directly against gsw_rho), which
#     is ~30x larger than the actual thermobaric signal (the change in Θ-sensitivity of
#     density with depth, ~0.06 kg/m³ over the same range). This polynomial family has no
#     plain-Z term to absorb that compression trend (same limitation as the literature
#     coefficient sets), so an unweighted absolute-density fit lets the large,
#     Θ-independent compression trend hijack the Θ·Z term meant to represent the much
#     smaller thermobaric effect. Fitting R₀₁₁ directly against the T_MD(Z) trend isolates
#     the effect this term is actually meant to capture.
#
# Usage: julia --project=. fit_lake_superior_eastern_mooring_EOS_coefficients.jl
#####

using GibbsSeaWater
using SeawaterPolynomials
using SeawaterPolynomials.SecondOrderSeawaterPolynomials
using Statistics
using Printf

const S_A  = 0.05                     # g/kg — Lake Superior Absolute Salinity (constant)
const ρ₀   = 999.8                    # kg/m³ — model reference density (fresh 4 °C water)
const lat  = 47.0 + 32.2 / 60.0       # eastern mooring latitude (47° 32.2' N)
const Θmin, Θmax = 0.0, 4.5           # °C — Conservative Temperature range
const Zmax = 400.0                    # m — max lake depth considered

const Θ_grid = range(Θmin, Θmax, length = 451)   # 0.01 °C resolution
const Z_grid = range(-Zmax, 0.0, length = 81)    # 5 m resolution (Z=0 surface, negative down)

# Sea pressure (dbar) from geopotential height Z (m, negative down), at the mooring latitude.
p_from_Z(Z) = gsw_p_from_z(Z, lat)

# Design-matrix columns for ρ' = R₁₀₀·Sᴬ + R₀₁₀·Θ + R₀₂₀·Θ² - R₀₁₁·Θ·Z
design_row(Θ, Z) = (S_A, Θ, Θ^2, -Θ * Z)

target_ρ(Θ, Z) = gsw_rho(S_A, Θ, p_from_Z(Z))

function least_squares_fit(Θs, Zs, cols)
    n = length(Θs)
    A = Matrix{Float64}(undef, n, length(cols))
    b = Vector{Float64}(undef, n)
    for k in 1:n
        row = design_row(Θs[k], Zs[k])
        A[k, :] = [row[c] for c in cols]
        b[k]    = target_ρ(Θs[k], Zs[k]) - ρ₀
    end
    coeffs = A \ b
    resid  = A * coeffs .- b
    return coeffs, sqrt(mean(resid .^ 2)), maximum(abs, resid)
end

# Density error using the literature Roquet coefficients (for comparison).
function literature_error(Θs, Zs, poly)
    errs = [SeawaterPolynomials.ρ′(Θs[k], S_A, Zs[k],
                SeawaterPolynomials.BoussinesqEquationOfState(poly, ρ₀)) -
            (target_ρ(Θs[k], Zs[k]) - ρ₀) for k in eachindex(Θs)]
    return sqrt(mean(errs .^ 2)), maximum(abs, errs)
end

# Temperature of maximum density implied by a fitted polynomial at a given Z:
# ρ' = R₁₀₀Sᴬ + R₀₁₀Θ + R₀₂₀Θ² - R₀₁₁ΘZ  ⟹  dρ'/dΘ = R₀₁₀ + 2R₀₂₀Θ - R₀₁₁Z = 0
T_MD_from_fit(R₀₁₀, R₀₂₀, R₀₁₁, Z) = (R₀₁₁ * Z - R₀₁₀) / (2 * R₀₂₀)

# Reference T_MD straight from GSW, by direct search over the Θ grid.
function T_MD_from_gsw(Z)
    ρs = [target_ρ(Θ, Z) for Θ in Θ_grid]
    return Θ_grid[argmax(ρs)]
end

println("═"^78)
println("Cabbeling fit (surface only, Z = 0)")
println("═"^78)
Θs_surf = collect(Θ_grid)
Zs_surf = zeros(length(Θs_surf))
coeffs_cab, rms_cab, max_cab = least_squares_fit(Θs_surf, Zs_surf, (1, 2, 3))
R₁₀₀_cab, R₀₁₀_cab, R₀₂₀_cab = coeffs_cab
rms_lit_cab, max_lit_cab = literature_error(Θs_surf, Zs_surf, RoquetSeawaterPolynomial(:Cabbeling))
@printf("  fitted:     R₁₀₀=%.6e  R₀₁₀=%.6e  R₀₂₀=%.6e\n", R₁₀₀_cab, R₀₁₀_cab, R₀₂₀_cab)
@printf("  fit error:       rms=%.4e kg/m³  max=%.4e kg/m³\n", rms_cab, max_cab)
@printf("  literature error: rms=%.4e kg/m³  max=%.4e kg/m³\n", rms_lit_cab, max_lit_cab)
@printf("  T_MD (fit, GSW) at Z=0 m  : %.3f °C, %.3f °C\n",
        T_MD_from_fit(R₀₁₀_cab, R₀₂₀_cab, 0.0, 0.0), T_MD_from_gsw(0.0))

println()
println("═"^78)
println("CabbelingThermobaricity fit: R₀₁₁ calibrated to the T_MD(Z) trend")
println("═"^78)
R₁₀₀_thb, R₀₁₀_thb, R₀₂₀_thb = R₁₀₀_cab, R₀₁₀_cab, R₀₂₀_cab   # carried over from surface fit

# T_MD(Z) from GSW at a set of depths spanning the column.
TMD_gsw = T_MD_from_gsw.(Z_grid)

# Θ*(Z) = (R₀₁₁·Z - R₀₁₀) / (2·R₀₂₀)  ⟹  Θ*(Z) - Θ*(0) = (R₀₁₁ / (2·R₀₂₀))·Z
# Fit the slope (R₀₁₁ / 2R₀₂₀) by least squares against the GSW T_MD(Z) trend.
Θ0_gsw   = T_MD_from_gsw(0.0)
slope    = collect(Z_grid) \ (TMD_gsw .- Θ0_gsw)
R₀₁₁_thb = slope * 2 * R₀₂₀_thb

ρ′_thb(Θ, Z) = R₁₀₀_thb * S_A + R₀₁₀_thb * Θ + R₀₂₀_thb * Θ^2 - R₀₁₁_thb * Θ * Z
rms_thb, max_thb = let
    errs = [ρ′_thb(Θ, Z) - (target_ρ(Θ, Z) - ρ₀) for Θ in Θ_grid, Z in Z_grid]
    sqrt(mean(errs .^ 2)), maximum(abs, errs)
end
rms_lit_thb, max_lit_thb = literature_error(vec([Θ for Θ in Θ_grid, _ in Z_grid]),
                                             vec([Z for _ in Θ_grid, Z in Z_grid]),
                                             RoquetSeawaterPolynomial(:CabbelingThermobaricity))
@printf("  fitted:     R₁₀₀=%.6e  R₀₁₀=%.6e  R₀₂₀=%.6e  R₀₁₁=%.6e\n",
        R₁₀₀_thb, R₀₁₀_thb, R₀₂₀_thb, R₀₁₁_thb)
@printf("  absolute-density error: rms=%.4e kg/m³  max=%.4e kg/m³ (dominated by the\n", rms_thb, max_thb)
println("    unrepresented Θ-independent compression trend, not by T_MD accuracy — see header)")
@printf("  literature coefficients' absolute-density error: rms=%.4e kg/m³  max=%.4e kg/m³\n",
        rms_lit_thb, max_lit_thb)
println("  T_MD (fit vs GSW) by depth:")
for Z in (0.0, -100.0, -200.0, -300.0, -400.0)
    @printf("    Z=%6.0f m : fit=%.3f °C   gsw=%.3f °C\n",
            Z, T_MD_from_fit(R₀₁₀_thb, R₀₂₀_thb, R₀₁₁_thb, Z), T_MD_from_gsw(Z))
end

println()
println("═"^78)
println("Ready-to-paste Julia literals")
println("═"^78)
@printf("""
const CABBELING_POLY = SecondOrderSeawaterPolynomial{Float64}(
    R₁₀₀ = %.6e, R₀₁₀ = %.6e, R₀₂₀ = %.6e)

const CABBELING_THERMOBARIC_POLY = SecondOrderSeawaterPolynomial{Float64}(
    R₁₀₀ = %.6e, R₀₁₀ = %.6e, R₀₂₀ = %.6e, R₀₁₁ = %.6e)
""", R₁₀₀_cab, R₀₁₀_cab, R₀₂₀_cab, R₁₀₀_thb, R₀₁₀_thb, R₀₂₀_thb, R₀₁₁_thb)
