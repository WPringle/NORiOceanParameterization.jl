#####
##### Lake Superior Eastern Mooring — column model forced by Stannard Rock GLEN observations
##### DIRECTIONAL (UV-stress) variant
#####
# Identical to lake_superior_eastern_mooring_GLEN_forced.jl EXCEPT the surface
# momentum flux is split into x (Qᵁ) and y (Qᵛ) components using the measured
# wind_direction, rather than applying the entire stress magnitude in x.
#
# Wind stress decomposition:
#   wind_direction (θ) is meteorological — the direction the wind blows FROM,
#   measured clockwise from true north (0°=N, 90°=E, 180°=S, 270°=W).  The downwind
#   stress unit vector is therefore (êₓ, êᵧ) = (−sinθ, −cosθ).  With the
#   Oceananigans top-flux convention (BC value = −τ_component / ρ₀):
#       Qᵁ = (momentum_flux / ρ₀) · sinθ = τ_mag_kin · êₓ      [m²/s²]
#       Qᵛ = (momentum_flux / ρ₀) · cosθ = τ_mag_kin · êᵧ      [m²/s²]
#   (Reduces to the single-component sister script: wind from the west, θ = 270°,
#    gives Qᵁ = −mom/ρ₀, Qᵛ = 0 — a purely eastward stress.)
#
#   Gap-filling of the direction is done on its Cartesian unit components
#   (sinθ, cosθ), NOT on the degrees, so it respects the 0°/360° cyclicity
#   (linearly averaging degrees would turn 359° & 1° into 180°).  The stress
#   magnitude (momentum_flux) is filled separately as a scalar.
#
# Heat-flux forcing is unchanged from the sister script:
#   Q_net = SHF + LHF - SW↓ - LW↓ + LW↑(T_sfc)       [W/m²]
# LW↑ = ε·σ·T_sfc⁴ is added online from the model surface temperature.
#
# Forcing source flag (`direct` | `coare_wind`):
#   direct      : measured eddy-covariance momentum_flux / sensible_heat_flux /
#                 latent_heat_flux.
#   coare_wind  : the *_coare_wind variants — momentum/SHF/LHF re-derived from the
#                 measured wind speed via the COARE bulk algorithm.
#   The downwelling SW/LW radiation AND the wind_direction are identical for both
#   sources (coare_wind only changes the stress magnitude, not its direction).
#
# Equation of state, selectable via the `eos` argument:
#   cabbeling             : SecondOrderSeawaterPolynomial (Roquet et al. 2015 form),
#                            quadratic-in-Θ cabbeling term only, no thermobaricity.
#                            Coefficients re-fit for Lake Superior's near-freshwater
#                            regime (see fit_lake_superior_eastern_mooring_EOS_coefficients.jl)
#                            rather than the literature oceanic coefficients that
#                            RoquetEquationOfState(:Cabbeling) would otherwise use. (default)
#   cabbelingthermobaric  : as above, plus a Θ·Z term capturing how T_MD shifts with depth.
#   teos10cabbeling       : the full (55-term) TEOS10EquationOfState, but with the
#                            geopotential height Z clamped to 0 in every density/expansion
#                            evaluation — i.e. density always computed at surface pressure.
#                            This reproduces the exact TEOS-10 cabbeling curve (no
#                            polynomial-fit residual at all) with thermobaricity excluded
#                            by construction, rather than approximated by a reduced
#                            polynomial. See the SurfaceTEOS10EquationOfState definition
#                            below.
#
# Usage:
#   julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl <winter_year> [closure_name] [forcing_source] [eos]
#   e.g.  julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl 2009
#         julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl 2009 CATKE
#         julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl 2009 CATKE coare_wind
#         julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl 2009 coare_wind   # all closures
#         julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl 2009 CATKE cabbelingthermobaric
#         julia lake_superior_eastern_mooring_GLEN_forced_UVstress.jl 2009 CATKE teos10cabbeling
#
# Inputs  : figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc
# Outputs : data/TURB_outputs/eastern_mooring_GLEN_forced/winter<YEAR>/
#               <closure>[_coare_wind]_UVstress_EOS<variant>_winter<YEAR>.jld2      (averaged profiles)
#               <closure>[_coare_wind]_UVstress_EOS<variant>_winter<YEAR>_LMO.jld2  (Monin-Obukhov length)
#####

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    TKEDissipationVerticalDiffusivity, TKEDissipationEquations,
    CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using Oceananigans.Operators: ℑxᶜᵃᵃ, ℑyᵃᶜᵃ, ∂zᶠᶜᶠ, ∂zᶜᶠᶠ
using Oceananigans.BuoyancyFormulations: ∂z_b
using Oceananigans.AbstractOperations: KernelFunctionOperation
using JLD2
using NCDatasets
using Dates
using Statistics
using Printf
using SeawaterPolynomials
using SeawaterPolynomials.SecondOrderSeawaterPolynomials
using SeawaterPolynomials.TEOS10: TEOS10SeawaterPolynomial
using SeawaterPolynomials: AbstractSeawaterPolynomial
import SeawaterPolynomials: ρ′, thermal_sensitivity, haline_sensitivity, with_float_type
using GibbsSeaWater
using NORiOceanParameterization
using NORiOceanParameterization.Implementation

# ── Command-line arguments ────────────────────────────────────────────────────
length(ARGS) < 1 &&
    error("Usage: julia $(basename(@__FILE__)) <winter_year> [closure_name] [forcing_source] [eos]\n" *
          "  winter_year   : e.g. 2009, 2010, 2011, 2014\n" *
          "  closure_name  : kepsilon | kepsilon_Rist035 | CATKE | CATKE_highRi  (default: all)\n" *
          "  forcing_source: direct | coare_wind  (default: direct)\n" *
          "  eos           : cabbeling | cabbelingthermobaric | teos10cabbeling  (default: cabbeling)")

const winter_year = parse(Int, ARGS[1])

# Remaining args (any order): a closure name, a forcing-source flag, and/or an EOS flag.
# Parsed inside a function so the assignments are not caught by top-level soft-scope
# rules (which would silently treat them as loop-locals and ignore the flag).
function parse_extra_args(extra)
    forcing_choices = ("direct", "coare", "coare_wind")
    eos_thermobaric_choices = ("cabbelingthermobaric", "cabbeling_thermobaric", "thermobaric")
    eos_cabbeling_choices   = ("cabbeling",)
    eos_teos10_choices      = ("teos10cabbeling", "teos10-cabbeling", "teos10_cabbeling", "surfaceteos10")
    run_closure    = nothing
    forcing_source = "direct"
    eos_variant    = "cabbeling"
    for a in extra
        al = lowercase(strip(a))
        if al in forcing_choices
            forcing_source = al == "coare" ? "coare_wind" : al
        elseif al in eos_thermobaric_choices
            eos_variant = "cabbelingthermobaric"
        elseif al in eos_cabbeling_choices
            eos_variant = "cabbeling"
        elseif al in eos_teos10_choices
            eos_variant = "teos10cabbeling"
        else
            run_closure = a
        end
    end
    return run_closure, forcing_source, eos_variant
end

const RUN_CLOSURE, FORCING_SOURCE, EOS_VARIANT = parse_extra_args(ARGS[2:end])
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"    # forcing-source filename tag
const UV_TAG  = "_UVstress"                                        # directional-stress filename tag
const EOS_TAG = Dict("cabbeling"            => "_EOScabbeling",
                      "cabbelingthermobaric" => "_EOScabbelingthermobaric",
                      "teos10cabbeling"      => "_EOSteos10cabbeling")[EOS_VARIANT]
@info "Forcing source = $FORCING_SOURCE"
@info "Equation of state = $EOS_VARIANT"

# ── CSV: look up isothermal (start) date ──────────────────────────────────────
const CSV_FILE = joinpath(@__DIR__, "..", "figure_data",
    "lake_superior_eastern_mooring",
    "lake_superior_eastern_mooring_winter_start_dates.csv")

function read_isothermal_date(csv_file, year)
    return open(csv_file) do f
        readline(f)   # skip header
        for line in eachline(f)
            isempty(strip(line)) && continue
            parts = split(line, ',')
            length(parts) >= 2 || continue
            wy = tryparse(Int, strip(parts[1]))
            wy == year || continue
            return DateTime(Date(strip(parts[2])))  # "YYYY-MM-DD" → midnight DateTime
        end
        nothing
    end
end

t_iso = read_isothermal_date(CSV_FILE, winter_year)
isnothing(t_iso) &&
    error("Winter year $winter_year not found in $CSV_FILE")
@info "Winter $winter_year: isothermal (start) date = $t_iso"

# ── Constants ─────────────────────────────────────────────────────────────────
const ρ₀      = 999.8      # kg/m³ — reference density (fresh 4 °C water)
const cₚ      = 4182.0     # J/(kg·K) — freshwater specific heat
const ρ_air   = 1.225      # kg/m³  (used to convert momentum flux ↔ u*)
const g_conv   = 9.80665   # m/s²
const S_lake   = 0.05      # g/kg (Absolute Salinity, constant)
const T_insitu = 4.0       # °C  (initial in-situ temperature)

# ── Equation of state (Lake Superior-calibrated) ──────────────────────────────
# SecondOrderSeawaterPolynomial coefficients re-fit against GSW/TEOS-10 (gsw_rho) for
# S_A = 0.05 g/kg, Θ ∈ [0, 4.5] °C, over Z ∈ [-400, 0] m — the literature Roquet et al.
# (2015) :Cabbeling / :CabbelingThermobaricity coefficients are tuned for oceanic T/S
# ranges and fit this near-freshwater regime poorly (surface RMS error ~0.4 kg/m³
# vs ~1e-4 kg/m³ for the refit). See fit_lake_superior_eastern_mooring_EOS_coefficients.jl.
#
# CABBELING_THERMOBARIC_POLY carries over R₁₀₀/R₀₁₀/R₀₂₀ unchanged from the surface
# (Cabbeling) fit and calibrates only R₀₁₁ against the actual T_MD(Z) trend from GSW —
# an unweighted least-squares fit of absolute density over the full (Θ,Z) grid is
# dominated by the ~2 kg/m³ Θ-independent pressure-compression trend (this polynomial
# family has no plain-Z term to absorb it) and produces an unphysical fit with no
# density maximum within 0-4.5 °C.
const CABBELING_POLY = SecondOrderSeawaterPolynomial{Float64}(
    R₁₀₀ = 1.692692e+00, R₀₁₀ = 6.322260e-02, R₀₂₀ = -7.602699e-03)

const CABBELING_THERMOBARIC_POLY = SecondOrderSeawaterPolynomial{Float64}(
    R₁₀₀ = 1.692692e+00, R₀₁₀ = 6.322260e-02, R₀₂₀ = -7.602699e-03, R₀₁₁ = -3.272378e-05)

# ── TEOS10-at-the-surface equation of state ("teos10cabbeling") ───────────────
# Wraps the full 55-term TEOS10EquationOfState but clamps the geopotential height Z to
# 0 in every density/expansion evaluation, i.e. density is always computed at surface
# pressure regardless of actual depth. This reproduces the exact TEOS-10 cabbeling curve
# (no polynomial-fit residual) with thermobaricity excluded by construction, rather than
# approximated by a reduced Roquet polynomial as CABBELING_POLY does above.
struct SurfaceTEOS10SeawaterPolynomial{FT} <: AbstractSeawaterPolynomial end
SurfaceTEOS10SeawaterPolynomial(FT=Float64) = SurfaceTEOS10SeawaterPolynomial{FT}()
Base.eltype(::SurfaceTEOS10SeawaterPolynomial{FT}) where FT = FT
with_float_type(FT, ::SurfaceTEOS10SeawaterPolynomial) = SurfaceTEOS10SeawaterPolynomial{FT}()

const SurfaceTEOS10EOS = SeawaterPolynomials.BoussinesqEquationOfState{<:SurfaceTEOS10SeawaterPolynomial}

_teos10(eos::SurfaceTEOS10EOS) =
    SeawaterPolynomials.BoussinesqEquationOfState(
        TEOS10SeawaterPolynomial{typeof(eos.reference_density)}(), eos.reference_density)

ρ′(Θ, Sᴬ, Z, eos::SurfaceTEOS10EOS)                 = ρ′(Θ, Sᴬ, zero(Z), _teos10(eos))
thermal_sensitivity(Θ, Sᴬ, Z, eos::SurfaceTEOS10EOS) = thermal_sensitivity(Θ, Sᴬ, zero(Z), _teos10(eos))
haline_sensitivity(Θ, Sᴬ, Z, eos::SurfaceTEOS10EOS)  = haline_sensitivity(Θ, Sᴬ, zero(Z), _teos10(eos))

SurfaceTEOS10EquationOfState(FT=Float64; reference_density) =
    SeawaterPolynomials.BoussinesqEquationOfState(SurfaceTEOS10SeawaterPolynomial{FT}(), convert(FT, reference_density))

const LAKE_EOS = if EOS_VARIANT == "cabbeling"
    SeawaterPolynomials.BoussinesqEquationOfState(CABBELING_POLY, ρ₀)
elseif EOS_VARIANT == "cabbelingthermobaric"
    SeawaterPolynomials.BoussinesqEquationOfState(CABBELING_THERMOBARIC_POLY, ρ₀)
else # "teos10cabbeling"
    SurfaceTEOS10EquationOfState(reference_density = ρ₀)
end

# Radiation constants for net SW and LW
const albedo_sw = 0.08     # broadband shortwave albedo of fresh water surface
const ε_water   = 0.98     # longwave emissivity of water
const σ_SB      = 5.67e-8  # W/(m²·K⁴) — Stefan-Boltzmann constant

# ── Load GLEN half-hourly data ────────────────────────────────────────────────
# Gap-filled product: momentum/heat fluxes are provided directly.  Choose between
# the measured (`direct`) and COARE-bulk-from-wind (`coare_wind`) flux variants.
# wind_direction is shared (no *_coare_wind variant): coare_wind changes only the
# stress magnitude, not its direction.
const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"

# Variable names that differ between the two forcing sources.
const VAR_SUFFIX = FORCING_SOURCE == "direct" ? "" : "_coare_wind"
const LHF_VAR    = "latent_heat_flux"   * VAR_SUFFIX
const SHF_VAR    = "sensible_heat_flux" * VAR_SUFFIX
const MOM_VAR    = "momentum_flux"      * VAR_SUFFIX

glen_times_all, lhf_all, shf_all, sw_all, lw_all, mom_all, wspd_all, wdir_all =
    NCDataset(GLEN_FILE) do ds
        times = DateTime.(ds["time"][:])
        function load_var(name)
            v = ds[name][:]
            [ismissing(x) ? missing : Float64(x) for x in v]
        end
        times,
        load_var(LHF_VAR),
        load_var(SHF_VAR),
        load_var("downwelling_shortwave_flux"),
        load_var("downwelling_longwave_flux"),
        load_var(MOM_VAR),
        load_var("wind_speed"),
        load_var("wind_direction")
    end
@info "GLEN dataset: $(glen_times_all[1]) → $(glen_times_all[end]) ($(length(glen_times_all)) half-hourly records)"
@info "Flux variables: SHF=$SHF_VAR  LHF=$LHF_VAR  τ=$MOM_VAR  (direction = wind_direction)"

# ── Slice to simulation window: [t_iso, t_iso + 60 days] ─────────────────────
const sim_days = 60
t_end = t_iso + Day(sim_days)
idx   = findall(t -> t_iso <= t <= t_end, glen_times_all)
isempty(idx) && error("No GLEN data in window $t_iso → $t_end")
@info "GLEN window: $(glen_times_all[idx[1]]) → $(glen_times_all[idx[end]]) ($(length(idx)) records)"

glen_t   = glen_times_all[idx]
lhf_raw  = lhf_all[idx]
shf_raw  = shf_all[idx]
sw_raw   = sw_all[idx]
lw_raw   = lw_all[idx]
mom_raw  = mom_all[idx]
ws_raw   = wspd_all[idx]
wdir_raw = wdir_all[idx]

# ── Fill missing values (linear interpolation, flat extrapolation at edges) ───
function fill_missing_linear(vals)
    n      = length(vals)
    filled = Vector{Float64}(undef, n)
    valid  = findall(!ismissing, vals)
    isempty(valid) && error("Cannot fill: all values are missing")
    n_miss = n - length(valid)
    n_miss > 0 && @warn "  Filling $n_miss / $n missing values by linear interpolation"
    vx = Float64.(valid)
    vy = Float64.(vals[valid])
    for k in 1:n
        if !ismissing(vals[k])
            filled[k] = Float64(vals[k])
        else
            i = searchsortedlast(vx, Float64(k))
            if i == 0
                filled[k] = vy[1]
            elseif i == length(vx)
                filled[k] = vy[end]
            else
                α = (k - vx[i]) / (vx[i+1] - vx[i])
                filled[k] = vy[i] + α * (vy[i+1] - vy[i])
            end
        end
    end
    return filled
end

# The gap-fill is complete over the winter windows, so these should not need to
# interpolate anything; they remain as a safety net (and warn if they ever fire).
# Scalars (magnitudes) are safe to interpolate linearly.
lhf_f = fill_missing_linear(lhf_raw)
shf_f = fill_missing_linear(shf_raw)
sw_f  = fill_missing_linear(sw_raw)
lw_f  = fill_missing_linear(lw_raw)
mom_f = fill_missing_linear(mom_raw)
ws_f  = fill_missing_linear(ws_raw)

# Wind direction is CYCLICAL — linearly interpolating degrees is wrong across the
# 0°/360° wrap (e.g. 359° and 1° would average to 180° instead of 0°).  Instead,
# fill the Cartesian unit-vector components (êₓ = sinθ, êᵧ = cosθ), which are
# continuous, then renormalise back to a unit vector.  θ is the meteorological
# from-direction (cw from north).
ex_raw = Union{Missing,Float64}[ismissing(d) ? missing : sind(d) for d in wdir_raw]
ey_raw = Union{Missing,Float64}[ismissing(d) ? missing : cosd(d) for d in wdir_raw]
ex_f   = fill_missing_linear(ex_raw)
ey_f   = fill_missing_linear(ey_raw)
enorm  = @. sqrt(ex_f^2 + ey_f^2)
ex_f   = @. ifelse(enorm > 0, ex_f / enorm, 0.0)   # unit êₓ = sinθ
ey_f   = @. ifelse(enorm > 0, ey_f / enorm, 0.0)   # unit êᵧ = cosθ

# Reconstructed direction (degrees, for diagnostics): θ = atan2(sinθ, cosθ)
wdir_f = @. mod(atand(ex_f, ey_f), 360.0)

# ── Build flux time series ────────────────────────────────────────────────────
# Simulation time axis: seconds since t_iso
const t_forcing = Float64[Dates.value(t - t_iso) / 1000.0 for t in glen_t]

# Pre-computed heat flux terms (W/m²), positive upward = lake cooling.
# LW_up = ε·σ·T_sfc⁴ is excluded here because it depends on the live surface
# temperature; it is added inside the discrete boundary-condition function below.
#   SW_net = (1 - albedo) · SW↓   — net absorbed shortwave (into lake → negative contribution)
#   Q_precomp = SHF + LHF - SW_net - LW↓
const SW_net_vals     = @. (1.0 - albedo_sw) * sw_f
const Q_precomp_Wm2   = @. shf_f + lhf_f - SW_net_vals - ε_water * lw_f
const Q_precomp_kin   = @. Q_precomp_Wm2 / (ρ₀ * cₚ)   # °C·m/s

# Directional kinematic momentum flux (m²/s²), split into x/y from wind_direction.
# θ = wind_direction is meteorological (direction wind blows FROM, cw from north);
# downwind stress unit vector = (−sinθ, −cosθ); top-flux BC value = −τ_c / ρ₀:
#     Qᵁ = (mom_f / ρ₀) · sinθ = τ_mag_kin · êₓ
#     Qᵛ = (mom_f / ρ₀) · cosθ = τ_mag_kin · êᵧ
# using the gap-filled unit-vector components êₓ, êᵧ built above (cyclicity-safe).
# (Negative Qᵁ/Qᵛ = momentum into the lake along +x/+y respectively.)
# Friction velocity (diagnostic only) follows from u* = sqrt(τ / ρ_air).
const τ_mag_kin  = @. mom_f / ρ₀                 # kinematic stress magnitude (≥ 0)
const Qu_vals    = @. τ_mag_kin * ex_f
const Qv_vals    = @. τ_mag_kin * ey_f
const ustar_vals = @. sqrt(max(mom_f, 0.0) / ρ_air)

@info "Forcing summary (60-day window):"
@info "  Q_precomp (W/m²): mean=$(round(mean(Q_precomp_Wm2), digits=1))" *
      "  min=$(round(minimum(Q_precomp_Wm2), digits=1))  max=$(round(maximum(Q_precomp_Wm2), digits=1))"
@info "  LW_up at 4 °C (W/m²): $(round(ε_water * σ_SB * (4.0 + 273.15)^4, digits=1))  (added online from T_sfc)"
@info "  u*     (m/s)  : mean=$(round(mean(ustar_vals), digits=4))  max=$(round(maximum(ustar_vals), digits=4))"
@info "  U_meas (m/s)  : mean=$(round(mean(ws_f), digits=2))  max=$(round(maximum(ws_f), digits=2))"
@info "  wind_dir (°)  : mean=$(round(mean(wdir_f), digits=1))  [meteorological, from-direction]"
@info "  |Qᵁ| (m²/s²)  : mean=$(round(mean(abs.(Qu_vals)), digits=6))  max=$(round(maximum(abs.(Qu_vals)), digits=6))"
@info "  |Qᵛ| (m²/s²)  : mean=$(round(mean(abs.(Qv_vals)), digits=6))  max=$(round(maximum(abs.(Qv_vals)), digits=6))"

# ── Linear interpolation helper (no external package required) ────────────────
# Flat extrapolation outside [t_arr[1], t_arr[end]].
@inline function interp_linear(t_arr, v_arr, t::Float64)
    i = searchsortedlast(t_arr, t)
    i == 0             && return v_arr[1]
    i == length(t_arr) && return v_arr[end]
    α = (t - t_arr[i]) / (t_arr[i+1] - t_arr[i])
    return v_arr[i] + α * (v_arr[i+1] - v_arr[i])
end

# Heat flux BC — discrete form so it can read the live surface temperature.
# Full Q_net = Q_precomp(t) + LW_up(T_sfc)
#   LW_up = ε·σ·(T_sfc + 273.15)⁴  (upward thermal emission from lake, positive upward)
# T_sfc is Conservative Temperature in °C at the top model cell.
@inline function Qᵀ_obs(i, j, grid, clock, model_fields, p)
    T_sfc   = @inbounds model_fields.T[i, j, p.Nz]
    LW_up   = p.ε_water * p.σ_SB * (T_sfc + 273.15)^4 / (p.ρ₀ * p.cₚ)
    return interp_linear(p.t_forcing, p.Q_precomp_kin, Float64(clock.time)) + LW_up
end

# Momentum flux BCs — continuous form; no state dependence.
@inline Qᵁ_obs(t, p) = interp_linear(p.t_forcing, p.Qu_vals, t)   # x-stress
@inline Qᵛ_obs(t, p) = interp_linear(p.t_forcing, p.Qv_vals, t)   # y-stress

# ── Grid ──────────────────────────────────────────────────────────────────────
model_architecture = CPU()
const Nz = 53      # dz ≈ 4 m
const Lz = 212.0   # m  (eastern mooring depth)

grid = RectilinearGrid(model_architecture,
                       topology = (Flat, Flat, Bounded),
                       size     = Nz,
                       halo     = 3,
                       z        = (-Lz, 0))

# ── Location & Coriolis ──────────────────────────────────────────────────────
const lat = 47.0 + 32.2 / 60.0   # 47° 32.2' N (eastern mooring)
const Ω   = 7.2921150e-5         # rad/s
const f₀  = 2 * Ω * sind(lat)

# ── Initial temperature (uniform 4 °C, converted to Conservative Temperature) ─
function Θ_conservative(z::Float64)
    p_dbar = ρ₀ * g_conv * abs(z) / 1e4
    return gsw_ct_from_t(S_lake, T_insitu, p_dbar)
end

# ── Closure definitions ───────────────────────────────────────────────────────
function make_kepsilon_default()
    TKEDissipationVerticalDiffusivity()
end

function make_kepsilon_Rist035()
    eqs = TKEDissipationEquations(Cᵇϵ⁺ = -0.26, Cᵇϵ⁻ = -0.26)
    TKEDissipationVerticalDiffusivity(; tke_dissipation_equations = eqs)
end

function make_CATKE()
    CATKEVerticalDiffusivity(; mixing_length                    = CATKEMixingLength(),
                               turbulent_kinetic_energy_equation = CATKEEquation())
end

function make_CATKE_highRi()
    CATKEVerticalDiffusivity(; mixing_length                    = CATKEMixingLength(Cˡᵒu = 0.52, Cˡᵒc = 0.64, Cʰⁱc = 0.115),
                               turbulent_kinetic_energy_equation = CATKEEquation())
end

# ── Richardson number diagnostic ──────────────────────────────────────────────
@inline ϕ²_Ri(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function Ri_ccf(i, j, k, grid, u, v, buoyancy, tracers)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ²_Ri, ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ²_Ri, ∂zᶜᶠᶠ, v)
    S²     = ∂z_u² + ∂z_v²
    N²     = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return ifelse(N² == 0, zero(grid), N² / (S² + 1e-11))
end

# ── Model setup ───────────────────────────────────────────────────────────────
function setup_model(closure)
    T_bcs = FieldBoundaryConditions(
                top = FluxBoundaryCondition(Qᵀ_obs,
                          discrete_form = true,
                          parameters = (; t_forcing, Q_precomp_kin,
                                          Nz, ρ₀, cₚ, ε_water, σ_SB)))
    u_bcs = FieldBoundaryConditions(
                top = FluxBoundaryCondition(Qᵁ_obs,
                          parameters = (; t_forcing, Qu_vals)))
    v_bcs = FieldBoundaryConditions(
                top = FluxBoundaryCondition(Qᵛ_obs,
                          parameters = (; t_forcing, Qv_vals)))

    tracers = if closure isa CATKEVerticalDiffusivity
        (:T, :e)
    elseif closure isa TKEDissipationVerticalDiffusivity
        (:T, :e, :ϵ)
    else
        (:T)
    end

    model = HydrostaticFreeSurfaceModel(
        grid                = grid,
        free_surface        = ImplicitFreeSurface(),
        momentum_advection  = WENO(grid = grid),
        tracer_advection    = WENO(grid = grid),
        buoyancy            = SeawaterBuoyancy(
                                  equation_of_state = LAKE_EOS,
				  constant_salinity = S_lake),
        coriolis            = FPlane(f = f₀),
        closure             = closure,
        tracers             = tracers,
        boundary_conditions = (; T = T_bcs, u = u_bcs, v = v_bcs),
    )

    noise(z) = 1e-6 * rand() * exp(z / 8)
    set!(model, T = z -> Θ_conservative(z) + noise(z))
    update_state!(model)
    return model
end

# ── Simulation ────────────────────────────────────────────────────────────────
function run_simulation(model, closure_name, OUTPUT_PATH)
    stop_time = sim_days * days
    Δt        = 5minutes

    simulation = Simulation(model; Δt, stop_time)

    wall_clock = [time_ns()]
    function print_progress(sim)
        @printf("[%05.2f%%] i: %d, t: %s, wall: %s, max|T|: %.3f °C, Δt: %s\n",
            100 * sim.model.clock.time / sim.stop_time,
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.tracers.T),
            prettytime(sim.Δt))
        wall_clock[1] = time_ns()
    end
    simulation.callbacks[:progress] = Callback(print_progress, IterationInterval(500))

    # Approximate ice formation: clamp T ≥ 0 °C (Conservative Temperature)
    function clamp_freezing!(sim)
        clamp!(parent(sim.model.tracers.T), 0.0, Inf)
    end
    simulation.callbacks[:freeze] = Callback(clamp_freezing!, IterationInterval(1))

    # ── Averaged profile outputs ──
    u, v, _ = model.velocities
    T = model.tracers.T

    ubar  = Field(Average(u, dims = (1, 2)))
    vbar  = Field(Average(v, dims = (1, 2)))
    Tbar  = Field(Average(T, dims = (1, 2)))
    Ri_op = KernelFunctionOperation{Center, Center, Face}(
                Ri_ccf, grid, u, v, model.buoyancy, model.tracers)
    Ribar = Field(Average(Ri_op, dims = (1, 2)))
    N²_op = KernelFunctionOperation{Center, Center, Face}(
                ∂z_b, grid, model.buoyancy, model.tracers)
    N²bar = Field(Average(N²_op, dims = (1, 2)))

    # ── Monin–Obukhov length timeseries ──────────────────────────────────────
    κ_vonK = 0.41
    L_MO_series      = Float64[]
    t_series         = Float64[]

    QT_series  = Float64[]
    Qu_series  = Float64[]
    Qv_series  = Float64[]

    function compute_L_MO(sim)
        t_now  = sim.model.clock.time
        T_sfc  = Array(interior(sim.model.tracers.T, 1, 1, Nz))[1]
        α_sfc  = gsw_alpha(S_lake, T_sfc, 0.0)
        LW_up_kin = ε_water * σ_SB * (T_sfc + 273.15)^4 / (ρ₀ * cₚ)
        Qᵀ_now = interp_linear(t_forcing, Q_precomp_kin, t_now) + LW_up_kin
        Qu_now = interp_linear(t_forcing, Qu_vals, t_now)
        Qv_now = interp_linear(t_forcing, Qv_vals, t_now)
        Qmag   = sqrt(Qu_now^2 + Qv_now^2)   # total kinematic stress magnitude (m²/s²)
        push!(t_series,  t_now)
        push!(QT_series, Qᵀ_now)
        push!(Qu_series, Qu_now)
        push!(Qv_series, Qv_now)
        abs(Qᵀ_now) < 1e-10 && return nothing   # avoid divide by zero during net heating
        L_MO = -Qmag^(3/2) / (κ_vonK * α_sfc * g_conv * Qᵀ_now)
        push!(L_MO_series, L_MO)
    end
    simulation.callbacks[:L_MO] = Callback(compute_L_MO, TimeInterval(1days))

    mkpath(OUTPUT_PATH)
    filename = joinpath(OUTPUT_PATH, "$(closure_name)$(SRC_TAG)$(UV_TAG)$(EOS_TAG)_winter$(winter_year).jld2")
    simulation.output_writers[:jld2] = JLD2OutputWriter(model,
        (; ubar, vbar, Tbar, Ribar, N²bar),
        filename           = filename,
        schedule           = AveragedTimeInterval(1hours),
        overwrite_existing = true)

    @info "═══ Running: $closure_name (UVstress) | winter $winter_year | t₀ = $t_iso ═══"
    run!(simulation)

    lmo_file = joinpath(OUTPUT_PATH, "$(closure_name)$(SRC_TAG)$(UV_TAG)$(EOS_TAG)_winter$(winter_year)_LMO.jld2")
    jldopen(lmo_file, "w") do f
        f["t"]    = t_series     # time of every callback fire (seconds since t_iso)
        f["QT"]   = QT_series    # kinematic heat flux (°C·m/s), positive upward
        f["Qu"]   = Qu_series    # kinematic x-momentum flux (m²/s²)
        f["Qv"]   = Qv_series    # kinematic y-momentum flux (m²/s²)
        f["L_MO"] = L_MO_series  # Monin-Obukhov length (m); shorter than t when QT ≈ 0
    end
end

# ── Closure registry ──────────────────────────────────────────────────────────
const closure_specs = [
    ("kepsilon",         make_kepsilon_default),
    ("kepsilon_Rist035", make_kepsilon_Rist035),
    ("CATKE",            make_CATKE),
    ("CATKE_highRi",     make_CATKE_highRi),
]

active_specs = if !isnothing(RUN_CLOSURE)
    specs = filter(((n, _),) -> n == RUN_CLOSURE, closure_specs)
    isempty(specs) &&
        error("Unknown closure '$RUN_CLOSURE'. Options: $(join(first.(closure_specs), ", "))")
    specs
else
    closure_specs
end

const OUTPUT_PATH = joinpath(@__DIR__, "..", "data", "TURB_outputs",
                             "eastern_mooring_GLEN_forced", "winter$(winter_year)")

for (closure_name, make_closure) in active_specs
    model = setup_model(make_closure())
    run_simulation(model, closure_name, OUTPUT_PATH)
end
