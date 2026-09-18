#####
##### Lake Superior Eastern Mooring — column model forced by Stannard Rock GLEN observations
#####
# Replaces the idealised constant wind/heat-flux forcing in the sister script with
# half-hourly eddy-covariance + radiation measurements from the GLEN Stannard Rock buoy.
#
# Net upward heat flux (Oceananigans convention, positive = lake cooling):
#   Q_net = SHF + LHF - SW↓ - LW↓       [W/m²]
# SHF and LHF are measured as surface_upward (positive upward).
# SW↓ and LW↓ are downwelling only (positive into lake surface), so they subtract.
#
# Wind speed is measured 39 m above the lake surface (station at 222 m a.s.l.,
# Lake Superior at ~183 m a.s.l.).  Corrected to the 10-m reference height using a
# neutral log-wind profile before applying the bulk drag formula.
#
# Usage:
#   julia lake_superior_eastern_mooring_GLEN_forced.jl <winter_year> [closure_name]
#   e.g.  julia lake_superior_eastern_mooring_GLEN_forced.jl 2009
#         julia lake_superior_eastern_mooring_GLEN_forced.jl 2009 CATKE
#
# Inputs  : figure_data/lake_superior_eastern_mooring/
#               lake_superior_eastern_mooring_winter_start_dates.csv
#           /lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/
#               US_StannardRockSuperior_processed_halfhourly_qc.nc
# Outputs : data/TURB_outputs/eastern_mooring_GLEN_forced/winter<YEAR>/
#               <closure>_winter<YEAR>.jld2        (daily-averaged profiles)
#               <closure>_winter<YEAR>_LMO.jld2   (Monin-Obukhov length timeseries)
#               forcing_winter<YEAR>.jld2          (GLEN forcing arrays used)
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
using SeawaterPolynomials: TEOS10
using GibbsSeaWater
using NORiOceanParameterization
using NORiOceanParameterization.Implementation

# ── Command-line arguments ────────────────────────────────────────────────────
length(ARGS) < 1 &&
    error("Usage: julia $(basename(@__FILE__)) <winter_year> [closure_name]\n" *
          "  winter_year : e.g. 2009, 2010, 2011, 2014\n" *
          "  closure_name: kepsilon | kepsilon_Rist035 | CATKE | CATKE_highRi  (default: all)")

const winter_year = parse(Int, ARGS[1])

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
const ρ_air   = 1.225      # kg/m³
const cpa_air = 1005.0     # J/(kg·K) — specific heat of dry air
const g_conv   = 9.80665   # m/s²
const S_lake   = 0.05      # g/kg (Absolute Salinity, essentially constant)
const T_insitu = 4.0       # °C  (initial in-situ temperature)
const T_ref    = 273.15    # K   — approximate fall air temperature (used for L_MO computation)

# Radiation constants for net SW and LW
const albedo_sw = 0.08     # broadband shortwave albedo of fresh water surface
const ε_water   = 0.98     # longwave emissivity of water
const σ_SB      = 5.67e-8  # W/(m²·K⁴) — Stefan-Boltzmann constant

# Wind anemometer height above the lake surface.
# Station sits 222 m a.s.l.; Lake Superior surface ≈ 183 m a.s.l. → 39 m above lake.
const z_anem  = 39.0      # m

# Charnock constant for roughness length: z₀ = α_c · u*² / g
const α_charnock = 0.0101

# ── Load GLEN half-hourly data ────────────────────────────────────────────────
const GLEN_FILE = "/lcrc/project/HSOFS_Ensemble/COMPASS_GLM/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc.nc"

glen_times_all, lhf_all, shf_all, sw_all, lw_all, wspd_all =
    NCDataset(GLEN_FILE) do ds
        times = DateTime.(ds["time"][:])
        function load_var(name)
            v = ds[name][:]
            [ismissing(x) ? missing : Float64(x) for x in v]
        end
        times,
        load_var("latent_heat_flux"),
        load_var("sensible_heat_flux"),
        load_var("downwelling_shortwave_flux"),
        load_var("downwelling_longwave_flux"),
        load_var("wind_speed")
    end
@info "GLEN dataset: $(glen_times_all[1]) → $(glen_times_all[end]) ($(length(glen_times_all)) half-hourly records)"

# ── Slice to simulation window: [t_iso, t_iso + 60 days] ─────────────────────
const sim_days = 60
t_end = t_iso + Day(sim_days)
idx   = findall(t -> t_iso <= t <= t_end, glen_times_all)
isempty(idx) && error("No GLEN data in window $t_iso → $t_end")
@info "GLEN window: $(glen_times_all[idx[1]]) → $(glen_times_all[idx[end]]) ($(length(idx)) records)"

glen_t  = glen_times_all[idx]
lhf_raw = lhf_all[idx]
shf_raw = shf_all[idx]
sw_raw  = sw_all[idx]
lw_raw  = lw_all[idx]
ws_raw  = wspd_all[idx]

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

lhf_f = fill_missing_linear(lhf_raw)
shf_f = fill_missing_linear(shf_raw)
sw_f  = fill_missing_linear(sw_raw)
lw_f  = fill_missing_linear(lw_raw)
ws_f  = fill_missing_linear(ws_raw)

# ── Stability function ψₘ(ζ), ζ = z/L_MO ────────────────────────────────────
# Businger-Dyer-Paulson (unstable) and Webb (stable) forms:
#   ζ ≤ 0:        S1 = 2·ln((1+x)/2) + ln((1+x²)/2) - 2·arctan(x) + π/2,  x = (1-a1·ζ)^(1/4)
#   0 < ζ < 1:    S1 = -a2·ζ
#   ζ ≥ 1:        S1 = -a2*(1 + ln(ζ))  #(very stable, avoid overcorrection)
const a1_unstable = 16.0   # Laird et al. (2002) #Paulson 1970
const a2_stable   =  5.2   # Laird et al. (2002) #Dyer 1974 / Webb 1970

function psi_m(zeta)
    if zeta <= 0.0
        x = (1.0 - a1_unstable * zeta)^0.25
        return 2.0*log((1.0 + x)/2.0) + log((1.0 + x^2)/2.0) - 2.0*atan(x) + π/2.0
    elseif zeta < 1.0
        return -a2_stable * zeta
    else
        return -a2_stable * (1.0 + log(zeta))
    end
end

# ── Iterative friction velocity (Charnock + MOST) ─────────────────────────────
# z₀ = α_c · u*² / g  (Charnock)
# u* = U_meas · κ / (ln(z/z₀) - S1(z/L_MO))
# L_MO = -u*³ · T_ref · ρ_air · cpa_air / (κ · g · SHF)   [from sensible heat flux]
#
# Initialise with u* from a neutral guess, then iterate.
const κ_bulk = 0.41
const z0_min = 1.0e-5    # floor for z₀ (smooth limit)
const z0_max = 1.0e-2    # cap for z₀ (avoid blow-up at very low winds)

function compute_ustar(U_meas, SHF_val; n_iter = 15)
    U_meas <= 0.0 && return 0.0
    # Initial guess: neutral, Charnock z₀ from a rough CD
    z_0    = max(α_charnock * (κ_bulk * U_meas / log(z_anem / 1.5e-4))^2 / g_conv, z0_min)
    u_star = κ_bulk * U_meas / log(z_anem / z_0)

    for _ in 1:n_iter
        z_0 = clamp(α_charnock * u_star^2 / g_conv, z0_min, z0_max)
        # Monin-Obukhov length from measured sensible heat flux (positive upward)
        # B = (g/T_ref) · SHF / (ρ_air · cpa_air)  [buoyancy flux, m²/s³]
        # L = -u*³ / (κ · B)
        B = (g_conv / T_ref) * SHF_val / (ρ_air * cpa_air)
        L_MO = if abs(B) > 1.0e-8
            -u_star^3 / (κ_bulk * B)
        else
            1.0e6   # effectively neutral
        end
        S1     = psi_m(z_anem / L_MO)
        u_star = max(κ_bulk * U_meas / (log(z_anem / z_0) - S1), 1.0e-4)
    end
    return u_star
end

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

# Kinematic momentum flux (m²/s²), negative = into lake surface
# u* from iterative Charnock + MOST;  τ = -ρ_air · u*² / ρ₀
const ustar_vals = [compute_ustar(ws_f[k], shf_f[k]) for k in eachindex(ws_f)]
const τ_kin_vals = @. -ρ_air * ustar_vals^2 / ρ₀

@info "Forcing summary (60-day window):"
@info "  Q_precomp (W/m²): mean=$(round(mean(Q_precomp_Wm2), digits=1))" *
      "  min=$(round(minimum(Q_precomp_Wm2), digits=1))  max=$(round(maximum(Q_precomp_Wm2), digits=1))"
@info "  LW_up at 4 °C (W/m²): $(round(ε_water * σ_SB * (4.0 + 273.15)^4, digits=1))  (added online from T_sfc)"
@info "  u*     (m/s)  : mean=$(round(mean(ustar_vals), digits=4))  max=$(round(maximum(ustar_vals), digits=4))"
@info "  U_meas (m/s)  : mean=$(round(mean(ws_f), digits=2))  max=$(round(maximum(ws_f), digits=2))"

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

# Momentum flux BC — continuous form; no state dependence.
@inline Qᵁ_obs(t, p) = interp_linear(p.t_forcing, p.τ_kin_vals, t)

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
                          parameters = (; t_forcing, τ_kin_vals)))
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0))

    tracers = if closure isa CATKEVerticalDiffusivity
        (:T, :S, :e)
    elseif closure isa TKEDissipationVerticalDiffusivity
        (:T, :S, :e, :ϵ)
    else
        (:T, :S)
    end

    model = HydrostaticFreeSurfaceModel(
        grid                = grid,
        free_surface        = ImplicitFreeSurface(),
        momentum_advection  = WENO(grid = grid),
        tracer_advection    = WENO(grid = grid),
        buoyancy            = SeawaterBuoyancy(
                                  equation_of_state = TEOS10.TEOS10EquationOfState(
                                      reference_density = ρ₀)),
        coriolis            = FPlane(f = f₀),
        closure             = closure,
        tracers             = tracers,
        boundary_conditions = (; T = T_bcs, S = S_bcs, u = u_bcs),
    )

    noise(z) = 1e-6 * rand() * exp(z / 8)
    set!(model, T = z -> Θ_conservative(z) + noise(z),
                S = z -> S_lake + 1e-8 * noise(z))
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
    T, S    = model.tracers.T, model.tracers.S

    ubar  = Field(Average(u, dims = (1, 2)))
    vbar  = Field(Average(v, dims = (1, 2)))
    Tbar  = Field(Average(T, dims = (1, 2)))
    Sbar  = Field(Average(S, dims = (1, 2)))
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
    QU_series  = Float64[]

    function compute_L_MO(sim)
        t_now  = sim.model.clock.time
        T_sfc  = Array(interior(sim.model.tracers.T, 1, 1, Nz))[1]
        α_sfc  = gsw_alpha(S_lake, T_sfc, 0.0)
        LW_up_kin = ε_water * σ_SB * (T_sfc + 273.15)^4 / (ρ₀ * cₚ)
        Qᵀ_now = interp_linear(t_forcing, Q_precomp_kin, t_now) + LW_up_kin
        Qᵁ_now = interp_linear(t_forcing, τ_kin_vals, t_now)
        push!(t_series,  t_now)
        push!(QT_series, Qᵀ_now)
        push!(QU_series, Qᵁ_now)
        abs(Qᵀ_now) < 1e-10 && return nothing   # avoid divide by zero during net heating
        L_MO = -abs(Qᵁ_now)^(3/2) / (κ_vonK * α_sfc * g_conv * Qᵀ_now)
        push!(L_MO_series, L_MO)
    end
    simulation.callbacks[:L_MO] = Callback(compute_L_MO, TimeInterval(1days))

    mkpath(OUTPUT_PATH)
    filename = joinpath(OUTPUT_PATH, "$(closure_name)_winter$(winter_year).jld2")
    simulation.output_writers[:jld2] = JLD2OutputWriter(model,
        (; ubar, vbar, Tbar, Sbar, Ribar, N²bar),
        filename           = filename,
        schedule           = AveragedTimeInterval(1hours),
        overwrite_existing = true)

    @info "═══ Running: $closure_name | winter $winter_year | t₀ = $t_iso ═══"
    run!(simulation)

    lmo_file = joinpath(OUTPUT_PATH, "$(closure_name)_winter$(winter_year)_LMO.jld2")
    jldopen(lmo_file, "w") do f
        f["t"]    = t_series     # time of every callback fire (seconds since t_iso)
        f["QT"]   = QT_series    # kinematic heat flux (°C·m/s), positive upward
        f["QU"]   = QU_series    # kinematic momentum flux (m²/s²), negative into lake
        f["L_MO"] = L_MO_series  # Monin-Obukhov length (m); shorter than t when QT ≈ 0
    end
end

# ── Save GLEN forcing used in this run ────────────────────────────────────────
function save_forcing(OUTPUT_PATH)
    mkpath(OUTPUT_PATH)
    forcing_file = joinpath(OUTPUT_PATH, "forcing_winter$(winter_year).jld2")
    jldopen(forcing_file, "w") do f
        f["t_iso"]           = string(t_iso)
        f["t_forcing"]       = t_forcing        # seconds since t_iso
        f["Q_precomp_Wm2"]   = Q_precomp_Wm2   # W/m²  SHF+LHF-SW_net-LW↓  (LW_up added online)
        f["Q_precomp_kin"]   = Q_precomp_kin    # °C·m/s  (pre-computed kinematic heat flux)
        f["SW_net_vals"]     = SW_net_vals      # W/m²  (1-albedo)·SW↓
        f["ustar_vals"]      = ustar_vals       # m/s  (friction velocity from Charnock+MOST)
        f["tau_kin_vals"]    = τ_kin_vals       # m²/s² (kinematic momentum flux, negative = into lake)
        f["lhf"]             = lhf_f            # W/m²
        f["shf"]             = shf_f            # W/m²
        f["sw_down"]         = sw_f             # W/m²  (raw downwelling SW)
        f["lw_down"]         = lw_f             # W/m²  (downwelling LW)
        f["wspd_meas"]       = ws_f             # m/s  (at z_anem = 39 m)
        f["z_anem_m"]        = z_anem
        f["albedo_sw"]       = albedo_sw
        f["eps_water"]       = ε_water
        f["alpha_charnock"]  = α_charnock
        f["a1_unstable"]     = a1_unstable
        f["a2_stable"]       = a2_stable
    end
    @info "Saved forcing → $forcing_file"
end

# ── Closure registry ──────────────────────────────────────────────────────────
const closure_specs = [
    ("kepsilon",         make_kepsilon_default),
    ("kepsilon_Rist035", make_kepsilon_Rist035),
    ("CATKE",            make_CATKE),
    ("CATKE_highRi",     make_CATKE_highRi),
]

active_specs = if length(ARGS) >= 2
    run_closure = ARGS[2]
    specs = filter(((n, _),) -> n == run_closure, closure_specs)
    isempty(specs) &&
        error("Unknown closure '$run_closure'. Options: $(join(first.(closure_specs), ", "))")
    specs
else
    closure_specs
end

const OUTPUT_PATH = joinpath(@__DIR__, "..", "data", "TURB_outputs",
                             "eastern_mooring_GLEN_forced", "winter$(winter_year)")

save_forcing(OUTPUT_PATH)

for (closure_name, make_closure) in active_specs
    model = setup_model(make_closure())
    run_simulation(model, closure_name, OUTPUT_PATH)
end
