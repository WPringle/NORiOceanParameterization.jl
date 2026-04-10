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
using Printf
using SeawaterPolynomials
using SeawaterPolynomials: TEOS10
using GibbsSeaWater
using NORiOceanParameterization
using NORiOceanParameterization.Implementation

#####
##### Lake Superior Western Mooring — idealized column model
#####
# Location: 47° 19.3' N
# Depth:    H = 185 m
# Season:   cooling cycle with wind forcing
# Purpose:  compare k-ε, CATKE, and NORi NN closures under three
#           typical autumn wind speeds (4, 6, 8 m/s) and three
#           mean surface cooling rates (150, 200, 250 W/m²)
#####

model_architecture = CPU()

# ── Grid ────────────────────────────────────────────────────────────────────
const Nz = 23      # 184 / 8 = 23 exactly → dz = 8 m (NN training resolution)
const Lz = 184.0   # m (closest to 185 m divisible by 8)

grid = RectilinearGrid(model_architecture,
                       topology = (Flat, Flat, Bounded),
                       size = Nz,
                       halo = 3,
                       z = (-Lz, 0))

# ── Location & Coriolis ─────────────────────────────────────────────────────
# 47° 19.3' N  →  47 + 19.3/60 ≈ 47.3217°
const lat = 47.0 + 19.3 / 60.0          # decimal degrees
const Ω   = 7.2921150e-5                 # rad/s, Earth rotation rate
const f₀  = 2 * Ω * sind(lat)           # s⁻¹  ≈ 1.076e-4

# ── Lake Superior freshwater properties ─────────────────────────────────────
const S_lake = 0.05 / 1000               # kg/kg (Absolute Salinity, ~constant)

const T_insitu = 4.0                     # °C  (in-situ, uniform)
const ρ₀ = 999.8                         # kg/m³ (approx density of fresh 4°C water)
const g_conv = 9.80665                   # m/s²

# In-situ → Conservative Temperature conversion via the GibbsSeaWater toolbox.
#
# gsw_ct_from_t(SA, t, p):
#   SA  – Absolute Salinity [g/kg]
#   t   – in-situ temperature [°C]
#   p   – sea pressure [dbar]        (≈ ρ₀·g·|z| / 1e4)

function Θ_conservative(z::Float64)
    p_dbar = ρ₀ * g_conv * abs(z) / 1e4
    return gsw_ct_from_t(S_lake, T_insitu, p_dbar)
end

# ── Forcing parameters ───────────────────────────────────────────────────────
# Qᵀ > 0 for cooling in Oceananigans (upward heat flux out of the lake)
# Qᵁ < 0 for wind stress (downward momentum flux into the lake surface)
#
# Wind stress (kinematic) from bulk formula:
#   Qᵁ = -ρ_air · C_D · U² / ρ₀
# where C_D = 1.2×10⁻³ (neutral drag coefficient, standard for lakes)
#       ρ_air = 1.225 kg/m³

const ρ_air = 1.225     # kg/m³
const C_D   = 1.2e-3    # neutral drag coefficient

wind_stress(U_ms) = -ρ_air * C_D * U_ms^2 / ρ₀   # m²/s² (negative = into surface)

# Diurnal surface heat flux (positive = cooling)
# Amplitude is fixed at ±50 W/m² around each mean; e.g. for mean=200:
#   Q(t) = [200 + 50·cos(2π·t / 1day)] / (ρ₀ · cₚ)
#   cos peaks at t=0 (midnight) → 250 W/m²; trough at t=12h (noon) → 150 W/m²
const Q_amp   = 50.0                           # W/m² (half-range, same for all cases)
const cₚ      = 4182.0                         # J/(kg·K), freshwater specific heat
@inline Qᵀ_diurnal(t, p) = (p.Q_mean + p.Q_amp * cos(2π * t / (1days))) / (ρ₀ * cₚ)

# Three mean heat flux cases
const QT_mean_cases = [150.0, 200.0, 250.0]   # W/m²

# Three wind speed cases: typical autumn conditions on Lake Superior
const wind_cases = [
    ("U4ms",  4.0),    # moderate wind
    ("U6ms",  6.0),    # intermediate wind
    ("U8ms",  8.0),    # strong wind
]

# ── Closure definitions ──────────────────────────────────────────────────────

# k-ε with default parameters
function make_kepsilon_default()
    return TKEDissipationVerticalDiffusivity()
end

# k-ε with Cᵇϵ⁺ = Cᵇϵ⁻ = -0.26 → Riₛₜ ≈ 0.35
function make_kepsilon_Rist035()
    tke_diss_eqs = TKEDissipationEquations(Cᵇϵ⁺ = -0.26, Cᵇϵ⁻ = -0.26)
    return TKEDissipationVerticalDiffusivity(; tke_dissipation_equations = tke_diss_eqs)
end

# CATKE with default parameters (CRi⁰ = 0.25)
function make_CATKE()
    mixing_length = CATKEMixingLength()
    tke_eq = CATKEEquation()
    return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation = tke_eq)
end

# CATKE with Cˡᵒu = 0.52, Cˡᵒc = 0.64, Cʰⁱc = 0.115 → Riₛₜ ≈ 0.23, Γ⁰ ≈ 0.4, Γ∞ ≈ 0.2
function make_CATKE_highRi()
    mixing_length = CATKEMixingLength(Cˡᵒu = 0.52, Cˡᵒc = 0.64, Cʰⁱc = 0.115)
    tke_eq = CATKEEquation()
    return CATKEVerticalDiffusivity(; mixing_length, turbulent_kinetic_energy_equation = tke_eq)
end

# NORi NN closure with default Riᶜ (≈ 0.437, calibrated)
function make_NN()
    return NORiClosureWithNN(arch = model_architecture)
end

# NORi NN closure with CATKE-like Ri profile:
# constant mixing for 0 < Ri < 0.25, then linear decrease to background for 0.25 < Ri < 1.27
function make_NN_highRi()
    return NORiClosureWithNN(arch = model_architecture, Riˢʰ = 0.25, Riᶜ = 1.27)
end

# ── Richardson number diagnostic ─────────────────────────────────────────────
# Ri = N² / S²  at (Center, Center, Face), consistent with NORi base closure.
# Works for all turbulence closures.
@inline ϕ²_Ri(i, j, k, grid, ϕ, args...) = ϕ(i, j, k, grid, args...)^2

@inline function Ri_ccf(i, j, k, grid, u, v, buoyancy, tracers)
    ∂z_u² = ℑxᶜᵃᵃ(i, j, k, grid, ϕ²_Ri, ∂zᶠᶜᶠ, u)
    ∂z_v² = ℑyᵃᶜᵃ(i, j, k, grid, ϕ²_Ri, ∂zᶜᶠᶠ, v)
    S²     = ∂z_u² + ∂z_v²
    N²     = ∂z_b(i, j, k, grid, buoyancy, tracers)
    return ifelse(N² == 0, zero(grid), N² / (S² + 1e-11))
end

# ── Model setup ──────────────────────────────────────────────────────────────

function setup_model(closure, Qᵁ, Q_mean, Q_amp)
    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵁ))
    T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ_diurnal,
                parameters = (Q_mean = Q_mean, Q_amp = Q_amp)))
    # No salinity flux for freshwater lake
    S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0))

    coriolis = FPlane(f = f₀)

    if closure isa CATKEVerticalDiffusivity
        tracers = (:T, :S, :e)
    elseif closure isa TKEDissipationVerticalDiffusivity
        tracers = (:T, :S, :e, :ϵ)
    else
        tracers = (:T, :S)
    end

    model = HydrostaticFreeSurfaceModel(
        grid                 = grid,
        free_surface         = ImplicitFreeSurface(),
        momentum_advection   = WENO(grid = grid),
        tracer_advection     = WENO(grid = grid),
        buoyancy             = SeawaterBuoyancy(equation_of_state = TEOS10.TEOS10EquationOfState(reference_density = ρ₀)),
        coriolis             = coriolis,
        closure              = closure,
        tracers              = tracers,
        boundary_conditions  = (; T = T_bcs, S = S_bcs, u = u_bcs),
    )

    # Initial conditions: uniform T (converted to conservative), constant S
    # Small noise for numerical stability
    noise(z) = 1e-6 * rand() * exp(z / 20)
    T_ic(z) = Θ_conservative(z) + noise(z)
    S_ic(z) = S_lake + 1e-8 * noise(z)

    set!(model, T = T_ic, S = S_ic)
    update_state!(model)

    return model
end

# ── Simulation ───────────────────────────────────────────────────────────────

function run_simulation(model, Qᵁ, Q_mean, Q_amp, wind_label, closure_name, OUTPUT_PATH)
    QT_tag = "QTmean$(round(Int, Q_mean))Wm2"
    stop_time = 60days                   # two months
    Δt        = 5minutes

    simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

    # Progress reporting
    wall_clock = [time_ns()]

    function print_progress(sim)
        @printf("[%05.2f%%] i: %d, t: %s, wall time: %s, max|T|: %6.3f °C, next Δt: %s\n",
            100 * (sim.model.clock.time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim.model.clock.time),
            prettytime(1e-9 * (time_ns() - wall_clock[1])),
            maximum(abs, sim.model.tracers.T),
            prettytime(sim.Δt))
        wall_clock[1] = time_ns()
        return nothing
    end

    simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(500))

    # Approximate ice formation: clamp T ≥ 0°C (Conservative Temperature) each time step
    function clamp_freezing!(sim)
        clamp!(parent(sim.model.tracers.T), 0.0, Inf)
        return nothing
    end
    simulation.callbacks[:freeze] = Callback(clamp_freezing!, IterationInterval(1))

    # ── Outputs ──
    u, v, w = model.velocities
    T, S    = model.tracers.T, model.tracers.S

    # Column-averaged profiles (trivial for Flat,Flat topology)
    ubar = Field(Average(u, dims = (1, 2)))
    vbar = Field(Average(v, dims = (1, 2)))
    Tbar = Field(Average(T, dims = (1, 2)))
    Sbar = Field(Average(S, dims = (1, 2)))

    # Richardson number profile
    Ri_op  = KernelFunctionOperation{Center, Center, Face}(
                 Ri_ccf, grid, model.velocities.u, model.velocities.v,
                 model.buoyancy, model.tracers)
    Ribar  = Field(Average(Ri_op, dims = (1, 2)))

    # ── L_MO timeseries ──────────────────────────────────────────────────────
    # L_MO = -|Qᵁ|^(3/2) / (κ · α(T_sfc) · g · Qᵀ)
    # α is updated each output step from the surface conservative temperature
    # using gsw_alpha_wrt_t_exact(SA, T_sfc, 0).  κ = 0.41 (von Kármán).
    κ_vonK = 0.41
    g_grav  = 9.80665

    L_MO_series = zeros(0)
    t_series    = zeros(0)

    function compute_L_MO(sim)
        t     = sim.model.clock.time
        T_sfc = Array(interior(sim.model.tracers.T, 1, 1, Nz))[1]   # surface cell Θ (conservative T) [°C]
        # α w.r.t. conservative temperature at surface pressure (p = 0), consistent with TEOS10 EOS
        α_sfc = gsw_alpha(S_lake, T_sfc, 0.0)
        # Instantaneous kinematic heat flux at this time step
        Qᵀ_now = Qᵀ_diurnal(t, (; Q_mean, Q_amp))
        # L_MO = -|Qᵁ|^(3/2) / (κ · α · g · Qᵀ)
        # α > 0 (T > 4°C): L_MO < 0  — convective (destabilising cooling)
        # α < 0 (T < 4°C): L_MO > 0  — stable (cooling stabilises via inverse stratification)
        L_MO = -abs(Qᵁ)^(3/2) / (κ_vonK * α_sfc * g_grav * Qᵀ_now)
        push!(L_MO_series, L_MO)
        push!(t_series, t)
        return nothing
    end

    simulation.callbacks[:L_MO] = Callback(compute_L_MO, TimeInterval(1days))

    averaged_outputs = (; ubar, vbar, Tbar, Sbar, Ribar)

    mkpath(OUTPUT_PATH)

    filename = joinpath(OUTPUT_PATH,
        "$(closure_name)_$(wind_label)_$(QT_tag).jld2")

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, averaged_outputs,
        filename           = filename,
        schedule           = AveragedTimeInterval(1days),   # daily mean profiles
        overwrite_existing = true)

    @info "Running $closure_name | wind = $wind_label | Qᵀ = diurnal (150–250 W/m²) | Qᵁ = $(round(Qᵁ, sigdigits=4))"

    run!(simulation)

    # Save L_MO timeseries alongside the main output
    lmo_filename = joinpath(OUTPUT_PATH,
        "$(closure_name)_$(wind_label)_$(QT_tag)_LMO.jld2")
    jldopen(lmo_filename, "w") do f
        f["t"]   = t_series
        f["L_MO"] = L_MO_series
    end

    return nothing
end

# ── Experiment loop ───────────────────────────────────────────────────────────

OUTPUT_PATH = joinpath(@__DIR__, "..", "data", "lake_superior_western_mooring")

# Closure name → factory function mapping
closure_specs = [
    ("kepsilon",         make_kepsilon_default),
    ("kepsilon_Rist035", make_kepsilon_Rist035),
    ("CATKE",            make_CATKE),
    ("CATKE_highRi",     make_CATKE_highRi),
    ("NN",               make_NN),
    ("NN_highRi",        make_NN_highRi),
]

# Optional command-line arguments to run a single combination:
#   julia script.jl <wind_label> <closure_name> [QT_mean]
#   e.g. julia script.jl U4ms kepsilon 200
# With no arguments all 27 combinations run sequentially.
if length(ARGS) >= 2
    run_wind    = ARGS[1]
    run_closure = ARGS[2]
    active_wind_cases    = filter(((l, _),) -> l == run_wind,    wind_cases)
    active_closure_specs = filter(((n, _),) -> n == run_closure, closure_specs)
    isempty(active_wind_cases)    && error("Unknown wind label '$run_wind'. Options: $(first.(wind_cases))")
    isempty(active_closure_specs) && error("Unknown closure '$run_closure'. Options: $(first.(closure_specs))")
    if length(ARGS) == 3
        active_QT_means = [parse(Float64, ARGS[3])]
        active_QT_means[1] ∈ QT_mean_cases || error("Unknown QT_mean '$(ARGS[3])'. Options: $QT_mean_cases")
    else
        active_QT_means = QT_mean_cases
    end
elseif isempty(ARGS)
    active_wind_cases    = wind_cases
    active_closure_specs = closure_specs
    active_QT_means      = QT_mean_cases
else
    error("Usage: julia script.jl [wind_label closure_name [QT_mean]]\n" *
          "  wind labels : $(first.(wind_cases))\n" *
          "  closures    : $(first.(closure_specs))\n" *
          "  QT_means    : $QT_mean_cases")
end

for Q_mean in active_QT_means
    for (wind_label, U_ms) in active_wind_cases
        Qᵁ = wind_stress(U_ms)

        @info "═══════════════════════════════════════════════"
        @info "Wind case: $wind_label  (U = $(U_ms) m/s) | Qᵀ mean = $(round(Int, Q_mean)) W/m²"
        @info "  Qᵁ = $(round(Qᵁ, sigdigits=4)) m²/s²"

        for (closure_name, make_closure) in active_closure_specs
            closure = make_closure()
            model   = setup_model(closure, Qᵁ, Q_mean, Q_amp)
            run_simulation(model, Qᵁ, Q_mean, Q_amp, wind_label, closure_name, OUTPUT_PATH)
        end
    end
end
