"""
Large Eddy Simulation (LES) of the Lake Superior Western Mooring Water Column — Single GPU

Freshwater setup matching the Lake Superior western mooring column-model experiment:
  - Location  : 47° 19.3' N  (f₀ ≈ 1.076 × 10⁻⁴ s⁻¹)
  - Depth     : 184 m, Δz = 2 m  →  Nz = 92  (fixed)
  - Salinity  : 0.05 g/kg (uniform — freshwater lake)
  - Temperature: 4 °C in-situ (uniform, converted to Conservative Temperature via TEOS-10)
  - Forcing   : constant 6 m/s wind + 200 W/m² mean cooling with ±50 W/m² diurnal cycle
  - Duration  : 30 days (configurable)

Grid: Nx × Ny adjusted to powers of 2 for efficient GPU FFTs (default 128 × 128).
       Lx, Ly set to maintain 2 m isotropic resolution (default 256 × 256 m).

Usage:
    julia --project=<project> run_LES_lake_superior_gpu.jl [--stop_time 30] [--Nx 128] [--Ny 128]
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyFormulations
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using GibbsSeaWater
using JLD2
using FileIO
using Printf
using Random
using Statistics
using ArgParse
using Glob

import Dates

#####
##### Command Line Argument Parsing
#####

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--U_wind"
            help = "Wind speed (m/s) for bulk wind stress calculation"
            arg_type = Float64
            default = 6.0
        "--Q_mean"
            help = "Mean surface heat flux (W/m²). Positive = cooling."
            arg_type = Float64
            default = 200.0
        "--Lz"
            help = "Domain depth (m). Western mooring: 184, Eastern mooring: 212."
            arg_type = Float64
            default = 184.0
        "--Nx"
            help = "Number of grid points in x-direction (should be power of 2)"
            arg_type = Int64
            default = 128
        "--Ny"
            help = "Number of grid points in y-direction (should be power of 2)"
            arg_type = Int64
            default = 128
        "--dt"
            help = "Initial timestep (seconds)"
            arg_type = Float64
            default = 0.1
        "--max_dt"
            help = "Maximum timestep (minutes)"
            arg_type = Float64
            default = 2.
        "--stop_time"
            help = "Stop time of simulation (days)"
            arg_type = Float64
            default = 30.
        "--time_interval"
            help = "Time interval for time series output (minutes)"
            arg_type = Float64
            default = 10.
        "--checkpoint_interval"
            help = "Time interval for checkpoint files (days)"
            arg_type = Float64
            default = 1.
        "--pickup"
            help = "Whether to pickup from latest checkpoint if available"
            arg_type = Bool
            default = true
        "--file_location"
            help = "Root directory for output files"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end

args = parse_commandline()

Random.seed!(123)

#####
##### Domain & Grid
#####

const Δ  = 2.0    # isotropic resolution (m)
const Lz = args["Lz"]  # domain depth (m)
const Nz = Int(Lz / Δ)   # = 92

const Nx = args["Nx"]
const Ny = args["Ny"]
const Lx = Nx * Δ  # horizontal extent follows from resolution
const Ly = Ny * Δ

const size_halo = 5

grid = RectilinearGrid(GPU(), Float64,
                       size     = (Nx, Ny, Nz),
                       halo     = (size_halo, size_halo, size_halo),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

#####
##### Physical Constants & Lake Superior Properties
#####

const eos = TEOS10EquationOfState(reference_density = 999.8)
const ρ₀  = eos.reference_density      # kg/m³ (≈ fresh water at 4 °C)
const g   = 9.80665                     # m/s²

# Salinity: 0.05 g/kg (Absolute Salinity, nearly zero)
const S_lake = 0.05  # g/kg

# In-situ temperature at 4 °C, uniform throughout the column.
# Convert to Conservative Temperature at each depth using TEOS-10.
const T_insitu = 4.0  # °C

function Θ_conservative(z::Float64)
    p_dbar = ρ₀ * g * abs(z) / 1e4  # sea pressure [dbar]
    return gsw_ct_from_t(S_lake, T_insitu, p_dbar)
end

# Coriolis: 47° 19.3' N
const lat = 47.0 + 19.3 / 60.0
const Ω   = 7.2921150e-5            # rad/s
const f₀  = 2 * Ω * sind(lat)      # s⁻¹ ≈ 1.076e-4

#####
##### Surface Forcing
#####

# Wind stress (kinematic) from bulk formula:
#   τ = ρ_air · C_D · U²  →  Qᵁ = -τ / ρ₀  (negative = downward momentum)
const ρ_air  = 1.225       # kg/m³
const C_D    = 1.2e-3      # neutral drag coefficient
const U_wind = args["U_wind"]  # m/s
const Qᵁ     = -ρ_air * C_D * U_wind^2 / ρ₀  # m²/s²

# Heat flux (positive = cooling, upward out of water in Oceananigans convention):
#   Q(t) = [Q_mean + Q_amp · cos(2π t / 1day)] / (ρ₀ · cₚ)
#   cos peaks at t=0 (midnight); trough at noon.
const cₚ     = 4182.0      # J/(kg·K), freshwater specific heat
const Q_mean = args["Q_mean"]  # W/m²
const Q_amp  = 50.0        # W/m²

@inline Qᵀ_diurnal(x, y, t) =
    (Q_mean + Q_amp * cos(2π * t / (1days))) / (ρ₀ * cₚ)  # °C m/s

# No salinity flux for a freshwater lake
const Qˢ = 0.0

#####
##### Initial Conditions (uniform, with tiny noise to seed turbulence)
#####

noise(x, y, z) = rand() * exp(z / 8)

T_initial(x, y, z) = Θ_conservative(z) + 1e-6 * noise(x, y, z)
S_initial(x, y, z) = S_lake            + 1e-8 * noise(x, y, z)

#####
##### Boundary Conditions
#####

# Temperature: diurnal flux at surface; no-flux at bottom (default)
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ_diurnal))

# Salinity: no flux (freshwater lake)
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qˢ))

# Wind stress in x-direction
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵁ))

#####
##### Model (no sponge layer — true no-flux bottom BC)
#####

model = NonhydrostaticModel(
    grid               = grid,
    closure            = nothing,          # pure LES (no subgrid model)
    coriolis           = FPlane(f = f₀),
    buoyancy           = SeawaterBuoyancy(equation_of_state = eos),
    tracers            = (:T, :S),
    timestepper        = :RungeKutta3,
    advection          = WENO(order = 9),
    boundary_conditions = (T = T_bcs, S = S_bcs, u = u_bcs),
)

set!(model, T = T_initial, S = S_initial)

T = model.tracers.T
S = model.tracers.S
u, v, w = model.velocities

#####
##### Output Directory
#####

FILE_NAME = "LES_lakesuperior_U$(U_wind)ms_QT$(round(Int,Q_mean))_Lxy$(round(Int,Lx))_Lz$(round(Int,Lz))_Nxy$(Nx)_Nz$(Nz)"
FILE_DIR  = joinpath(args["file_location"], FILE_NAME)
mkpath(FILE_DIR)

#####
##### Simulation
#####

simulation = Simulation(model,
                        Δt        = args["dt"]second,
                        stop_time = args["stop_time"]days)

wizard = TimeStepWizard(max_change = 1.05, max_Δt = args["max_dt"]minutes, cfl = 0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

#####
##### Progress Monitoring
#####

wall_clock_start    = time_ns()
wall_clock_interval = [time_ns()]

function print_progress(sim)
    now_ns         = time_ns()
    sim_time       = sim.model.clock.time
    wall_elapsed   = 1e-9 * (now_ns - wall_clock_start)
    wall_interval  = 1e-9 * (now_ns - wall_clock_interval[1])

    sim_rate  = sim_time / max(wall_elapsed, 1e-9)
    remaining = sim.stop_time - sim_time
    eta_s     = remaining / max(sim_rate, 1e-9)

    # simulation minutes per wall-clock hour
    rate_simmin_per_wallhr = sim_rate * 60.0

    @printf("%s [%05.2f%%] i: %d, t: %s, Δt: %s\n  wall elapsed: %s | interval: %s | rate: %.1f sim-min/wall-hr | ETA: %s\n  max|u|: %.3e  max|v|: %.3e  max|w|: %.3e m/s  max|T|: %.4f °C\n",
            Dates.now(),
            100 * (sim_time / sim.stop_time),
            sim.model.clock.iteration,
            prettytime(sim_time),
            prettytime(sim.Δt),
            prettytime(wall_elapsed),
            prettytime(wall_interval),
            rate_simmin_per_wallhr,
            prettytime(eta_s),
            maximum(sim.model.velocities.u),
            maximum(sim.model.velocities.v),
            maximum(sim.model.velocities.w),
            maximum(sim.model.tracers.T))

    wall_clock_interval[1] = now_ns
    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(100))

#####
##### Metadata Saving
#####

function init_save_some_metadata!(file, model)
    file["metadata/coriolis_parameter"]    = f₀
    file["metadata/momentum_flux"]         = Qᵁ
    file["metadata/heat_flux_mean_Wm2"]    = Q_mean
    file["metadata/heat_flux_amp_Wm2"]     = Q_amp
    file["metadata/salinity_flux"]         = Qˢ
    file["metadata/surface_salinity_gkg"]  = S_lake
    file["metadata/insitu_temperature_C"]  = T_insitu
    file["metadata/equation_of_state"]     = eos
    file["metadata/gravitational_accel"]   = g
    file["metadata/reference_density"]     = ρ₀
    file["metadata/Lx_m"]  = Lx
    file["metadata/Ly_m"]  = Ly
    file["metadata/Lz_m"]  = Lz
    file["metadata/Nx"]    = Nx
    file["metadata/Ny"]    = Ny
    file["metadata/Nz"]    = Nz
    return nothing
end

#####
##### Diagnostic Fields
#####

@inline function get_buoyancy(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyFormulations.get_temperature_and_salinity(b, C)
    @inbounds ρ = TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
    return -g * (ρ - ρ₀) / ρ₀
end

@inline function get_density(i, j, k, grid, b, C)
    T, S = Oceananigans.BuoyancyFormulations.get_temperature_and_salinity(b, C)
    @inbounds return TEOS10.ρ(T[i, j, k], S[i, j, k], 0, eos)
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid,
                                                        model.buoyancy, model.tracers)
b = Field(b_op)

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid,
                                                        model.buoyancy, model.tracers)
ρ = Field(ρ_op)

#####
##### Horizontally-Averaged Profiles (training data)
#####

ubar = Field(Average(u, dims = (1, 2)))
vbar = Field(Average(v, dims = (1, 2)))
Tbar = Field(Average(T, dims = (1, 2)))
Sbar = Field(Average(S, dims = (1, 2)))
bbar = Field(Average(b, dims = (1, 2)))
ρbar = Field(Average(ρ, dims = (1, 2)))

uw = Field(Average(w * u, dims = (1, 2)))
vw = Field(Average(w * v, dims = (1, 2)))
wb = Field(Average(w * b, dims = (1, 2)))
wT = Field(Average(w * T, dims = (1, 2)))
wS = Field(Average(w * S, dims = (1, 2)))
wρ = Field(Average(w * ρ, dims = (1, 2)))

timeseries_outputs = (; ubar, vbar, Tbar, Sbar, bbar, ρbar,
                        uw, vw, wT, wS, wb, wρ)

#####
##### Output Writers
#####

# 3D instantaneous fields (saved separately to manage file sizes)
field_schedule = TimeInterval(6hours)

simulation.output_writers[:u] = JLD2OutputWriter(model, (; u),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_u.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:v] = JLD2OutputWriter(model, (; v),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_v.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:w] = JLD2OutputWriter(model, (; w),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_w.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:T] = JLD2OutputWriter(model, (; T),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_T.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:S] = JLD2OutputWriter(model, (; S),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_S.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:b] = JLD2OutputWriter(model, (; b),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:ρ] = JLD2OutputWriter(model, (; ρ),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_rho.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

# Time series of horizontal averages and fluxes (key training data)
simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(args["time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

# Checkpoints for restart capability
simulation.output_writers[:checkpointer] = Checkpointer(model,
    schedule = TimeInterval(args["checkpoint_interval"]days),
    prefix   = "$(FILE_DIR)/model_checkpoint")

#####
##### Run Simulation
#####

pickup_path = nothing
if args["pickup"] && isdir(FILE_DIR)
    cp_files = filter(f -> occursin("model_checkpoint_iteration", f), readdir(FILE_DIR))
    if !isempty(cp_files)
        iters = parse.(Int, [fn[findfirst("iteration", fn)[end]+1:findfirst(".jld2", fn)[1]-1]
                             for fn in cp_files])
        iter_max = maximum(iters)
        pickup_path = "$(FILE_DIR)/model_checkpoint_iteration$(iter_max).jld2"
        @info "Picking up from checkpoint at iteration $(iter_max)"
    end
end

if isnothing(pickup_path)
    @info "Starting from initial conditions"
    run!(simulation)
else
    run!(simulation, pickup = pickup_path)
end

# Clean up checkpoint files after successful completion
cp_glob = glob("$(FILE_DIR)/model_checkpoint_iteration*.jld2")
if !isempty(cp_glob)
    @info "Removing checkpoint files..."
    rm.(cp_glob)
end

@info "Simulation completed successfully!"
