"""
Large Eddy Simulation (LES) of the Lake Superior Western Mooring Water Column

Freshwater setup matching the Lake Superior western mooring column-model experiment:
  - Location : 47° 19.3' N  (f₀ ≈ 1.076 × 10⁻⁴ s⁻¹)
  - Depth     : 184 m (uniform), domain 184 × 184 × 184 m, Δx=Δy=Δz=2 m isotropic
  - Salinity  : 0.05 g/kg (uniform, no gradient — freshwater lake)
  - Temperature: 4 °C in-situ (uniform, converted to Conservative Temperature)
  - Forcing   : constant 6 m/s wind + 200 W/m² mean cooling with ±50 W/m² diurnal cycle
  - Duration  : 30 days

Parallelism: CPU() with MPI Distributed (x × y partition).
Run with:
    mpiexec -n <Rx*Ry> julia --project=. --threads=<Nthreads> experiments/run_LES_lake_superior.jl
"""

using Oceananigans
using Oceananigans.Units
using Oceananigans.DistributedComputations
using Oceananigans.BuoyancyFormulations
using Oceananigans.AbstractOperations: KernelFunctionOperation
using MPI
MPI.Init()
using JLD2
using FileIO
using Printf
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using GibbsSeaWater
using Random
using Glob

import Dates

#####
##### Distributed architecture
#####

# Hybrid MPI+threads: Rx=2, Ry=2 → 4 MPI ranks (1 per node), 128 threads each.
# 4 nodes × 128 cores = 512 cores total.
# NonhydrostaticModel's FFT pressure solver requires Rz=1 (z must be global).
# FFT solver also requires Ry | Nz: 92 / 2 = 46 ✓.
# 4 = 2×2; Nx=Ny=92 → 92/2=46 cells per subdomain (both ≥ halo=5 ✓).
# Subdomains: 46 × 46 × 92 per rank → 194768 cells/rank.
const Rx = 2
const Ry = 2

arch = Distributed(CPU(), partition = Partition(x = Rx, y = Ry))

# Seed RNG differently per rank for independent noise
Random.seed!(123 + arch.local_rank)

#####
##### Domain & grid
#####

# Fully isotropic domain: 184 × 184 × 184 m at 2 m resolution → 92 points per direction.
# 184 / 2 = 92 exactly. Ry=2 divides Nz=92 (92/2=46 ✓).
# Rx=2 divides Nx=92 (92/2=46 ✓). All subdomain widths ≥ halo=5 ✓.
const Lx = 184.0   # m
const Ly = 184.0   # m
const Lz = 184.0   # m
const Nx = 92
const Ny = 92
const Nz = 92

const size_halo = 5

grid = RectilinearGrid(arch, Float64,
                       size     = (Nx, Ny, Nz),
                       halo     = (size_halo, size_halo, size_halo),
                       x        = (0, Lx),
                       y        = (0, Ly),
                       z        = (-Lz, 0),
                       topology = (Periodic, Periodic, Bounded))

#####
##### Physical constants & Lake Superior properties
#####

const eos  = TEOS10EquationOfState(reference_density = 999.8)
const ρ₀   = eos.reference_density      # kg/m³ (≈ fresh water at 4 °C)
const g    = 9.80665                     # m/s²

# Salinity: 0.05 g/L = 0.05 g/kg (Absolute Salinity, nearly zero)
const S_lake = 0.05 / 1.0               # g/kg

# In-situ temperature at 4 °C, uniform throughout the column.
# Convert to Conservative Temperature at each depth using TEOS-10.
const T_insitu = 4.0                    # °C

function Θ_conservative(z::Float64)
    p_dbar = ρ₀ * g * abs(z) / 1e4     # sea pressure [dbar]
    return gsw_ct_from_t(S_lake, T_insitu, p_dbar)
end

# Coriolis: 47° 19.3' N
const lat = 47.0 + 19.3 / 60.0
const Ω   = 7.2921150e-5                # rad/s
const f₀  = 2 * Ω * sind(lat)          # s⁻¹ ≈ 1.076e-4

#####
##### Surface forcing
#####

# Wind stress (kinematic) from bulk formula:
#   τ = ρ_air · C_D · U²  →  Qᵁ = -τ / ρ₀  (negative = downward momentum)
const ρ_air = 1.225                     # kg/m³
const C_D   = 1.2e-3                    # neutral drag coefficient
const U_wind = 6.0                      # m/s  (median case)
const Qᵁ    = -ρ_air * C_D * U_wind^2 / ρ₀   # m²/s²

# Heat flux (positive = cooling, i.e. upward flux out of water in Oceananigans convention):
#   Q(t) = [Q_mean + Q_amp · cos(2π t / 1day)] / (ρ₀ · cₚ)
#   cos peaks at t=0 (midnight, 250 W/m²); trough at noon (150 W/m²).
const cₚ      = 4182.0                  # J/(kg·K), freshwater specific heat
const Q_mean  = 200.0                   # W/m²
const Q_amp   = 50.0                    # W/m²

@inline Qᵀ_diurnal(x, y, t) =
    (Q_mean + Q_amp * cos(2π * t / (1days))) / (ρ₀ * cₚ)  # °C m/s

# No salinity flux for a freshwater lake
const Qˢ = 0.0

#####
##### Initial conditions  (uniform, with tiny noise to seed turbulence)
#####

noise(x, y, z) = rand() * exp(z / 8)

T_initial(x, y, z) = Θ_conservative(z) + 1e-6 * noise(x, y, z)
S_initial(x, y, z) = S_lake            + 1e-8 * noise(x, y, z)

#####
##### Boundary conditions
#####

# Temperature: diurnal flux at surface; no-flux at bottom (default)
T_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵀ_diurnal))

# Salinity: no flux at top and bottom (freshwater lake)
S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qˢ))

# Wind stress in x-direction; no-slip bottom handled by default (no-flux on u)
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
##### Output directory (only rank 0 creates it)
#####

FILE_NAME = "LES_lakesuperior_U6ms_QT200_Lxy$(round(Int,Lx))_Lz$(round(Int,Lz))_Nxy$(Nx)_Nz$(Nz)"
FILE_DIR  = joinpath(@__DIR__, "..", "data", "LES", FILE_NAME)

arch.local_rank == 0 && mkpath(FILE_DIR)
MPI.Barrier(MPI.COMM_WORLD)

#####
##### Simulation
#####

simulation = Simulation(model,
                        Δt        = 0.1second,
                        stop_time = 30days)

wizard = TimeStepWizard(max_change = 1.05, max_Δt = 2minutes, cfl = 0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

#####
##### Progress
#####

wall_clock_start    = time_ns()   # fixed at simulation start
wall_clock_interval = [time_ns()] # resets each callback for interval rate

function print_progress(sim)
    sim.model.grid.architecture.local_rank == 0 || return nothing

    now_ns         = time_ns()
    sim_time       = sim.model.clock.time          # seconds of simulation time
    wall_elapsed   = 1e-9 * (now_ns - wall_clock_start)      # total wall seconds
    wall_interval  = 1e-9 * (now_ns - wall_clock_interval[1]) # wall seconds this interval

    # Cumulative rate (sim-seconds per wall-second), used for ETA
    sim_rate  = sim_time / max(wall_elapsed, 1e-9)   # sim-s / wall-s
    remaining = sim.stop_time - sim_time              # sim-s left
    eta_s     = remaining / max(sim_rate, 1e-9)       # wall-s to completion

    # Convert rate to convenient units: simulation minutes per wall-clock hour
    # sim_rate [sim-s/wall-s] × (1 sim-min/60 sim-s) × (3600 wall-s/wall-hr) = sim_rate × 60
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

# Fire every 30 simulation minutes so progress is meaningful regardless of timestep size.
# Early on (small Δt) many iterations pass silently; once Δt ramps to ~2 min you get
# ~15 iterations between prints — fine-grained enough to track without flooding logs.
#simulation.callbacks[:print_progress] = Callback(print_progress, TimeInterval(30minutes))

#####
##### Metadata
#####

function init_save_some_metadata!(file, model)
    model.grid.architecture.local_rank == 0 || return nothing
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
    file["metadata/Lx_m"]  = Lx   # 184 m
    file["metadata/Ly_m"]  = Ly   # 184 m
    file["metadata/Lz_m"]  = Lz   # 184 m
    file["metadata/Nx"]    = Nx   # 92
    file["metadata/Ny"]    = Ny   # 92
    file["metadata/Nz"]    = Nz   # 92
    return nothing
end

#####
##### Diagnostic fields (buoyancy, density)
#####

@inline function get_buoyancy(i, j, k, grid, b, C)
    T_loc, S_loc = Oceananigans.BuoyancyFormulations.get_temperature_and_salinity(b, C)
    @inbounds ρ  = TEOS10.ρ(T_loc[i, j, k], S_loc[i, j, k], 0, eos)
    return -g * (ρ - ρ₀) / ρ₀
end

@inline function get_density(i, j, k, grid, b, C)
    T_loc, S_loc = Oceananigans.BuoyancyFormulations.get_temperature_and_salinity(b, C)
    @inbounds return TEOS10.ρ(T_loc[i, j, k], S_loc[i, j, k], 0, eos)
end

b_op = KernelFunctionOperation{Center, Center, Center}(get_buoyancy, model.grid,
                                                        model.buoyancy, model.tracers)
b = Field(b_op)

ρ_op = KernelFunctionOperation{Center, Center, Center}(get_density, model.grid,
                                                        model.buoyancy, model.tracers)
ρ = Field(ρ_op)

#####
##### Horizontally-averaged profiles (training data)
#####

ubar = Field(Average(u, dims = (1, 2)))
vbar = Field(Average(v, dims = (1, 2)))
Tbar = Field(Average(T, dims = (1, 2)))
Sbar = Field(Average(S, dims = (1, 2)))
bbar = Field(Average(b, dims = (1, 2)))
ρbar = Field(Average(ρ, dims = (1, 2)))

uw   = Field(Average(w * u, dims = (1, 2)))
vw   = Field(Average(w * v, dims = (1, 2)))
wb   = Field(Average(w * b, dims = (1, 2)))
wT   = Field(Average(w * T, dims = (1, 2)))
wS   = Field(Average(w * S, dims = (1, 2)))
wρ   = Field(Average(w * ρ, dims = (1, 2)))

avg_fields = (; ubar, vbar, Tbar, Sbar, bbar, ρbar, uw, vw, wT, wS, wb, wρ)

#####
##### Output writers
#####

# 1-D horizontally-averaged time series written manually from rank 0 only.
# JLD2OutputWriter is NOT used here because it opens the same file from all ranks
# simultaneously, causing race conditions and Bus errors on GPFS (MmapIO issue).
# Instead, all ranks compute the averages collectively, then rank 0 appends to the
# file using IOStream mode (no mmap), which is GPFS-safe.
const TIMESERIES_FILE = "$(FILE_DIR)/instantaneous_timeseries.jld2"

# Initialise the file with metadata on rank 0 only.
if arch.local_rank == 0
    jldopen(TIMESERIES_FILE, "w", iotype = JLD2.IOStream) do f
        init_save_some_metadata!(f, model)
    end
end
MPI.Barrier(MPI.COMM_WORLD)

function save_timeseries!(sim)
    # Compute all averages (collective across ranks)
    for fld in avg_fields
        compute!(fld)
    end

    # Only rank 0 writes
    sim.model.grid.architecture.local_rank == 0 || return nothing

    t  = sim.model.clock.time
    it = sim.model.clock.iteration
    jldopen(TIMESERIES_FILE, "a+", iotype = JLD2.IOStream) do f
        f["timeseries/t/$it"]    = t
        f["timeseries/ubar/$it"] = Array(interior(ubar, 1, 1, :))
        f["timeseries/vbar/$it"] = Array(interior(vbar, 1, 1, :))
        f["timeseries/Tbar/$it"] = Array(interior(Tbar, 1, 1, :))
        f["timeseries/Sbar/$it"] = Array(interior(Sbar, 1, 1, :))
        f["timeseries/bbar/$it"] = Array(interior(bbar, 1, 1, :))
        f["timeseries/ρbar/$it"] = Array(interior(ρbar, 1, 1, :))
        f["timeseries/uw/$it"]   = Array(interior(uw,   1, 1, :))
        f["timeseries/vw/$it"]   = Array(interior(vw,   1, 1, :))
        f["timeseries/wT/$it"]   = Array(interior(wT,   1, 1, :))
        f["timeseries/wS/$it"]   = Array(interior(wS,   1, 1, :))
        f["timeseries/wb/$it"]   = Array(interior(wb,   1, 1, :))
        f["timeseries/wρ/$it"]   = Array(interior(wρ,   1, 1, :))
    end
    return nothing
end

simulation.callbacks[:timeseries] = Callback(save_timeseries!, TimeInterval(10minutes))

# Checkpoints every day (rank-specific prefix avoids multi-rank file conflicts)
simulation.output_writers[:checkpointer] = Checkpointer(model,
    schedule = TimeInterval(1days),
    prefix   = "$(FILE_DIR)/model_checkpoint_rank$(arch.local_rank)")

#####
##### Run
#####

# Pickup from latest checkpoint if one exists
pickup_path = nothing
if isdir(FILE_DIR)
    cp_files = filter(f -> occursin("model_checkpoint_iteration", f), readdir(FILE_DIR))
    if !isempty(cp_files)
        iters = parse.(Int, [f[findfirst("iteration", f)[end]+1:findfirst(".jld2", f)[1]-1]
                             for f in cp_files])
        iter_max = maximum(iters)
        pickup_path = "$(FILE_DIR)/model_checkpoint_iteration$(iter_max).jld2"
        arch.local_rank == 0 &&
            @info "Picking up from checkpoint iteration $(iter_max)"
    end
end

if isnothing(pickup_path)
    arch.local_rank == 0 && @info "Starting from initial conditions"
    run!(simulation)
else
    run!(simulation, pickup = pickup_path)
end

# Clean up checkpoint files on rank 0
if arch.local_rank == 0
    cp_glob = glob("$(FILE_DIR)/model_checkpoint_iteration*.jld2")
    isempty(cp_glob) || (rm.(cp_glob); @info "Removed checkpoint files.")
    @info "Simulation complete."
end
