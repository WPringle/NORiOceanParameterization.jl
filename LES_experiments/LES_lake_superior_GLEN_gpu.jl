"""
Large Eddy Simulation (LES) of the Lake Superior Eastern Mooring — Single GPU
Forced by GLEN Stannard Rock half-hourly observations (gap-filled product).

Replaces idealised constant wind / diurnal heat-flux forcing with measured
half-hourly fluxes from the GLEN Stannard Rock buoy gap-filled product, which
provides the momentum flux and turbulent heat fluxes DIRECTLY — no bulk-formula
(Charnock + MOST) computation is needed.  The only term still computed online is
the upwelling longwave radiation (depends on the live surface temperature):
  - Net upward heat flux: Q_net = SHF + LHF − SW_net − LW↓ + LW↑(T_sfc)
  - Wind stress read directly: τ_kin = −momentum_flux / ρ₀

Forcing source flag (`--forcing_source` = direct | coare_wind):
    direct      : measured eddy-covariance momentum_flux / sensible_heat_flux /
                  latent_heat_flux.
    coare_wind  : the *_coare_wind variants — momentum/SHF/LHF re-derived from the
                  measured wind speed via the COARE bulk algorithm.
    The downwelling SW/LW radiation is identical for both sources.

Grid: Nx × Ny = 128 × 128 at 2 m isotropic resolution (default 256 × 256 m).
       Lz from --Lz flag (default 212 m for eastern mooring).

Usage:
    julia --project=<project> run_LES_lake_superior_GLEN_gpu.jl <winter_year> [options]

    Positional:
        winter_year   e.g. 2009, 2010, 2011, 2014

    Options:
        --forcing_source  direct | coare_wind  (default: direct)
        --closure       Subgrid closure: none (default) | SmagorinskyLilly | DynamicSmagorinsky.
                        When set, the closure name is appended to the output directory.
        --Lz            Domain depth (m), default 212
        --Nx            Grid points in x (power of 2), default 128
        --Ny            Grid points in y (power of 2), default 128
        --sim_days      Simulation length (days), default 60
        --dt            Initial timestep (seconds), default 0.1
        --max_dt        Maximum timestep (minutes), default 2
        --time_interval Timeseries output interval (minutes), default 10
        --checkpoint_interval  Checkpoint interval (days), default 1
        --pickup        Pickup from latest checkpoint, default true
        --file_location Root directory for output files, default "."
"""

using CUDA
using Oceananigans
using Oceananigans.Units
using Oceananigans.Operators
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.BuoyancyFormulations
using Oceananigans.TurbulenceClosures: SmagorinskyLilly, DynamicSmagorinsky
using SeawaterPolynomials.TEOS10
using SeawaterPolynomials
using GibbsSeaWater
using JLD2
using NCDatasets
using FileIO
using Printf
using Random
using Statistics
using ArgParse

import Dates
using Dates: DateTime, Date, Day

#####
##### Command Line Argument Parsing
#####

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "winter_year"
            help = "Winter year (e.g. 2009, 2010, 2011, 2014)"
            arg_type = Int
            required = true
        "--forcing_source"
            help = "Flux source: direct (measured EC) | coare_wind (COARE bulk from wind)"
            arg_type = String
            default = "direct"
        "--closure"
            help = "Subgrid closure: none (default, resolved LES) | SmagorinskyLilly | DynamicSmagorinsky"
            arg_type = String
            default = ""
        "--Lz"
            help = "Domain depth (m). Eastern mooring: 212."
            arg_type = Float64
            default = 212.0
        "--Nx"
            help = "Number of grid points in x-direction (should be power of 2)"
            arg_type = Int64
            default = 128
        "--Ny"
            help = "Number of grid points in y-direction (should be power of 2)"
            arg_type = Int64
            default = 128
        "--sim_days"
            help = "Simulation length (days)"
            arg_type = Int
            default = 60
        "--dt"
            help = "Initial timestep (seconds)"
            arg_type = Float64
            default = 0.1
        "--max_dt"
            help = "Maximum timestep (minutes)"
            arg_type = Float64
            default = 2.0
        "--time_interval"
            help = "Time interval for time series output (minutes)"
            arg_type = Float64
            default = 10.0
        "--checkpoint_interval"
            help = "Time interval for checkpoint files (days)"
            arg_type = Float64
            default = 1.0
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

const winter_year = args["winter_year"]

# Forcing source: "direct" (measured EC) or "coare_wind" (COARE bulk from wind).
const FORCING_SOURCE = let s = lowercase(strip(args["forcing_source"]))
    s == "coare" ? "coare_wind" : s
end
FORCING_SOURCE in ("direct", "coare_wind") ||
    error("Unknown forcing_source '$(args["forcing_source"])'. Options: direct | coare_wind")
const SRC_TAG = "_" * FORCING_SOURCE  # filename tag: "_direct" or "_coare_wind"
@info "Forcing source = $FORCING_SOURCE"

# Subgrid-scale closure: none (default, resolved LES) or a Smagorinsky variant.
# Returns (closure_object, filename_tag).
function build_closure(name)
    isempty(name) && return nothing, ""
    n = lowercase(name)
    if n == "smagorinskylilly"
        return SmagorinskyLilly(), "_SmagorinskyLilly"
    elseif n == "dynamicsmagorinsky"
        # Directional (x,y) averaging of the dynamic coefficient — natural for a
        # horizontally-periodic column LES.
        return DynamicSmagorinsky(averaging = (1, 2)), "_DynamicSmagorinsky"
    else
        error("Unknown closure '$name'. Options: SmagorinskyLilly | DynamicSmagorinsky")
    end
end
const closure_model, CLOSURE_TAG = build_closure(strip(args["closure"]))
@info "Closure = $(isnothing(closure_model) ? "none (resolved LES)" : strip(args["closure"]))"

Random.seed!(123)

#####
##### CSV: look up isothermal (start) date
#####

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
            return DateTime(Date(strip(parts[2])))
        end
        nothing
    end
end

t_iso = read_isothermal_date(CSV_FILE, winter_year)
isnothing(t_iso) &&
    error("Winter year $winter_year not found in $CSV_FILE")
@info "Winter $winter_year: isothermal (start) date = $t_iso"

#####
##### Domain & Grid
#####

const Δ  = 2.0    # isotropic resolution (m)
const Lz = args["Lz"]
const Nz = Int(Lz / Δ)

const Nx = args["Nx"]
const Ny = args["Ny"]
const Lx = Nx * Δ
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
const ρ₀  = eos.reference_density
const g   = 9.80665

const S_lake   = 0.05    # g/kg (Absolute Salinity)
const T_insitu = 4.0     # °C (initial in-situ temperature)
const cₚ       = 4182.0  # J/(kg·K)
const ρ_air    = 1.225   # kg/m³ (used to convert momentum flux ↔ u*)

# Radiation constants
const albedo_sw = 0.08
const ε_water   = 0.98
const σ_SB      = 5.67e-8  # W/(m²·K⁴)

function Θ_conservative(z::Float64)
    p_dbar = ρ₀ * g * abs(z) / 1e4
    return gsw_ct_from_t(S_lake, T_insitu, p_dbar)
end

# Coriolis: 47° 32.2' N (Stannard Rock / eastern mooring)
const lat = 47.0 + 32.2 / 60.0
const Ω   = 7.2921150e-5
const f₀  = 2 * Ω * sind(lat)

#####
##### Load GLEN half-hourly data
#####

# Gap-filled product: momentum/heat fluxes are provided directly.  Choose between
# the measured (`direct`) and COARE-bulk-from-wind (`coare_wind`) flux variants.
const GLEN_FILE = "/lus/eagle/projects/COMPASS-GLM/wpringle/Lake_Julia_Runs/GLEN/" *
                  "US_StannardRockSuperior_processed_halfhourly_qc_gapfilled.nc"

# Variable names that differ between the two forcing sources.
const VAR_SUFFIX = FORCING_SOURCE == "direct" ? "" : "_coare_wind"
const LHF_VAR    = "latent_heat_flux"   * VAR_SUFFIX
const SHF_VAR    = "sensible_heat_flux" * VAR_SUFFIX
const MOM_VAR    = "momentum_flux"      * VAR_SUFFIX

glen_times_all, lhf_all, shf_all, sw_all, lw_all, mom_all, wspd_all =
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
        load_var("wind_speed")
    end
@info "GLEN dataset: $(glen_times_all[1]) → $(glen_times_all[end]) ($(length(glen_times_all)) half-hourly records)"
@info "Flux variables: SHF=$SHF_VAR  LHF=$LHF_VAR  τ=$MOM_VAR"

#####
##### Slice to simulation window
#####

const sim_days = args["sim_days"]
t_end = t_iso + Day(sim_days)
idx   = findall(t -> t_iso <= t <= t_end, glen_times_all)
isempty(idx) && error("No GLEN data in window $t_iso → $t_end")
@info "GLEN window: $(glen_times_all[idx[1]]) → $(glen_times_all[idx[end]]) ($(length(idx)) records)"

glen_t  = glen_times_all[idx]
lhf_raw = lhf_all[idx]
shf_raw = shf_all[idx]
sw_raw  = sw_all[idx]
lw_raw  = lw_all[idx]
mom_raw = mom_all[idx]
ws_raw  = wspd_all[idx]

#####
##### Fill missing values (linear interpolation, flat extrapolation at edges)
#####

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
lhf_f = fill_missing_linear(lhf_raw)
shf_f = fill_missing_linear(shf_raw)
sw_f  = fill_missing_linear(sw_raw)
lw_f  = fill_missing_linear(lw_raw)
mom_f = fill_missing_linear(mom_raw)
ws_f  = fill_missing_linear(ws_raw)

#####
##### Build flux time series
#####
# Simulation time axis: seconds since t_iso
const t_forcing = Float64[Dates.value(t - t_iso) / 1000.0 for t in glen_t]

# Pre-computed heat flux terms (W/m²), positive upward = lake cooling.
# LW_up = ε·σ·T_sfc⁴ is excluded here because it depends on the live surface
# temperature; it is added inside the discrete boundary-condition function below.
#   SW_net = (1 - albedo) · SW↓
#   Q_precomp = SHF + LHF - SW_net - LW↓
const SW_net_vals     = @. (1.0 - albedo_sw) * sw_f
const Q_precomp_Wm2   = @. shf_f + lhf_f - SW_net_vals - ε_water * lw_f
const Q_precomp_kin   = @. Q_precomp_Wm2 / (ρ₀ * cₚ)

# Kinematic momentum flux (m²/s²), negative = into lake surface.
# mom_f is the measured/derived surface stress (Pa = kg m⁻¹ s⁻², positive into lake):
#   τ_kin = -mom_f / ρ₀
# Friction velocity (diagnostic only) follows from u* = sqrt(τ / ρ_air).
const τ_kin_vals = @. -mom_f / ρ₀
const ustar_vals = @. sqrt(max(mom_f, 0.0) / ρ_air)

@info "Forcing summary ($(sim_days)-day window):"
@info "  Q_precomp (W/m²): mean=$(round(mean(Q_precomp_Wm2), digits=1))" *
      "  min=$(round(minimum(Q_precomp_Wm2), digits=1))  max=$(round(maximum(Q_precomp_Wm2), digits=1))"
@info "  LW_up at 4 °C (W/m²): $(round(ε_water * σ_SB * (4.0 + 273.15)^4, digits=1))  (added online from T_sfc)"
@info "  u*     (m/s)  : mean=$(round(mean(ustar_vals), digits=4))  max=$(round(maximum(ustar_vals), digits=4))"
@info "  U_meas (m/s)  : mean=$(round(mean(ws_f), digits=2))  max=$(round(maximum(ws_f), digits=2))"

#####
##### Linear interpolation helper
#####

@inline function gpu_searchsortedlast(arr, val)
    lo = 1
    hi = length(arr)
    while lo <= hi
        mid = (lo + hi) ÷ 2
        if @inbounds arr[mid] <= val
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return hi
end

@inline function interp_linear(t_arr, v_arr, t::Float64)
    n = length(t_arr)
    i = gpu_searchsortedlast(t_arr, t)
    i <= 0 && return @inbounds v_arr[1]
    i >= n && return @inbounds v_arr[n]
    @inbounds α = (t - t_arr[i]) / (t_arr[i+1] - t_arr[i])
    return @inbounds v_arr[i] + α * (v_arr[i+1] - v_arr[i])
end

#####
##### Surface Forcing — discrete BCs for 3D NonhydrostaticModel
#####

# Heat flux: Q_precomp(t) + LW_up(T_sfc), discrete form to access live surface T
@inline function Qᵀ_obs(i, j, grid, clock, model_fields, p)
    T_sfc   = @inbounds model_fields.T[i, j, p.Nz]
    LW_up   = p.ε_water * p.σ_SB * (T_sfc + 273.15)^4 / (p.ρ₀ * p.cₚ)
    return interp_linear(p.t_forcing, p.Q_precomp_kin, Float64(clock.time)) + LW_up
end

# Momentum flux: time-interpolated, no state dependence but use discrete form for consistency
@inline function Qᵁ_obs(i, j, grid, clock, model_fields, p)
    return interp_linear(p.t_forcing, p.τ_kin_vals, Float64(clock.time))
end

const t_forcing_gpu    = CuArray(t_forcing)
const Q_precomp_kin_gpu = CuArray(Q_precomp_kin)
const τ_kin_vals_gpu    = CuArray(τ_kin_vals)

T_bcs = FieldBoundaryConditions(
            top = FluxBoundaryCondition(Qᵀ_obs,
                      discrete_form = true,
                      parameters = (; t_forcing = t_forcing_gpu,
                                      Q_precomp_kin = Q_precomp_kin_gpu,
                                      Nz, ρ₀, cₚ, ε_water, σ_SB)))

u_bcs = FieldBoundaryConditions(
            top = FluxBoundaryCondition(Qᵁ_obs,
                      discrete_form = true,
                      parameters = (; t_forcing = t_forcing_gpu,
                                      τ_kin_vals = τ_kin_vals_gpu)))

S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0))

#####
##### Initial Conditions (uniform, with tiny noise to seed turbulence)
#####

noise(x, y, z) = rand() * exp(z / 8)

T_initial(x, y, z) = Θ_conservative(z) + 1e-6 * noise(x, y, z)

#####
##### Model
#####

model = NonhydrostaticModel(
    grid               = grid,
    closure            = closure_model,
    coriolis           = FPlane(f = f₀),
    buoyancy           = SeawaterBuoyancy(equation_of_state = eos, constant_salinity = S_lake),
    tracers            = (:T),
    timestepper        = :RungeKutta3,
    advection          = WENO(order = 9),
    boundary_conditions = (T = T_bcs, u = u_bcs),
)

set!(model, T = T_initial)

T = model.tracers.T
u, v, w = model.velocities

#####
##### Output Directory
#####

FILE_NAME = "LES_GLEN_winter$(winter_year)$(SRC_TAG)$(CLOSURE_TAG)_Lxy$(round(Int,Lx))_Lz$(round(Int,Lz))_Nxy$(Nx)_Nz$(Nz)"
FILE_DIR  = joinpath(args["file_location"], FILE_NAME)
mkpath(FILE_DIR)

#####
##### Simulation
#####

simulation = Simulation(model,
                        Δt        = args["dt"]second,
                        stop_time = sim_days * days)

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
    file["metadata/winter_year"]           = winter_year
    file["metadata/forcing_source"]        = FORCING_SOURCE
    file["metadata/closure"]               = isnothing(closure_model) ? "none" : string(strip(args["closure"]))
    file["metadata/isothermal_date"]       = string(t_iso)
    file["metadata/sim_days"]              = sim_days
    file["metadata/salinity_flux"]         = 0.0
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

b_op = KernelFunctionOperation{Center, Center, Face}( 
                ∂z_b, grid, model.buoyancy, model.tracers)
b = Field(b_op)

#####
##### Horizontally-Averaged Profiles
#####

ubar = Field(Average(u, dims = (1, 2)))
vbar = Field(Average(v, dims = (1, 2)))
Tbar = Field(Average(T, dims = (1, 2)))
bbar = Field(Average(b, dims = (1, 2)))

uw = Field(Average(w * u, dims = (1, 2)))
vw = Field(Average(w * v, dims = (1, 2)))
wb = Field(Average(w * b, dims = (1, 2)))
wT = Field(Average(w * T, dims = (1, 2)))

timeseries_outputs = (; ubar, vbar, Tbar, bbar,
                        uw, vw, wT, wb)

#####
##### Output Writers
#####

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

simulation.output_writers[:b] = JLD2OutputWriter(model, (; b),
                                                 filename = "$(FILE_DIR)/instantaneous_fields_b.jld2",
                                                 schedule = field_schedule,
                                                 with_halos = true,
                                                 init = init_save_some_metadata!)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/instantaneous_timeseries.jld2",
                                                          schedule = TimeInterval(args["time_interval"]minutes),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:hourly_avg] = JLD2OutputWriter(model, timeseries_outputs,
                                                          filename = "$(FILE_DIR)/hourly_averaged_timeseries.jld2",
                                                          schedule = AveragedTimeInterval(1hours),
                                                          with_halos = true,
                                                          init = init_save_some_metadata!)

simulation.output_writers[:checkpointer] = Checkpointer(model,
    schedule = TimeInterval(args["checkpoint_interval"]days),
    prefix   = "$(FILE_DIR)/model_checkpoint")

#####
##### Save GLEN forcing used in this run
#####

forcing_file = joinpath(FILE_DIR, "forcing_winter$(winter_year)$(SRC_TAG).jld2")
jldopen(forcing_file, "w") do f
    f["t_iso"]           = string(t_iso)
    f["forcing_source"]  = FORCING_SOURCE
    f["t_forcing"]       = t_forcing
    f["Q_precomp_Wm2"]   = Q_precomp_Wm2
    f["Q_precomp_kin"]   = Q_precomp_kin
    f["SW_net_vals"]     = SW_net_vals
    f["mom_flux"]        = mom_f
    f["ustar_vals"]      = ustar_vals
    f["tau_kin_vals"]    = τ_kin_vals
    f["lhf"]             = lhf_f
    f["shf"]             = shf_f
    f["sw_down"]         = sw_f
    f["lw_down"]         = lw_f
    f["wspd_meas"]       = ws_f
    f["albedo_sw"]       = albedo_sw
    f["eps_water"]       = ε_water
end
@info "Saved forcing → $forcing_file"

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

# Clean up checkpoint files after successful completion.
# (Use readdir/filter rather than glob: FILE_DIR is an absolute path and
#  Glob.jl rejects patterns that start with "/".)
cp_cleanup = filter(f -> occursin("model_checkpoint_iteration", f), readdir(FILE_DIR))
if !isempty(cp_cleanup)
    @info "Removing checkpoint files..."
    rm.(joinpath.(FILE_DIR, cp_cleanup))
end

@info "Simulation completed successfully!"
