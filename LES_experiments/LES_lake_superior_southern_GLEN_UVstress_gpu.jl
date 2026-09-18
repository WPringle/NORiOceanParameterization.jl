"""
Large Eddy Simulation (LES) of the Lake Superior SOUTHERN Mooring — Single GPU
Forced by GLEN Stannard Rock half-hourly observations (gap-filled product),
DIRECTIONAL (UV-stress) variant.

This is the southern-mooring companion of LES_lake_superior_GLEN_UVstress_gpu.jl.
It shares that script's physics (GLEN surface fluxes, directional wind stress split
into Qᵁ/Qᵛ from wind_direction, optional Smagorinsky / WENO5 closure and quadratic
bottom drag) and differs only in the southern-mooring configuration:

  * Location  : 47° 2.0' N  (f₀ from this latitude)
  * Depth     : year-dependent bottom depth (from the mooring's own sensor data):
                  winter 2010 → 380 m,  winter 2011 → 384 m
                (override with --Lz).
  * Start date: uses the EASTERN mooring's isothermal dates (the much deeper
                southern column rarely reaches a clean whole-column isothermal
                state — same choice as the southern TURB experiment).
  * Initial T : the SOUTHERN mooring's *observed* profile at the start date
                (Austin2023 SM*.mat, natural cubic spline in depth), NOT uniform
                4 °C.  For winter 2010 the sensor depths are lifted up 20 m
                (dep − 20) to correct the SMS09h `dep` offset; winter 2011 is
                used unadjusted.

Wind stress decomposition (unchanged from the eastern UVstress script):
  wind_direction (θ) is meteorological — direction the wind blows FROM, cw from
  north.  Downwind stress unit vector (êₓ, êᵧ) = (−sinθ, −cosθ); top-flux BC
  value = −τ_component / ρ₀:
      Qᵁ = (momentum_flux / ρ₀) · sinθ = τ_mag · êₓ      [m²/s²]
      Qᵛ = (momentum_flux / ρ₀) · cosθ = τ_mag · êᵧ      [m²/s²]
  Direction gap-filling is done on the Cartesian unit components (sinθ, cosθ) to
  respect 0°/360° cyclicity; the magnitude is filled separately as a scalar.

Forcing source flag (`--forcing_source` = direct | coare_wind):
    direct      : measured eddy-covariance momentum_flux / sensible/latent heat.
    coare_wind  : the *_coare_wind variants (COARE bulk from wind).  SW/LW down
                  and wind_direction are identical for both sources.

Grid: Nx × Ny = 128 × 128 at 2 m isotropic resolution (default 256 × 256 m).

Usage:
    julia --project=<project> run_..._southern_gpu.jl <winter_year> [options]

    Positional:
        winter_year   2010 | 2011  (2009 unsupported: SMS09h `dep` is unreliable)

    Options:
        --forcing_source  direct | coare_wind  (default: direct)
        --closure       none (default) | SmagorinskyLilly | DynamicSmagorinsky | WENO5
        --Cd            Quadratic bottom drag coefficient (default 0.0 = free-slip)
        --Lz            Domain depth (m). Default (<=0) = auto by year (2010→380, else→384)
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
using MAT
using NCDatasets
using FileIO
using Printf
using Random
using Statistics
using ArgParse

import Dates
using Dates: DateTime, Date, Day, Millisecond

#####
##### Command Line Argument Parsing
#####

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "winter_year"
            help = "Winter year (2010 | 2011)"
            arg_type = Int
            required = true
        "--forcing_source"
            help = "Flux source: direct (measured EC) | coare_wind (COARE bulk from wind)"
            arg_type = String
            default = "direct"
        "--closure"
            help = "Subgrid closure: none (default, resolved LES) | SmagorinskyLilly | DynamicSmagorinsky | WENO5 (no closure, WENO order 9→5)"
            arg_type = String
            default = ""
        "--Cd"
            help = "Quadratic bottom drag coefficient. 0.0 (default) = free-slip bottom, no drag."
            arg_type = Float64
            default = 0.0
        "--Lz"
            help = "Domain depth (m). Default (<=0) = auto by year: 2010→380, else→384 (southern mooring bottom depth)."
            arg_type = Float64
            default = -1.0
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

# Subgrid-scale closure choice.  Returns (closure_object, filename_tag, weno_order).
#   none (default)     : resolved LES, WENO order 9
#   SmagorinskyLilly   : Smagorinsky–Lilly closure, WENO order 9
#   DynamicSmagorinsky : dynamic Smagorinsky (x,y averaging), WENO order 9
#   WENO5              : no explicit closure; WENO advection order 9 → 5 (implicit LES)
function build_closure(name)
    isempty(name) && return nothing, "", 9
    n = lowercase(name)
    if n == "smagorinskylilly"
        return SmagorinskyLilly(), "_SmagorinskyLilly", 9
    elseif n == "dynamicsmagorinsky"
        return DynamicSmagorinsky(averaging = (1, 2)), "_DynamicSmagorinsky", 9
    elseif n == "weno5"
        return nothing, "_WENO5", 5
    else
        error("Unknown closure '$name'. Options: SmagorinskyLilly | DynamicSmagorinsky | WENO5")
    end
end
const closure_model, CLOSURE_TAG, WENO_ORDER = build_closure(strip(args["closure"]))
@info "Closure = $(isnothing(closure_model) ? "none (resolved LES)" : strip(args["closure"]))  |  WENO order = $WENO_ORDER"

# Quadratic bottom drag coefficient. 0.0 (default) = free-slip lake bed, no drag.
const Cd = args["Cd"]
const DRAG_TAG = Cd > 0 ? "_Cd$(Cd)" : ""
@info "Bottom drag = $(Cd > 0 ? "quadratic, Cd=$Cd" : "none (free-slip)")"

Random.seed!(123)

#####
##### CSV: look up isothermal (start) date  (EASTERN mooring dates — see docstring)
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
##### Domain & Grid  (year-dependent bottom depth)
#####

const Δ  = 2.0    # isotropic resolution (m)
# Southern mooring bottom depth differs by deployment (from the mooring's own
# sensor data): winter 2010 → 380 m, winter 2011 → 384 m.  --Lz overrides.
const Lz = let l = args["Lz"]
    l > 0 ? l : (winter_year == 2010 ? 380.0 : 384.0)
end
const Nz = Int(round(Lz / Δ))
@info "Domain depth Lz = $Lz m  (Nz = $Nz at Δ = $Δ m)"

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
const cₚ       = 4182.0  # J/(kg·K)
const ρ_air    = 1.225   # kg/m³ (used to convert momentum flux ↔ u*)

# Radiation constants
const albedo_sw = 0.08
const ε_water   = 0.98
const σ_SB      = 5.67e-8  # W/(m²·K⁴)

# Coriolis: 47° 2.0' N (southern mooring)
const lat = 47.0 + 2.0 / 60.0
const Ω   = 7.2921150e-5
const f₀  = 2 * Ω * sind(lat)

#####
##### Initial temperature: observed Southern Mooring profile (Austin2023 SM*.mat)
#####
# Built from the southern mooring's own sensor profile at t_iso (natural cubic
# spline in depth, constant extrapolation outside the sensor range).  For winter
# 2010 the sensor depths are lifted up 20 m (dep − 20) to correct the SMS09h `dep`
# offset; winter 2011 is used unadjusted.
const OBS_DIR    = "/lus/eagle/projects/COMPASS-GLM/wpringle/Lake_Julia_Runs/Austin2023"
const DEP_OFFSET = winter_year == 2010 ? 20.0 : 0.0   # metres to lift sensor depths up
@info "Southern-mooring IC: depth offset = $DEP_OFFSET m (dep → dep − offset)"

t2dt_obs(t::Real) = DateTime(2000, 1, 1) + Millisecond(round(Int64, (t - 730486.0) * 86_400_000.0))

function natural_cubic_spline(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    h = diff(x)
    a = zeros(n); b = ones(n); c = zeros(n); d = zeros(n)
    for i in 2:n-1
        a[i] = h[i-1]
        b[i] = 2 * (h[i-1] + h[i])
        c[i] = h[i]
        d[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
    end
    for i in 2:n-1   # Thomas algorithm (natural boundary: M[1] = M[n] = 0)
        w = a[i] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]
    end
    M = zeros(n)
    for i in (n-1):-1:2
        M[i] = (d[i] - c[i] * M[i+1]) / b[i]
    end
    return function (xq::Float64)
        xq <= x[1]   && return y[1]      # constant extrapolation above shallowest sensor
        xq >= x[end] && return y[end]    # constant extrapolation below deepest sensor
        i  = searchsortedlast(x, xq)
        hi = h[i]
        A  = (x[i+1] - xq) / hi
        B  = (xq - x[i]) / hi
        return A * y[i] + B * y[i+1] + ((A^3 - A) * M[i] + (B^3 - B) * M[i+1]) * hi^2 / 6
    end
end

# Find the Austin2023 SM*.mat file spanning t_target and build a depth → in-situ
# temperature spline from the sensor readings at (the nearest available time to) it.
# dep_offset (m) lifts sensor depths up: dep_used = dep − dep_offset.
function observed_profile_spline(t_target, dep_offset)
    for f in sort(readdir(OBS_DIR; join = true))
        isdir(f) || continue
        for fn in sort(readdir(f))
            (startswith(fn, "SM") && endswith(fn, ".mat")) || continue
            path = joinpath(f, fn)
            data = try matread(path) catch; continue end
            haskey(data, "dep") || continue
            t_raw = vec(Float64.(reshape(data["t"], :)))
            dts   = t2dt_obs.(t_raw)
            (dts[1] <= t_target <= dts[end]) || continue

            dep   = vec(Float64.(reshape(data["dep"], :))) .- dep_offset
            T_raw = data["T"]
            Ndep, Nt = length(dep), length(t_raw)
            T = size(T_raw) == (Ndep, Nt) ? Float64.(T_raw) : Float64.(T_raw')

            i = findfirst(==(t_target), dts)
            isnothing(i) && ((_, i) = findmin(abs.(Dates.value.(dts .- t_target))))
            col = T[:, i]
            any(isnan, col) && error("Observed Southern Mooring profile at $(dts[i]) in $fn has NaNs")

            idx = sortperm(dep)
            @info "Southern Mooring IC: observed profile from $fn at $(dts[i]) (dep offset $dep_offset m)"
            return natural_cubic_spline(dep[idx], col[idx])
        end
    end
    error("No Southern Mooring raw .mat file found spanning t_target = $t_target")
end

const T_obs_spline = observed_profile_spline(t_iso, DEP_OFFSET)

function Θ_conservative(z::Float64)
    p_dbar     = ρ₀ * g * abs(z) / 1e4
    T_insitu_z = T_obs_spline(abs(z))
    return gsw_ct_from_t(S_lake, T_insitu_z, p_dbar)
end

#####
##### Load GLEN half-hourly data
#####

# Gap-filled product: momentum/heat fluxes are provided directly.  Choose between
# the measured (`direct`) and COARE-bulk-from-wind (`coare_wind`) flux variants.
# wind_direction is shared (no *_coare_wind variant): coare_wind changes only the
# stress magnitude, not its direction.
const GLEN_FILE = "/lus/eagle/projects/COMPASS-GLM/wpringle/Lake_Julia_Runs/GLEN/" *
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

#####
##### Slice to simulation window
#####

const sim_days = args["sim_days"]
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
# Scalars (magnitudes) are safe to interpolate linearly.
lhf_f  = fill_missing_linear(lhf_raw)
shf_f  = fill_missing_linear(shf_raw)
sw_f   = fill_missing_linear(sw_raw)
lw_f   = fill_missing_linear(lw_raw)
mom_f  = fill_missing_linear(mom_raw)
ws_f   = fill_missing_linear(ws_raw)

# Wind direction is CYCLICAL — linearly interpolating degrees is wrong across the
# 0°/360° wrap.  Fill the Cartesian unit-vector components (êₓ = sinθ, êᵧ = cosθ),
# which are continuous, then renormalise back to a unit vector.
ex_raw = Union{Missing,Float64}[ismissing(d) ? missing : sind(d) for d in wdir_raw]
ey_raw = Union{Missing,Float64}[ismissing(d) ? missing : cosd(d) for d in wdir_raw]
ex_f   = fill_missing_linear(ex_raw)
ey_f   = fill_missing_linear(ey_raw)
enorm  = @. sqrt(ex_f^2 + ey_f^2)
ex_f   = @. ifelse(enorm > 0, ex_f / enorm, 0.0)   # unit êₓ = sinθ
ey_f   = @. ifelse(enorm > 0, ey_f / enorm, 0.0)   # unit êᵧ = cosθ

# Reconstructed direction (degrees, for diagnostics/saving): θ = atan2(sinθ, cosθ)
wdir_f = @. mod(atand(ex_f, ey_f), 360.0)

#####
##### Build flux time series
#####
# Simulation time axis: seconds since t_iso
const t_forcing = Float64[Dates.value(t - t_iso) / 1000.0 for t in glen_t]

# Pre-computed heat flux terms (W/m²), positive upward = lake cooling.
#   SW_net = (1 - albedo) · SW↓ ;  Q_precomp = SHF + LHF - SW_net - LW↓
# LW_up = ε·σ·T_sfc⁴ is added online inside the discrete BC below.
const SW_net_vals     = @. (1.0 - albedo_sw) * sw_f
const Q_precomp_Wm2   = @. shf_f + lhf_f - SW_net_vals - ε_water * lw_f
const Q_precomp_kin   = @. Q_precomp_Wm2 / (ρ₀ * cₚ)

# Directional kinematic momentum flux (m²/s²), split into x/y from wind_direction.
#     Qᵁ = (mom_f / ρ₀) · sinθ = τ_mag · êₓ ;  Qᵛ = (mom_f / ρ₀) · cosθ = τ_mag · êᵧ
# using the gap-filled unit-vector components êₓ, êᵧ built above (cyclicity-safe).
const τ_mag_kin = @. mom_f / ρ₀                 # kinematic stress magnitude (≥ 0)
const Qu_vals   = @. τ_mag_kin * ex_f
const Qv_vals   = @. τ_mag_kin * ey_f
const ustar_vals = @. sqrt(max(mom_f, 0.0) / ρ_air)

@info "Forcing summary ($(sim_days)-day window):"
@info "  Q_precomp (W/m²): mean=$(round(mean(Q_precomp_Wm2), digits=1))" *
      "  min=$(round(minimum(Q_precomp_Wm2), digits=1))  max=$(round(maximum(Q_precomp_Wm2), digits=1))"
@info "  u*     (m/s)  : mean=$(round(mean(ustar_vals), digits=4))  max=$(round(maximum(ustar_vals), digits=4))"
@info "  U_meas (m/s)  : mean=$(round(mean(ws_f), digits=2))  max=$(round(maximum(ws_f), digits=2))"
@info "  wind_dir (°)  : mean=$(round(mean(wdir_f), digits=1))  [meteorological, from-direction]"
@info "  |Qᵁ| (m²/s²)  : mean=$(round(mean(abs.(Qu_vals)), digits=6))  max=$(round(maximum(abs.(Qu_vals)), digits=6))"
@info "  |Qᵛ| (m²/s²)  : mean=$(round(mean(abs.(Qv_vals)), digits=6))  max=$(round(maximum(abs.(Qv_vals)), digits=6))"

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

# x-momentum flux: time-interpolated Qᵁ
@inline function Qᵁ_obs(i, j, grid, clock, model_fields, p)
    return interp_linear(p.t_forcing, p.Qu_vals, Float64(clock.time))
end

# y-momentum flux: time-interpolated Qᵛ
@inline function Qᵛ_obs(i, j, grid, clock, model_fields, p)
    return interp_linear(p.t_forcing, p.Qv_vals, Float64(clock.time))
end

# Quadratic bottom drag: τ = -Cd*|u|*u (and v analogue), near-bed (k=1) velocity.
@inline function u_bottom_drag(i, j, grid, clock, model_fields, p)
    u1 = @inbounds model_fields.u[i, j, 1]
    v1 = ℑxyᶠᶜᵃ(i, j, 1, grid, model_fields.v)
    return -p.Cd * u1 * sqrt(u1^2 + v1^2)
end

@inline function v_bottom_drag(i, j, grid, clock, model_fields, p)
    v1 = @inbounds model_fields.v[i, j, 1]
    u1 = ℑxyᶜᶠᵃ(i, j, 1, grid, model_fields.u)
    return -p.Cd * v1 * sqrt(u1^2 + v1^2)
end

const t_forcing_gpu     = CuArray(t_forcing)
const Q_precomp_kin_gpu = CuArray(Q_precomp_kin)
const Qu_vals_gpu       = CuArray(Qu_vals)
const Qv_vals_gpu       = CuArray(Qv_vals)

T_bcs = FieldBoundaryConditions(
            top = FluxBoundaryCondition(Qᵀ_obs,
                      discrete_form = true,
                      parameters = (; t_forcing = t_forcing_gpu,
                                      Q_precomp_kin = Q_precomp_kin_gpu,
                                      Nz, ρ₀, cₚ, ε_water, σ_SB)))

u_bottom_bc = Cd > 0 ? FluxBoundaryCondition(u_bottom_drag, discrete_form = true, parameters = (; Cd)) : nothing
v_bottom_bc = Cd > 0 ? FluxBoundaryCondition(v_bottom_drag, discrete_form = true, parameters = (; Cd)) : nothing

u_bcs = FieldBoundaryConditions(
            top    = FluxBoundaryCondition(Qᵁ_obs,
                      discrete_form = true,
                      parameters = (; t_forcing = t_forcing_gpu,
                                      Qu_vals = Qu_vals_gpu)),
            bottom = u_bottom_bc)

v_bcs = FieldBoundaryConditions(
            top    = FluxBoundaryCondition(Qᵛ_obs,
                      discrete_form = true,
                      parameters = (; t_forcing = t_forcing_gpu,
                                      Qv_vals = Qv_vals_gpu)),
            bottom = v_bottom_bc)

S_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(0.0))

#####
##### Initial Conditions (observed profile + tiny noise to seed turbulence)
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
    advection          = WENO(order = WENO_ORDER),
    boundary_conditions = (T = T_bcs, u = u_bcs, v = v_bcs),
)

set!(model, T = T_initial)

T = model.tracers.T
u, v, w = model.velocities

#####
##### Output Directory
#####

FILE_NAME = "LES_GLEN_southern_winter$(winter_year)$(SRC_TAG)$(CLOSURE_TAG)$(DRAG_TAG)_UVstress_Lxy$(round(Int,Lx))_Lz$(round(Int,Lz))_Nxy$(Nx)_Nz$(Nz)"
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
    file["metadata/mooring"]               = "southern"
    file["metadata/coriolis_parameter"]    = f₀
    file["metadata/latitude_deg"]          = lat
    file["metadata/winter_year"]           = winter_year
    file["metadata/forcing_source"]        = FORCING_SOURCE
    file["metadata/closure"]               = isnothing(closure_model) ? "none" : string(strip(args["closure"]))
    file["metadata/bottom_drag_Cd"]        = Cd
    file["metadata/weno_order"]            = WENO_ORDER
    file["metadata/momentum_decomposition"] = "directional (Qu, Qv from wind_direction)"
    file["metadata/initial_condition"]     = "observed Southern Mooring profile (Austin2023)"
    file["metadata/dep_offset_m"]          = DEP_OFFSET
    file["metadata/isothermal_date"]       = string(t_iso)
    file["metadata/sim_days"]              = sim_days
    file["metadata/salinity_flux"]         = 0.0
    file["metadata/surface_salinity_gkg"]  = S_lake
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
    f["tau_mag_kin"]     = τ_mag_kin
    f["Qu_vals"]         = Qu_vals
    f["Qv_vals"]         = Qv_vals
    f["wind_dir_deg"]    = wdir_f
    f["lhf"]             = lhf_f
    f["shf"]             = shf_f
    f["sw_down"]         = sw_f
    f["lw_down"]         = lw_f
    f["wspd_meas"]       = ws_f
    f["albedo_sw"]       = albedo_sw
    f["eps_water"]       = ε_water
    f["dep_offset_m"]    = DEP_OFFSET
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
