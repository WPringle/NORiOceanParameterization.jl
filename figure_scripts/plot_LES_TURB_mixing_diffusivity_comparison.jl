#####
##### GLEN-forced Eastern + Southern Mooring — LES vs TURB k-ε mixing (eddy
##### diffusivity) comparison
#####
# Companion to plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl:
# that figure shows the two model families produce different temperature
# profiles; this one shows *how much vertical mixing* each model actually
# applies, as two effective diffusivity profiles over three windows in the
# run: day 30 (29 ≤ t < 30), day 60 (59 ≤ t < 60), and a day-30-to-60 mean
# (30 ≤ t < 60, deliberately excluding the noisy early LES spin-up transient
# — see load_all_nanmean/load_turb_diffusivity_nanmean/load_les_diffusivity_nanmean
# below):
#
#   • Heat/tracer diffusivity κ_T(z)   — directly comparable mixing strength
#   • Momentum diffusivity   κ_u(z)    — the shear-driven mixing that feeds
#                                         into Ri/stratification in these
#                                         closures
#
# TURB (k-ε / CATKE column models) carries κ_T, κ_u internally as the
# closure's own κc/κu diffusivity fields (saved as κcbar/κubar — see the
# κubar/κcbar additions in TURB_experiments/lake_superior_*_mooring_*.jl).
#
# The production LES runs use no explicit subgrid closure (resolved LES,
# implicit WENO9 dissipation), so there is no native eddy viscosity field.
# Instead an *effective* diffusivity is inferred from the resolved turbulent
# fluxes already saved in hourly_averaged_timeseries.jld2, via the standard
# down-gradient relation flux = -κ·∂(mean)/∂z, solved for κ:
#
#   κ_T,eff(z) = -⟨w'T'⟩ / ∂⟨T⟩/∂z
#   κ_u,eff(z) = -(⟨u'w'⟩·∂⟨u⟩/∂z + ⟨v'w'⟩·∂⟨v⟩/∂z) / ((∂⟨u⟩/∂z)² + (∂⟨v⟩/∂z)²)
#
# (κ_u,eff generalizes the 1D flux/gradient ratio to two velocity components by
# projecting the momentum flux vector onto the shear vector — the same S² =
# (∂u/∂z)² + (∂v/∂z)² combination already used for the Ri diagnostic in the
# TURB experiment scripts.) κ is formed at *each hourly snapshot* within a
# window and then nanmean'd over that window at each depth — not a ratio of
# window-averaged flux/gradient fields, which was tried first and discarded:
# that approach lets a single noisy window-mean gradient blank out an entire
# depth, whereas forming the ratio hourly and ignoring NaNs when averaging
# only ever throws out the individual hours actually near a zero-gradient
# crossing. Per-hour values are masked (→ NaN) wherever the denominator is
# small relative to its own profile's peak — division-by-near-zero at
# mixed-layer edges — or non-positive, since a down-gradient diffusivity has
# no physical meaning there.
#
# Inputs  : data/TURB_outputs/{eastern,southern}_mooring_GLEN_forced/winter<YEAR>/
#               kepsilon[_coare_wind][_UVstress][_EOSteos10cabbeling]_winter<YEAR>.jld2
#               (κubar, κcbar — requires the TURB experiment scripts to have
#               been rerun after the κu/κc output was added)
#           data/LES_outputs/{eastern,southern}_mooring_GLEN/.../hourly_averaged_timeseries.jld2
#               (ubar, vbar, Tbar, uw, vw, wT — already present, no rerun needed)
# Outputs : figures/LES_TURB_eastern_southern_mooring_comparison_heat_diffusivity_profiles_day{30,60,30to60mean}_{direct,coare_wind}.pdf
#           figures/LES_TURB_eastern_southern_mooring_comparison_momentum_diffusivity_profiles_day{30,60,30to60mean}_{direct,coare_wind}.pdf
#
# Usage:
#   julia plot_LES_TURB_mixing_diffusivity_comparison.jl               # direct (default)
#   julia plot_LES_TURB_mixing_diffusivity_comparison.jl coare_wind
#####

using Oceananigans
using CairoMakie
using JLD2
using Statistics
using LaTeXStrings

const FORCING_SOURCE = length(ARGS) >= 1 ? ARGS[1] : "direct"
FORCING_SOURCE ∈ ("direct", "coare_wind") ||
    error("FORCING_SOURCE must be \"direct\" or \"coare_wind\", got \"$(FORCING_SOURCE)\"")
const SRC_TAG = FORCING_SOURCE == "direct" ? "" : "_coare_wind"

const TURB_DIR    = joinpath(@__DIR__, "..", "data", "TURB_outputs", "eastern_mooring_GLEN_forced")
const TURB_DIR_SM = joinpath(@__DIR__, "..", "data", "TURB_outputs", "southern_mooring_GLEN_forced")
const LES_DIR      = joinpath(@__DIR__, "..", "data", "LES_outputs", "eastern_mooring_GLEN")
const LES_DIR_SM   = joinpath(@__DIR__, "..", "data", "LES_outputs", "southern_mooring_GLEN")
FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

const winter_years    = [2009, 2010, 2011, 2014, 2015]   # eastern mooring
const winter_years_sm = [2010, 2011]                     # southern mooring

# k-ε output file for a given winter, UV-stress tag, and EOS tag — same scheme
# as plot_LES_TURB_eastern_southern_mooring_comparison_T_profiles.jl.
turb_file(year, uv_tag, eos_tag) = joinpath(TURB_DIR, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")
turb_file_sm(year, uv_tag, eos_tag) = joinpath(TURB_DIR_SM, "winter$(year)",
    "kepsilon$(SRC_TAG)$(uv_tag)$(eos_tag)_winter$(year).jld2")

function les_file(year)
    isdir(LES_DIR) || return nothing
    for dir in readdir(LES_DIR, join = true)
        if isdir(dir) && contains(dir, "LES_GLEN_winter$(year)_$(FORCING_SOURCE)_UVstress")
            return joinpath(dir, "hourly_averaged_timeseries.jld2")
        end
    end
    return nothing
end

function les_file_sm(year)
    isdir(LES_DIR_SM) || return nothing
    for dir in readdir(LES_DIR_SM, join = true)
        if isdir(dir) && contains(dir, "LES_GLEN_southern_winter$(year)_$(FORCING_SOURCE)_UVstress")
            return joinpath(dir, "hourly_averaged_timeseries.jld2")
        end
    end
    return nothing
end

#####
##### The three k-ε variants — identical to the T-profile comparison figure
#####
const VARIANTS = [
    (uv_tag = "_UVstress", eos_tag = "",                   color = :steelblue4, label = "k-ε",                               linestyle = :solid),
    (uv_tag = "",          eos_tag = "",                   color = :darkorange, label = "k-ε (unidirectional wind stress)",  linestyle = :dash),
    (uv_tag = "_UVstress", eos_tag = "_EOSteos10cabbeling", color = :seagreen,   label = "k-ε (No thermobaricity)",           linestyle = :dot),
]
const LES_COLOR = :black
const LES_LABEL = "LES"

#####
##### Time-averaging + finite-difference helpers
#####

# Indices of a FieldTimeSeries' snapshots falling in [day_lo, day_hi) (days
# since start). If none fall in that window (e.g. a run ending at t=59.96
# instead of 60.0), falls back to the single nearest-in-time snapshot.
function window_indices(fts, day_lo, day_hi)
    td   = Float64.(fts.times) ./ 86400.0
    idxs = findall(t -> day_lo <= t < day_hi, td)
    if isempty(idxs)
        _, i = findmin(abs.(td .- min(day_hi, maximum(td))))
        idxs = [i]
    end
    return idxs
end

# Mean of a vector of values, ignoring NaNs; NaN if all are NaN.
function nanmean(v)
    vals = filter(isfinite, v)
    return isempty(vals) ? NaN : mean(vals)
end

# Elementwise nanmean across a vector of equal-length profile vectors.
function nanmean_profiles(profiles)
    Nz = length(profiles[1])
    return [nanmean([p[k] for p in profiles]) for k in 1:Nz]
end

# d(profile)/dz for a Center-located profile, evaluated at the Nz-1 *interior*
# Face points (dropping the top/bottom boundary faces, where a flux-BC-driven
# gradient/diffusivity isn't a meaningful interior mixing diagnostic).
function center_diff_interior(profile_center, zC)
    Nz = length(profile_center)
    return [(profile_center[k] - profile_center[k-1]) / (zC[k] - zC[k-1]) for k in 2:Nz]
end

interior_faces(zF) = zF[2:end-1]

# κ = -num/denom, masked (→ NaN) wherever |denom| is small relative to its own
# profile's peak magnitude (division-by-near-zero at mixed-layer edges) or the
# result is non-positive (no physical meaning for a down-gradient closure).
function flux_gradient_ratio(num, denom; frac = 0.02)
    thresh = frac * maximum(abs, denom)
    out = similar(num)
    for i in eachindex(num)
        v = -num[i] / denom[i]
        out[i] = (abs(denom[i]) < thresh || !(v > 0)) ? NaN : v
    end
    return out
end

# TURB's native closure diffusivities are already ≥ 0; floor tiny/zero values
# to NaN so they don't appear as -Inf gaps on a log-scale axis.
clean_diffusivity(κ; floor = 1e-8) = [v <= floor ? NaN : v for v in κ]

struct DiffProfiles
    z   :: Vector{Float64}   # interior-face depths (m, ≤ 0)
    κ_T :: Vector{Float64}   # heat/tracer diffusivity (m²/s)
    κ_u :: Vector{Float64}   # momentum diffusivity (m²/s)
end

get_profile(dp::DiffProfiles, field::Symbol) = field === :κ_T ? dp.κ_T : dp.κ_u

# Diffusivity over any [day_lo, day_hi) window, computed the robust way: κ is
# formed at *each* hourly snapshot (so the NaN-masking in
# flux_gradient_ratio/clean_diffusivity only ever throws out that one hour at
# that one depth, wherever its gradient happens to pass through ~0), then
# `nanmean`d over the window at each depth. That's what lets a depth that's
# only occasionally near a zero-gradient crossing still get a well-defined
# mean, instead of a single time-averaged-fields-then-ratio estimate (the
# earlier approach) being thrown out entirely whenever its *window-mean*
# gradient happens to be small — the single biggest source of gaps, even for
# a 1-day window.
function load_turb_diffusivity_nanmean(file, day_lo, day_hi)
    isfile(file) || return nothing
    try
        κc_fts = FieldTimeSeries(file, "κcbar")
        κu_fts = FieldTimeSeries(file, "κubar")
        zF     = znodes(κc_fts.grid, Face())

        κc_samples = [clean_diffusivity(Float64.(interior(κc_fts[i], 1, 1, :))[2:end-1])
                      for i in window_indices(κc_fts, day_lo, day_hi)]
        κu_samples = [clean_diffusivity(Float64.(interior(κu_fts[i], 1, 1, :))[2:end-1])
                      for i in window_indices(κu_fts, day_lo, day_hi)]

        return DiffProfiles(interior_faces(zF), nanmean_profiles(κc_samples), nanmean_profiles(κu_samples))
    catch err
        @warn "Skipping $file (missing κubar/κcbar — needs rerunning with the updated experiment script)" exception = err
        return nothing
    end
end

function load_les_diffusivity_nanmean(file, day_lo, day_hi)
    isfile(file) || return nothing
    try
        Tbar_fts = FieldTimeSeries(file, "Tbar")
        ubar_fts = FieldTimeSeries(file, "ubar")
        vbar_fts = FieldTimeSeries(file, "vbar")
        wT_fts   = FieldTimeSeries(file, "wT")
        uw_fts   = FieldTimeSeries(file, "uw")
        vw_fts   = FieldTimeSeries(file, "vw")

        zC = znodes(Tbar_fts.grid, Center())
        zF = znodes(wT_fts.grid, Face())

        κ_T_samples = Vector{Float64}[]
        κ_u_samples = Vector{Float64}[]
        for i in window_indices(Tbar_fts, day_lo, day_hi)
            Tbar = Float64.(interior(Tbar_fts[i], 1, 1, :))
            ubar = Float64.(interior(ubar_fts[i], 1, 1, :))
            vbar = Float64.(interior(vbar_fts[i], 1, 1, :))
            wT   = Float64.(interior(wT_fts[i], 1, 1, :))[2:end-1]
            uw   = Float64.(interior(uw_fts[i], 1, 1, :))[2:end-1]
            vw   = Float64.(interior(vw_fts[i], 1, 1, :))[2:end-1]

            dTdz = center_diff_interior(Tbar, zC)
            dudz = center_diff_interior(ubar, zC)
            dvdz = center_diff_interior(vbar, zC)

            push!(κ_T_samples, flux_gradient_ratio(wT, dTdz))
            S²    = dudz .^ 2 .+ dvdz .^ 2
            num_u = uw .* dudz .+ vw .* dvdz
            push!(κ_u_samples, flux_gradient_ratio(num_u, S²))
        end

        return DiffProfiles(interior_faces(zF), nanmean_profiles(κ_T_samples), nanmean_profiles(κ_u_samples))
    catch err
        @warn "Skipping $file" exception = err
        return nothing
    end
end

#####
##### Load everything for a given [day_lo, day_hi) window.
#####
function load_all_nanmean(day_lo, day_hi)
    variant_diff = [Dict{Int, Union{DiffProfiles, Nothing}}() for _ in VARIANTS]
    for (vi, v) in enumerate(VARIANTS), year in winter_years
        variant_diff[vi][year] = load_turb_diffusivity_nanmean(turb_file(year, v.uv_tag, v.eos_tag), day_lo, day_hi)
    end

    variant_diff_sm = [Dict{Int, Union{DiffProfiles, Nothing}}() for _ in VARIANTS]
    for (vi, v) in enumerate(VARIANTS), year in winter_years_sm
        variant_diff_sm[vi][year] = load_turb_diffusivity_nanmean(turb_file_sm(year, v.uv_tag, v.eos_tag), day_lo, day_hi)
    end

    les_diff = Dict{Int, Union{DiffProfiles, Nothing}}()
    for year in winter_years
        f = les_file(year)
        les_diff[year] = isnothing(f) ? nothing : load_les_diffusivity_nanmean(f, day_lo, day_hi)
    end

    les_diff_sm = Dict{Int, Union{DiffProfiles, Nothing}}()
    for year in winter_years_sm
        f = les_file_sm(year)
        les_diff_sm[year] = isnothing(f) ? nothing : load_les_diffusivity_nanmean(f, day_lo, day_hi)
    end

    return (; variant_diff, variant_diff_sm, les_diff, les_diff_sm)
end

const DAY30_DATA = load_all_nanmean(29.0, 30.0)
const DAY60_DATA = load_all_nanmean(59.0, 60.0)
const DAY30TO60_NANMEAN_DATA = load_all_nanmean(30.0, 60.0)

for (label, data) in (("day-30 snapshot", DAY30_DATA), ("day-60 snapshot", DAY60_DATA),
                       ("day 30-60 nanmean", DAY30TO60_NANMEAN_DATA))
    all(isnothing, vcat(values(data.les_diff)..., values(data.les_diff_sm)...)) &&
        @warn "[$label] No LES diffusivity profiles could be computed — check hourly_averaged_timeseries.jld2 paths"
    all(dp -> all(isnothing, values(dp)), data.variant_diff) &&
        error("[$label] No TURB κubar/κcbar found for FORCING_SOURCE=$(FORCING_SOURCE) — rerun the TURB_experiments PBS scripts first")
end

#####
##### Plotting: same 2 x 4 grid as the T-profile comparison figure.
##### Row 1: EM 2009/2010/2011/2014. Row 2: EM 2015, SM 2010, SM 2011, legend.
#####
show_yticklabels(col) = col <= 2
show_ylabel(col)      = col == 1

const EM_ROW1_YEARS = [2009, 2010, 2011, 2014]
const EM_ROW2_YEAR  = 2015
const NCOLS         = 4

const PANELS = vcat(
    [(mooring = :EM, year = y, row = 1, col = i) for (i, y) in enumerate(EM_ROW1_YEARS)],
    [(mooring = :EM, year = EM_ROW2_YEAR, row = 2, col = 1)],
    [(mooring = :SM, year = y, row = 2, col = i + 1) for (i, y) in enumerate(winter_years_sm)],
)
const LEGEND_ROW, LEGEND_COL = 2, NCOLS

function nice_log_limits(values)
    vals = filter(v -> isfinite(v) && v > 0, values)
    isempty(vals) && return (1e-7, 1e-1)
    lo = 10.0 ^ floor(log10(minimum(vals)))
    hi = 10.0 ^ ceil(log10(maximum(vals)))
    return (lo, hi)
end

function collect_all_values(data, field)
    vals = Float64[]
    for vd in (data.variant_diff, data.variant_diff_sm), d in vd, (_, dp) in d
        isnothing(dp) || append!(vals, get_profile(dp, field))
    end
    for d in (data.les_diff, data.les_diff_sm), (_, dp) in d
        isnothing(dp) || append!(vals, get_profile(dp, field))
    end
    return vals
end

function plot_diffusivity_comparison(filename, data, field::Symbol, xlabel_str, title_str)
    XLIMS = nice_log_limits(collect_all_values(data, field))

    fig = Figure(size = (195 * NCOLS + 40, 210 * 2 + 70),
                 fontsize = 11, figure_padding = (6, 10, 6, 4))

    em_axes = Axis[]
    sm_axes = Axis[]
    for p in PANELS
        row, col, year = p.row, p.col, p.year
        ax = CairoMakie.Axis(fig[row, col];
                 title              = p.mooring == :EM ? "EM Winter $year" : "SM Winter $year",
                 xlabel             = xlabel_str,
                 ylabel             = show_ylabel(col) ? L"z\;(\mathrm{m})" : "",
                 ylabelrotation     = π/2,
                 yticklabelsvisible = show_yticklabels(col),
                 xscale             = log10,
                 xgridvisible       = true,
                 ygridvisible       = true,
                 xticksize          = 4,
                 yticksize          = 4)
        push!(p.mooring == :EM ? em_axes : sm_axes, ax)

        les_dp = p.mooring == :EM ? get(data.les_diff, year, nothing) : get(data.les_diff_sm, year, nothing)
        Lz_here = 0.0
        if !isnothing(les_dp)
            lines!(ax, get_profile(les_dp, field), les_dp.z; color = LES_COLOR, linewidth = 2.5)
            Lz_here = max(Lz_here, abs(minimum(les_dp.z)))
        end

        vdata = p.mooring == :EM ? data.variant_diff : data.variant_diff_sm
        for (vi, v) in enumerate(VARIANTS)
            dp = get(vdata[vi], year, nothing)
            isnothing(dp) && continue
            lines!(ax, get_profile(dp, field), dp.z;
                   color = v.color, linewidth = 2.0, linestyle = v.linestyle)
            Lz_here = max(Lz_here, abs(minimum(dp.z)))
        end

        Lz_here = Lz_here > 0 ? Lz_here : 212.0
        ylims!(ax, (-Lz_here - 5, 5))
        xlims!(ax, XLIMS)
    end

    length(em_axes) > 1 && linkyaxes!(em_axes...)
    length(sm_axes) > 1 && linkyaxes!(sm_axes...)
    all_axes = vcat(em_axes, sm_axes)
    length(all_axes) > 1 && linkxaxes!(all_axes...)

    legend_elems  = Any[LineElement(color = LES_COLOR, linewidth = 2.5)]
    legend_labels = Any[LES_LABEL]
    for v in VARIANTS
        push!(legend_elems,  LineElement(color = v.color, linewidth = 2.0, linestyle = v.linestyle))
        push!(legend_labels, v.label)
    end
    Legend(fig[LEGEND_ROW, LEGEND_COL], legend_elems, legend_labels;
           labelsize = 11, framevisible = false, patchsize = (20, 12), tellwidth = false)

    Label(fig[0, 1:NCOLS],
          "$(title_str) (GLEN-forced, $(FORCING_SOURCE))";
          fontsize = 11, font = :bold, justification = :center)

    for col in 1:NCOLS
        colsize!(fig.layout, col, Relative(1 / NCOLS))
    end
    colgap!(fig.layout, 10)
    rowgap!(fig.layout, 6)

    outfile = joinpath(FIGURE_DIR, filename)
    save(outfile, fig)
    @info "Saved → $outfile"
end

plot_diffusivity_comparison(
    "LES_TURB_eastern_southern_mooring_comparison_heat_diffusivity_profiles_day30_$(FORCING_SOURCE).pdf",
    DAY30_DATA, :κ_T, L"\kappa_T \;(\mathrm{m^2\,s^{-1}})",
    "Eastern & Southern Mooring — LES vs k-ε heat diffusivity (day 30)")

plot_diffusivity_comparison(
    "LES_TURB_eastern_southern_mooring_comparison_momentum_diffusivity_profiles_day30_$(FORCING_SOURCE).pdf",
    DAY30_DATA, :κ_u, L"\kappa_u \;(\mathrm{m^2\,s^{-1}})",
    "Eastern & Southern Mooring — LES vs k-ε momentum diffusivity (day 30)")

plot_diffusivity_comparison(
    "LES_TURB_eastern_southern_mooring_comparison_heat_diffusivity_profiles_day60_$(FORCING_SOURCE).pdf",
    DAY60_DATA, :κ_T, L"\kappa_T \;(\mathrm{m^2\,s^{-1}})",
    "Eastern & Southern Mooring — LES vs k-ε heat diffusivity (day 60)")

plot_diffusivity_comparison(
    "LES_TURB_eastern_southern_mooring_comparison_momentum_diffusivity_profiles_day60_$(FORCING_SOURCE).pdf",
    DAY60_DATA, :κ_u, L"\kappa_u \;(\mathrm{m^2\,s^{-1}})",
    "Eastern & Southern Mooring — LES vs k-ε momentum diffusivity (day 60)")

plot_diffusivity_comparison(
    "LES_TURB_eastern_southern_mooring_comparison_heat_diffusivity_profiles_day30to60mean_$(FORCING_SOURCE).pdf",
    DAY30TO60_NANMEAN_DATA, :κ_T, L"\kappa_T \;(\mathrm{m^2\,s^{-1}})",
    "Eastern & Southern Mooring — LES vs k-ε heat diffusivity (day 30-60 mean)")

plot_diffusivity_comparison(
    "LES_TURB_eastern_southern_mooring_comparison_momentum_diffusivity_profiles_day30to60mean_$(FORCING_SOURCE).pdf",
    DAY30TO60_NANMEAN_DATA, :κ_u, L"\kappa_u \;(\mathrm{m^2\,s^{-1}})",
    "Eastern & Southern Mooring — LES vs k-ε momentum diffusivity (day 30-60 mean)")
