#####
##### Lake Superior bathymetry map with mooring and flux-station locations
#####
# Bathymetry: data/superior_lld.grd.gz (GMT NetCDF, depth relative to IGLD85 − 183.2 m)
# Negative z → below lake surface; NaN → land outside the lake.
#
# Stations plotted:
#   Stannard Rock  — GLEN half-hourly flux / met station (lat=47.184, lon=−87.225)
#   Eastern Mooring (EM) — Austin2023 thermistor chain  (lat=47.537, lon=−86.573)
#   Southern Mooring (SM)— Austin2023 thermistor chain  (lat=47.033, lon=−86.667)
#####

using NCDatasets
using CairoMakie
using ColorSchemes

# ── Decompress bathymetry to a temp file ──────────────────────────────────────
const BATHY_GZ = joinpath(@__DIR__, "..", "data", "superior_lld.grd.gz")
const TMP_NC   = tempname() * ".nc"
run(pipeline(`gunzip -c $BATHY_GZ`, TMP_NC))

lon, lat, depth = NCDatasets.Dataset(TMP_NC) do ds
    Float64.(ds["x"][:]),
    Float64.(ds["y"][:]),
    Float64.(coalesce.(ds["z"][:, :], NaN))   # (Nlon, Nlat); Missing → NaN
end
rm(TMP_NC)

# Depth convention: negative = below surface, positive / NaN = land
# Mask land (z > 0 or NaN → treat as NaN so we can set a land colour)
depth_lake = copy(depth)
depth_lake[depth_lake .>= 0.0] .= NaN   # land/above-surface → NaN

# ── Station coordinates ────────────────────────────────────────────────────────
stations = [
    (lon = -87.225,      lat = 47.18361, label = "Stannard Rock",   marker = :star5,  color = :darkorange),
    (lon = -86.573333,   lat = 47.53667, label = "Eastern Mooring", marker = :circle, color = :gold),
    (lon = -86.666667,   lat = 47.03333, label = "Southern Mooring",marker = :circle, color = :plum),
]

# ── Figure ─────────────────────────────────────────────────────────────────────
FIGURE_DIR = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURE_DIR)

with_theme(theme_latexfonts()) do
    fig = Figure(size = (700, 300), fontsize = 11, figure_padding = (6, 10, 6, 6))

    ax = Axis(fig[1, 1];
              xlabel      = "Longitude (°W)",
              ylabel      = "Latitude (°N)",
              aspect      = DataAspect(),
              xticks      = -92:2:-84,
              yticks      = 46:0.5:50,
              xticklabelsize = 9,
              yticklabelsize = 9)

    # --- bathymetry heatmap (depth ≤ 0) ----------------------------------------
    # depth_lake is (Nlon × Nlat), heatmap expects (x, y, z)
    hm = heatmap!(ax, lon, lat, depth_lake;
                  colormap   = :deep,
                  colorrange = (-400, 0),
                  nan_color  = (:sandybrown, 0.55),
                  rasterize  = 4)   # scale factor 4× base res ≈ 288 dpi; axes/contours/labels stay vector

    # --- depth contour lines at key intervals ----------------------------------
    contour_levels = [-400, -300, -200, -100, -50]
    contour!(ax, lon, lat, depth_lake;
             levels    = contour_levels,
             color     = (:white, 0.35),
             linewidth = 0.6)

    # --- lake outline (0 m isobath) at full opacity ----------------------------
    contour!(ax, lon, lat, depth;
             levels    = [0.0],
             color     = :black,
             linewidth = 0.9)

    # --- station markers -------------------------------------------------------
    for s in stations
        scatter!(ax, [s.lon], [s.lat];
                 marker    = s.marker,
                 color     = s.color,
                 markersize = 14,
                 strokewidth = 0.8,
                 strokecolor = :black)
    end

    # --- colorbar (short, centred on axis) ------------------------------------
    Colorbar(fig[1, 2], hm;
             label         = "Depth (m)",
             width          = 14,
             height         = Relative(0.4),
             valign         = :center,
             ticksize       = 4,
             labelsize      = 10,
             ticklabelsize  = 9,
             flipaxis       = false)

    # --- legend inside the axis (top-left, over land) -------------------------
    legend_elems = [MarkerElement(marker = s.marker, color = s.color,
                                  markersize = 10, strokewidth = 0.8,
                                  strokecolor = :black) for s in stations]
    legend_labels = [s.label for s in stations]
    axislegend(ax, legend_elems, legend_labels;
               position        = :lt,
               labelsize       = 9,
               framevisible    = true,
               padding         = (6, 6, 4, 4),
               backgroundcolor = (:white, 0.8))

    colgap!(fig.layout, 6)
    Makie.update_state_before_display!(fig)
    resize_to_layout!(fig)

    outfile = joinpath(FIGURE_DIR, "lake_superior_map.pdf")
    save(outfile, fig)
    @info "Saved → $outfile"
end
