"""
    pos = subplot_hor_vertpos(numph, numpv, hors, hore, vers, vere, Dsh, Dsv)

Direct Julia port of the MATLAB utility `subplot_hor_vertpos.m`
(https://github.com/maartenbuijsman/matlab-funcs/blob/master/plotting/subplot_hor_vertpos.m):
computes normalized `(left, bottom, width, height)` positions, each a
fraction of the full figure in [0,1], for a `numph x numpv` grid of panels,
filling left-to-right then top-to-bottom (same order as MATLAB's `subplot`).

# Arguments
- `numph, numpv`: number of panels across / down
- `hors, hore`: left / right margin (fraction of figure width)
- `vers, vere`: BOTTOM / TOP margin (fraction of figure height) -- NOT
  top/bottom as the names suggest! Verified directly (symbolically and
  numerically): because the loop places row j=1 at the top of the page
  while `pos` is built in bottom-up MATLAB-position coordinates, `vere`
  ends up controlling the gap ABOVE the first (top) row and `vers` the gap
  BELOW the last (bottom) row -- i.e. exactly swapped from the natural
  reading of "start"/"end". This matches the MATLAB original exactly, just
  documenting it here since it's an easy trap when porting/reusing.
- `Dsh, Dsv`: horizontal / vertical gap between adjacent panels (fraction)

# Returns
- `pos::Vector{NTuple{4,Float64}}`: one `(left, bottom, width, height)` per
  panel, length `numph*numpv`, left-to-right/top-to-bottom order.

# Usage with Makie
`pos` is normalized exactly like MATLAB's `subplot('position', pos)`. To
place a Makie axis at that exact spot (bypassing GridLayout's own auto
sizing, e.g. to sidestep the Relative/Auto rowsize fights and axis-
protrusion/gap surprises documented in `claudecodes/IW_nondim_vs_lat.jl`),
convert to a `BBox` in points via `subplot_bbox` below, then:
    `Axis(fig.scene, bbox=subplot_bbox(pos[i], fig_width_pt, fig_height_pt))`

IMPORTANT: an `Axis` built this way (parented to `fig.scene`, not to the
Figure's GridLayout) does NOT auto-update its limits from plotted data --
confirmed directly (`ax.finallimits[]` stays the Makie default
`(0,10)x(0,10)` even after `lines!`/`scatter!` calls). Call `autolimits!(ax)`
yourself after all plotting into that axis is done.

# Info
Maarten Buijsman, USM DMS, 2026-9-9 (ported from MATLAB plotting/subplot_hor_vertpos.m)
"""
function subplot_hor_vertpos(numph::Int, numpv::Int, hors::Real, hore::Real,
                              vers::Real, vere::Real, Dsh::Real, Dsv::Real)
    Dh = (1 - (numph-1)*Dsh - hors - hore) / numph
    Dv = (1 - (numpv-1)*Dsv - vers - vere) / numpv
    pos = Vector{NTuple{4,Float64}}()
    for j in 1:numpv, i in 1:numph
        left   = hors + (i-1)*(Dh+Dsh)
        bottom = 1 - (vere + Dv + (Dv+Dsv)*(j-1))
        push!(pos, (left, bottom, Dh, Dv))
    end
    return pos
end

"""
    bb = subplot_bbox(pos_i, fig_width, fig_height)

Convert one normalized `(left, bottom, width, height)` tuple from
`subplot_hor_vertpos` into absolute `(left, right, bottom, top)` values in
the same units as `fig_width`/`fig_height` (e.g. points, for a Makie
`Figure(size=...)`). Returns a plain tuple -- wrap in `BBox(bb...)` at the
call site so this file has no CairoMakie/Makie dependency of its own.

# Info
Maarten Buijsman, USM DMS, 2026-9-4
"""
function subplot_bbox(pos_i::NTuple{4,<:Real}, fig_width::Real, fig_height::Real)
    left, bottom, w, h = pos_i
    l = left*fig_width
    b = bottom*fig_height
    return (l, l + w*fig_width, b, b + h*fig_height)
end

"""
    bb = colorbar_bbox(pos_i, fig_width, fig_height, figbarspace, wbar, hbar, vertoffbar)

Direct Julia port of the MATLAB utility `colorbar_pos.m`
(https://github.com/maartenbuijsman/matlab-funcs/blob/master/plotting/colorbar_pos.m):
places a colorbar immediately to the right of a `subplot_hor_vertpos` panel,
without touching that panel's own position. Takes the same normalized
`pos_i` tuple used by `subplot_bbox` (so it stays correct even if the panel
layout changes) and returns an absolute `(left, right, bottom, top)` tuple
in the same units as `fig_width`/`fig_height` -- wrap in `BBox(bb...)` at
the call site, same convention as `subplot_bbox`.

# Arguments
- `pos_i`: this panel's normalized `(left, bottom, width, height)`, from `subplot_hor_vertpos`
- `fig_width, fig_height`: figure size, e.g. in points for a Makie `Figure(size=...)`
- `figbarspace`: gap between the panel's right edge and the colorbar (fraction of figure width)
- `wbar`: colorbar width (fraction of figure width)
- `hbar`: colorbar height (fraction of the PANEL's height, e.g. 1.0 = full height)
- `vertoffbar`: vertical offset of the colorbar's bottom edge above the panel's bottom edge (fraction of panel height)

# Info
Maarten Buijsman, USM DMS, 2026-9-9 (ported from MATLAB plotting/colorbar_pos.m)
"""
function colorbar_bbox(pos_i::NTuple{4,<:Real}, fig_width::Real, fig_height::Real,
                        figbarspace::Real, wbar::Real, hbar::Real, vertoffbar::Real)
    left, bottom, w, h = pos_i
    l = (left + w + figbarspace) * fig_width
    b = (bottom + vertoffbar*h) * fig_height
    return (l, l + wbar*fig_width, b, b + hbar*h*fig_height)
end
