"""
    text_fignum!(ax, str; offhorz=-0.05, offvert=-0.05, horloc=:right, vertloc=:top,
                 fs=10, bckclr=nothing, boxw=0.096, boxh=0.11, boxpad=0.015)

Direct Julia port of the MATLAB utility `text_fignum.m`
(https://github.com/maartenbuijsman/matlab-funcs/blob/master/plotting/text_fignum.m):
prints a bold panel-number label (e.g. "(a)") anchored to one corner of an
axis, positioned relative to that axis's OWN data range rather than a fixed
pixel offset -- so the label sits at a consistent visual corner regardless
of what's actually plotted.

# Arguments
- `ax`: the target `Axis` (its current `finallimits` are used, so call this
  AFTER `autolimits!`/`xlims!`/`ylims!` have been applied to `ax`)
- `str`: label text, e.g. `"(a)"`
- `offhorz, offvert`: offset from the chosen corner, as a fraction of the
  axis's x/y data range (default `offhorz=-0.05`, `offvert=-0.05`, i.e.
  shifted slightly in from both the right edge and the top; MATLAB's own
  default was `offhorz=-0.1`, `offvert=0.0`)
- `horloc`: `:left` or `:right`
- `vertloc`: `:top` or `:bottom`
- `fs`: fontsize
- `bckclr`: background color behind the text, or `nothing` for none (MATLAB's
  `'t'`). Makie's `Text` has no true background-box attribute (a `glowcolor`
  halo was tried first but is a no-op on the CairoMakie backend), so this
  draws an actual filled `poly!` rectangle behind the glyph instead -- sized
  as a fraction of THIS axis's own data range (`boxw`/`boxh`), same
  fraction-of-range convention as `offhorz`/`offvert`, so it scales with the
  panel regardless of its own x/y data range.
- `boxw, boxh`: background box size, as a fraction of the axis's x/y data
  range (only used when `bckclr !== nothing`); sized generously for a short
  "(a)"-style label -- tune if the label text is longer
- `boxpad`: extra padding around the label within the box, same fraction convention

# Info
Maarten Buijsman, USM DMS, 2026-9-9 (ported from MATLAB plotting/text_fignum.m,
originally MCB, UCLA, 2009-05-01)
"""
function text_fignum!(ax, str; offhorz::Real=-0.05, offvert::Real=-0.05,
                       horloc::Symbol=:right, vertloc::Symbol=:top,
                       fs::Real=10, bckclr=nothing,
                       boxw::Real=0.096, boxh::Real=0.11, boxpad::Real=0.015)
    lims = ax.finallimits[]
    xmin, ymin = lims.origin
    dx, dy = lims.widths

    offh = offhorz*dx
    offv = offvert*dy

    x  = horloc == :right ? xmin+dx+offh : xmin+offh
    y  = vertloc == :top   ? ymin+dy+offv : ymin+offv
    ha = horloc == :right ? :right : :left
    va = vertloc == :top   ? :top   : :bottom

    if bckclr !== nothing
        padx = boxpad*dx; pady = boxpad*dy
        bw = boxw*dx;      bh = boxh*dy
        bx0 = horloc == :right ? x-bw-padx : x-padx
        bx1 = horloc == :right ? x+padx    : x+bw+padx
        by0 = vertloc == :top   ? y-bh-pady : y-pady
        by1 = vertloc == :top   ? y+pady    : y+bh+pady
        poly!(ax, Rect2f(bx0, by0, bx1-bx0, by1-by0), color=bckclr, strokewidth=0)
    end

    text!(ax, x, y; text=str, align=(ha, va), font=:bold, fontsize=fs)
end
