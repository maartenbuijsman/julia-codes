#= savefig_dpi.jl
Maarten Buijsman, USM DMS, 2026-9-16

Save a Makie figure as a PNG that is correctly TAGGED with its print
resolution, not just correctly sized in pixels.

Background: Makie's drawing unit is the typographic point (72/inch), so a
Figure(size=(18cm, ...)) saved with px_per_unit = dpi/72 has the right PIXEL
COUNT for `dpi` at that physical width. CairoMakie, however, writes the PNG's
pHYs (physical pixel size) chunk as 96*px_per_unit, using the 96/inch CSS
reference instead. With px_per_unit = 300/72 the file therefore reports 400
dpi, and any layout software that honours the tag places the figure at 3/4 of
its intended width (an 18 cm panel lands at 13.5 cm).

savefig300(path, fig) saves at a true 300 dpi and rewrites the tag to match.
set_png_dpi!(path, dpi) rewrites the tag on an existing PNG (with a correct
CRC32, so the file stays valid).

Usage -- replaces a save call of the form
    save(fname, fig; px_per_unit = 300/72)
with
    savefig300(fname, fig)
=#

# CRC-32 (IEEE 802.3, reflected polynomial 0xEDB88320) -- PNG chunk checksum
const _PNG_CRCTAB = let t = zeros(UInt32, 256)
    for n in 0:255
        c = UInt32(n)
        for _ in 1:8
            c = (c & 0x00000001) != 0 ? (0xEDB88320 ⊻ (c >> 1)) : (c >> 1)
        end
        t[n+1] = c
    end
    t
end

function png_crc32(data)
    c = 0xFFFFFFFF
    for b in data
        c = _PNG_CRCTAB[((c ⊻ UInt32(b)) & 0xFF) + 1] ⊻ (c >> 8)
    end
    return c ⊻ 0xFFFFFFFF
end

_be32(v) = UInt8[(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]

"""
    set_png_dpi!(path, dpi) -> Bool

Rewrite the pHYs chunk of the PNG at `path` so it reports `dpi` in both
directions. Returns false (with a warning) if the file has no pHYs chunk.
Only the tag changes -- the raster is untouched.
"""
function set_png_dpi!(path::AbstractString, dpi::Real)
    b   = read(path)
    tag = UInt8['p','H','Y','s']
    i   = findfirst(k -> b[k:k+3] == tag, 1:length(b)-12)
    if i === nothing
        @warn "no pHYs chunk found in $path; DPI tag left unchanged"
        return false
    end
    ppm  = round(UInt32, dpi/0.0254)                  # pixels per metre
    data = vcat(_be32(ppm), _be32(ppm), UInt8[0x01])  # unit specifier 1 = metre
    b[i+4:i+12]  = data
    b[i+13:i+16] = _be32(png_crc32(vcat(tag, data)))
    write(path, b)
    return true
end

"""
    savefig300(path, fig; dpi=300) -> path

Save `fig` as a PNG rasterized at `dpi` (via px_per_unit = dpi/72, matching
Makie's 72/inch drawing unit) and tag the file with that same `dpi`.
For non-PNG targets the tagging step is skipped.
"""
function savefig300(path::AbstractString, fig; dpi::Real=300)
    save(path, fig; px_per_unit=dpi/72)
    lowercase(splitext(path)[2]) == ".png" && set_png_dpi!(path, dpi)
    return path
end
