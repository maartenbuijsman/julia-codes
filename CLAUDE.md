# Project conventions

- All figures (PNG/PDF/etc.) that Claude generates must be saved to `/home/mbui/ModelOutput/figs/`,
  regardless of where the source data or script lives.
- All `.jl` scripts that Claude creates must be stored in `/home/mbui/Documents/julia-codes/claudecodes/`.
- (Response numbering rule moved to global `~/.claude/CLAUDE.md` + `~/.claude/hooks/check_response_number.py`
  -- applies to all projects now, not just this one.)
- Whenever Claude edits an existing `.jl` file, update the date in that file's header/preamble
  comment (e.g. "MCB, USM, 2026-8-4" or "Maarten Buijsman, USM DMS, 2026-7-28") to the date of the
  edit, keeping the author name unchanged.
