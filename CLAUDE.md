# Project conventions

- All figures (PNG/PDF/etc.) that Claude generates must be saved to `/home/mbui/ModelOutput/figs/`,
  regardless of where the source data or script lives.
- All `.jl` scripts that Claude creates must be stored in `/home/mbui/Documents/julia-codes/claudecodes/`.
- Every response to a user request must start with a response number (e.g. "**#1**"), incrementing
  by one each time, so the user can refer back to previous answers by number. Numbering restarts
  at 1 at the start of each new conversation/session. The number must be the first thing in
  Claude's actual chat message text -- putting it only in a tool parameter (e.g. a SendUserFile
  `caption`) does NOT count, since that's not part of the visible response text.
  Enforced automatically by a Stop hook (`.claude/hooks/check_response_number.py`, registered in
  `.claude/settings.json`) that blocks the turn from ending if the number is missing or wrong.
- Whenever Claude edits an existing `.jl` file, update the date in that file's header/preamble
  comment (e.g. "MCB, USM, 2026-8-4" or "Maarten Buijsman, USM DMS, 2026-7-28") to the date of the
  edit, keeping the author name unchanged.
