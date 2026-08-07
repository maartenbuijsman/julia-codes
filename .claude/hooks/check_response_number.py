#!/usr/bin/env python3
"""
Stop hook: enforce the CLAUDE.md rule that every assistant response to the
user must start with an incrementing tracking number, e.g. "**#1**", "**#2**",
restarting at 1 each new session.

Reads the Stop-hook stdin JSON (expects a "transcript_path" field), finds the
LAST assistant text block in that session's transcript (the just-completed
turn's final reply), and checks it starts with **#N** where N is exactly one
more than the highest **#N** already used earlier in the same transcript. If
missing or wrong, blocks the stop so Claude fixes it before finishing.
"""
import json
import re
import sys

NUM_RE = re.compile(r"^\s*\*\*#(\d+)\*\*")

def last_assistant_text_blocks(transcript_path):
    """Yield (text) for every assistant text block, in file order."""
    try:
        with open(transcript_path, "r") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") != "assistant":
            continue
        content = obj.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.strip():
                    yield text

def main():
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)  # can't parse input, don't block

    transcript_path = hook_input.get("transcript_path")
    if not transcript_path:
        sys.exit(0)  # no transcript to check, don't block

    texts = list(last_assistant_text_blocks(transcript_path))
    if not texts:
        sys.exit(0)  # nothing to check yet

    last_text = texts[-1]

    # highest number used in any PRIOR text block (excludes the last one itself)
    prior_max = 0
    for t in texts[:-1]:
        m = NUM_RE.match(t)
        if m:
            prior_max = max(prior_max, int(m.group(1)))
    expected = prior_max + 1

    m = NUM_RE.match(last_text)
    if m and int(m.group(1)) == expected:
        sys.exit(0)  # correct, allow stop

    if m:
        reason = (f"Your response used **#{m.group(1)}** but the expected next "
                   f"tracking number is **#{expected}** (per CLAUDE.md). "
                   f"Please correct the number.")
    else:
        reason = (f"Your response is missing the required tracking number. "
                   f"Per CLAUDE.md, it must start with **#{expected}**. "
                   f"Please add it.")

    print(json.dumps({"decision": "block", "reason": reason}))
    sys.exit(0)

if __name__ == "__main__":
    main()
