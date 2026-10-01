import argparse
import json
import os
import shlex
import sys
from urllib.parse import unquote, urlparse

import requests

# =============================================================
# Configuration
# =============================================================
LM_STUDIO_BASE   = "http://localhost:1234"
CHAT_ENDPOINT    = f"{LM_STUDIO_BASE}/v1/chat/completions"
MODELS_ENDPOINT  = f"{LM_STUDIO_BASE}/api/v0/models"

MIN_CONTEXT_LENGTH = 20_000 # sanity floor for the loaded model's context length — comfortably
                             # covers a WINDOW_LINES-sized prompt plus instructions/response
WINDOW_LINES     = 600      # lines per LLM call for start/scan/resume detection. Empirically: a
                             # window with only one real metadata boundary in view is found
                             # correctly; a window with several genuine boundaries tends to pick a
                             # later one instead of the nearest one. 600 lines keeps the odds of
                             # more than one real boundary landing in a single window low without
                             # calling the LLM too often.
MAX_RESUME_WINDOWS = 5       # safety cap on how many WINDOW_LINES chunks to search forward through
                             # when looking for where the story resumes after a metadata block,
                             # before giving up and routing the file to review/ instead of risking
                             # either an infinite scan or a runaway deletion.
RESPONSE_MAX_TOKENS = 500    # generous for a single small JSON object; this model reasons even with
                             # chat_template_kwargs enable_thinking=False (known LM Studio/Gemma-4
                             # bug: https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2046),
                             # and content comes back empty if reasoning exhausts the budget first.

REVIEW_MAX_REMOVED_RATIO = 0.9  # flag to review/ instead of cleaned/ if more than this fraction is cut

# --- Iterative re-scan for residual metadata --------------------------------
# The first pass is windowed and reports only one metadata boundary per window, so a heavily-cut
# file tends to leave a few residual byline/title lines the scan walked past — the ones not short
# enough or not sandwiched between two cuts for verify_sandwiched_gaps to catch. Re-scanning the
# already-cleaned text picks them up because they now sit in clean, correctly-numbered context.
# This is deliberately gated and bounded: a blanket rescan of every cleaned file was tried before
# and over-deleted legitimate prose (see the verify_sandwiched_gaps note), so only files that were
# cut heavily get a second look, only a few passes run, and any pass that wants to remove more
# than a small margin routes the file to review/ instead of being trusted.
RESCAN_TRIGGER_RATIO  = 0.05  # re-scan if the first pass removed at least this fraction of lines...
RESCAN_TRIGGER_CUTS   = 5     # ...or produced at least this many separate cut ranges
RESCAN_MAX_PASSES     = 3     # hard cap on total scan_pass runs (1 initial + up to 2 re-scans)
RESCAN_SAFE_RATIO     = 0.03  # a re-scan removing more than this fraction of ITS OWN input is
                              # treated as untrustworthy and routed to review/ ...
RESCAN_SAFE_MIN_LINES = 5     # ...but always tolerate removing at least this many lines, so a
                              # small trailing-byline cleanup on a short file doesn't trip the guard

# --- Schemas -----------------------------------------------------------------

# Used for both "find the start of the story" and "find where the story resumes after metadata" —
# structurally the same question: is there a line of story prose in this excerpt, and if so, which
# is the first one?
FOUND_LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "line": {"type": "integer"},
    },
    "required": ["found", "line"],
}

# Metadata isn't always URL-tagged — related-story title suggestions and trailing author/genre/
# stat bylines are consistently missed otherwise, since structurally they look like short lines of
# plain text with nothing else to flag them. This names the general shape (label vs. sentence)
# rather than any one site's specific wording, so it should transfer across archive sources.
LABEL_VS_PROSE_CLAUSE = (
    "Metadata is not always marked with a URL. Also treat as metadata even without a link: story "
    "or chapter titles (including suggestions for OTHER stories), author names, and short "
    "attribution/stat lines such as 'Author Name | Category | N reads' — a name, category, and "
    "engagement count strung together. These read as labels, not narrative sentences. Genuine "
    "story prose — including short lines of dialogue — always reads as part of a sentence "
    "describing action, speech, or description; a bare title, name, or statistic never does."
)

FIND_STORY_INSTRUCTIONS = (
    "You are looking for where narrative story prose begins in an excerpt of text scraped from an "
    "old internet archive (forum, Usenet, or fan-fiction site), which may be mixed with site "
    "metadata: navigation menus, ads, headers/footers, login prompts, disclaimers, comment "
    "sections, forum boilerplate, and similar junk. Each line below is numbered.\n\n"
    f"{LABEL_VS_PROSE_CLAUSE}\n\n"
    "Does actual narrative story prose (dialogue, description, action) appear anywhere in this "
    "excerpt? If yes, set found to true and line to the line number of the FIRST such line. If the "
    "excerpt is entirely site metadata with no story prose at all, set found to false and line to 0."
)

# Gated on a boolean first — an earlier ungated version (a bare "line" field, 0 meaning "not found")
# was unreliable: it would fabricate a line number at the edge of the window even when no metadata
# was present, rather than reporting "not found." Forcing an explicit yes/no decision first fixed
# that, and also made it prefer the NEAREST boundary over a later one when more than one is visible.
BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata_found": {"type": "boolean"},
        "line": {"type": "integer"},
    },
    "required": ["metadata_found", "line"],
}

SCAN_INSTRUCTIONS = (
    "You are scanning a story text file for where site metadata (navigation, ads, headers, "
    "chapter-transition boilerplate) interrupts the story. Lines are numbered, and everything "
    "shown is presumed to be story prose until metadata begins.\n\n"
    f"{LABEL_VS_PROSE_CLAUSE}\n\n"
    "First decide: does metadata actually begin somewhere within this excerpt? Set metadata_found "
    "to true only if you can point to a specific line where non-story content starts. If the "
    "excerpt is entirely story prose with no metadata, set metadata_found to false and set line to "
    "0.\n\n"
    "If metadata_found is true, set line to the line number of the LAST line of story prose "
    "immediately before the FIRST point where metadata begins."
)

# --- Sandwiched-gap verification ---------------------------------------------
# A targeted second look, not a second full pass: only re-examines short runs of "kept" text that
# sit directly between two of the scan's own cuts (or between the final cut and EOF) — the specific
# shape of residual junk observed in practice (e.g. a single leftover related-story title stranded
# between two correctly-identified metadata blocks). A full rescan of the whole cleaned file was
# tried and rejected: it also wandered into re-judging legitimate content nowhere near a cut and
# wrongly deleted some of it. Scoping to only cut-adjacent gaps avoids that entirely.
SANDWICH_MAX_LINES   = 5   # only verify a gap this short or shorter — real story content between
                            # real metadata blocks runs to dozens or hundreds of lines, so this stays
                            # well clear of legitimate prose
SANDWICH_CONTEXT_LINES = 3 # lines of confirmed-clean story shown on each side of the snippet, for contrast

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"is_story": {"type": "boolean"}},
    "required": ["is_story"],
}

VERIFY_INSTRUCTIONS = (
    "You are reviewing a short SNIPPET of text that survived an automated cleanup pass on a story "
    "file scraped from an old internet archive. Confirmed, clean story prose is shown immediately "
    "before and after it — everything else around it has already been removed as site metadata "
    "(navigation, ads, headers, bylines, related-story links, etc.).\n\n"
    f"{LABEL_VS_PROSE_CLAUSE}\n\n"
    "Note: a story or chapter's own structural markers — such as 'THE END', 'End of Chapter N', or "
    "'To be continued' — are part of the story, not metadata, even though they are short and don't "
    "read as a narrative sentence.\n\n"
    "Does the SNIPPET belong in the story (continues the narrative from before into after), or is "
    "it leftover site metadata that should be removed? Set is_story accordingly."
)


def verify_sandwiched_gaps(model_id, lines, cuts, debug=False):
    """
    Re-examine short kept gaps that sit directly between two cuts (or between the final cut and
    EOF). Returns a list of additional 1-indexed inclusive cut ranges to fold in.
    """
    merged = merge_ranges(cuts)
    total_lines = len(lines)
    extra_cuts = []

    candidates = []  # (gap_start, gap_end, preceding_cut_start) — all 1-indexed
    for i in range(len(merged) - 1):
        gap_start = merged[i][1] + 1
        gap_end = merged[i + 1][0] - 1
        if gap_start <= gap_end <= gap_start + SANDWICH_MAX_LINES - 1:
            candidates.append((gap_start, gap_end, merged[i][0]))
    if merged:
        gap_start = merged[-1][1] + 1
        gap_end = total_lines
        if gap_start <= gap_end <= gap_start + SANDWICH_MAX_LINES - 1:
            candidates.append((gap_start, gap_end, merged[-1][0]))

    for gap_start, gap_end, preceding_cut_start in candidates:
        context_before_end = preceding_cut_start - 1
        context_before_start = max(1, context_before_end - SANDWICH_CONTEXT_LINES + 1)
        context_before = lines[context_before_start - 1:context_before_end]

        after_cut = next((c for c in merged if c[0] > gap_end), None)
        context_after_start = after_cut[1] + 1 if after_cut else gap_end + 1
        context_after = lines[context_after_start - 1: context_after_start - 1 + SANDWICH_CONTEXT_LINES]

        parts = ["=== STORY (before) ==="]
        parts.extend(context_before)
        parts.append("=== SNIPPET ===")
        parts.extend(lines[gap_start - 1:gap_end])
        parts.append("=== STORY (after) ===")
        parts.extend(context_after)
        text = "\n".join(parts)

        result = _post_chat(model_id, VERIFY_INSTRUCTIONS, text, "verify", VERIFY_SCHEMA, debug,
                             log_label=f"for gap {gap_start}-{gap_end}")
        if result is None:
            continue  # leave ambiguous/failed verification as-is rather than risk a false cut
        if not result.get("is_story", True):
            extra_cuts.append((gap_start, gap_end))
            if debug:
                print(f"    [verify {gap_start}-{gap_end}] confirmed metadata, cutting")
        elif debug:
            print(f"    [verify {gap_start}-{gap_end}] confirmed story, keeping")

    return extra_cuts


# =============================================================
# LM Studio plumbing
# =============================================================

def get_model_info(debug=False):
    """Confirm exactly one chat-capable model is loaded and its context length covers MIN_CONTEXT_LENGTH.
    Embedding models are excluded — LM Studio can have one loaded alongside a chat model (e.g. for
    other tools), and grabbing "whichever is loaded" without filtering picks the wrong one silently."""
    resp = requests.get(MODELS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    loaded = [m for m in models if m.get("state") == "loaded"]

    if debug:
        for m in loaded:
            print(f"[model] {m['id']} | type={m.get('type')} | "
                  f"loaded_context_length={m.get('loaded_context_length')}")

    chat_models = [m for m in loaded if m.get("type") != "embeddings"]
    if not chat_models:
        raise RuntimeError(
            "No chat-capable model is loaded in LM Studio (only an embedding model, or nothing, "
            "is loaded). Load a chat model in LM Studio and try again."
        )
    if len(chat_models) > 1:
        names = ", ".join(m["id"] for m in chat_models)
        raise RuntimeError(
            f"Multiple chat-capable models are loaded ({names}) — unload all but the one you want "
            f"cleanup.py to use, so there's no ambiguity about which one gets called."
        )

    model = chat_models[0]
    model_id = model["id"]
    context_length = model.get("loaded_context_length")

    if context_length is None:
        print(f"WARNING: could not read loaded_context_length for {model_id}; "
              f"assuming it covers the {MIN_CONTEXT_LENGTH}-token floor.")
    elif context_length < MIN_CONTEXT_LENGTH:
        raise RuntimeError(
            f"Model '{model_id}' is loaded with context length {context_length}, "
            f"which is smaller than the configured floor ({MIN_CONTEXT_LENGTH}). "
            f"Increase the context length in LM Studio or lower MIN_CONTEXT_LENGTH."
        )
    return model_id


def _post_chat(model_id, system_prompt, user_text, schema_name, schema, debug=False, log_label=""):
    """Lowest-level plumbing: POST one chat completion with a JSON-schema-constrained response.
    Returns the parsed dict, or None on any failure (empty content / parse error)."""
    tag = f" {log_label}" if log_label else ""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": RESPONSE_MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    content = message["content"]
    if not content:
        finish_reason = data["choices"][0].get("finish_reason")
        reasoning_chars = len(message.get("reasoning_content") or "")
        print(f"  WARNING: empty content{tag} "
              f"(finish_reason={finish_reason}, reasoning_chars={reasoning_chars})")
        return None
    if debug:
        print(f"    [llm raw{tag}] {content.strip()!r}")
    try:
        return json.loads(content)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  WARNING: failed to parse LLM response{tag}: {e}")
        return None


def _call_llm(model_id, system_prompt, window_start, window_end, lines, schema_name, schema, debug=False):
    """Shared plumbing for a single numbered-lines call with a JSON-schema-constrained response.
    Returns the parsed dict, or None on any failure (empty content / parse error)."""
    parts = [f"{i + 1}: {lines[i]}" for i in range(window_start, window_end)]
    text = "\n".join(parts)
    return _post_chat(model_id, system_prompt, text, schema_name, schema, debug,
                       log_label=f"for lines {window_start + 1}-{window_end}")


def find_story_line(model_id, lines, window_start, window_end, debug=False):
    """Ask whether story prose appears in lines[window_start:window_end] (0-indexed, end exclusive),
    and if so, the absolute 0-indexed position of the first such line. Returns None if not found."""
    result = _call_llm(model_id, FIND_STORY_INSTRUCTIONS, window_start, window_end, lines,
                        "found_line", FOUND_LINE_SCHEMA, debug)
    if result is None or not result.get("found"):
        if debug:
            print(f"    [find_story_line {window_start + 1}-{window_end}] not found")
        return None
    try:
        raw_line = int(result.get("line", 0))
    except (TypeError, ValueError):
        return None
    idx = max(window_start, min(raw_line - 1, window_end - 1))
    if debug:
        print(f"    [find_story_line {window_start + 1}-{window_end}] found at line {idx + 1}")
    return idx


def find_metadata_boundary(model_id, lines, window_start, window_end, debug=False):
    """Ask whether metadata begins within lines[window_start:window_end] (0-indexed, end exclusive),
    and if so, the absolute 0-indexed position of the last story line before it. Returns None if no
    metadata is found in this window."""
    result = _call_llm(model_id, SCAN_INSTRUCTIONS, window_start, window_end, lines,
                        "boundary", BOUNDARY_SCHEMA, debug)
    if result is None or not result.get("metadata_found"):
        if debug:
            print(f"    [find_metadata_boundary {window_start + 1}-{window_end}] no metadata found")
        return None
    try:
        raw_line = int(result.get("line", 0))
    except (TypeError, ValueError):
        return None
    idx = max(window_start, min(raw_line - 1, window_end - 1))
    if debug:
        print(f"    [find_metadata_boundary {window_start + 1}-{window_end}] boundary at line {idx + 1}")
    return idx


def scan_forward_for_resumption(model_id, lines, start_pos, debug=False, force=False):
    """
    Scan forward from start_pos in WINDOW_LINES chunks looking for where story resumes.
    Returns (index, gave_up):
      - (idx, False)  — story resumes at 0-indexed idx
      - (None, False) — legitimately reached end of file with no story found (e.g. a footer)
      - (None, True)  — gave up after MAX_RESUME_WINDOWS without reaching EOF or finding story

    With force=True the MAX_RESUME_WINDOWS cap is ignored and the search runs on to EOF. gave_up
    is then True whenever the cap was exceeded, alongside whatever the longer search found, so
    the caller can still tell that a normal run would have bailed here.
    """
    n = len(lines)
    pos = start_pos
    windows = 0
    while pos < n:
        if windows >= MAX_RESUME_WINDOWS and not force:
            if debug:
                print(f"    [scan_forward_for_resumption] gave up after {MAX_RESUME_WINDOWS} windows, "
                      f"still at line {pos + 1}")
            return None, True
        window_end = min(pos + WINDOW_LINES, n)
        idx = find_story_line(model_id, lines, pos, window_end, debug)
        windows += 1
        if idx is not None:
            return idx, windows > MAX_RESUME_WINDOWS
        pos = window_end
    return None, windows > MAX_RESUME_WINDOWS


# =============================================================
# Applying cuts
# =============================================================

def merge_ranges(ranges):
    """Merge overlapping/adjacent (start, end) 1-indexed inclusive ranges into the fewest spans."""
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def print_cut_preview(lines, cuts, max_lines_per_cut=40):
    """Debug helper: print the actual text of every line about to be removed, so a human can
    eyeball whether it's really metadata without diffing input/output files by hand."""
    merged = merge_ranges(cuts)
    if not merged:
        print("  [cuts] none")
        return
    for start, end in merged:
        count = end - start + 1
        print(f"  [cut {start}-{end}] ({count} line{'s' if count != 1 else ''})")
        block = range(start, end + 1)
        if count > max_lines_per_cut:
            half = max_lines_per_cut // 2
            for i in list(block)[:half]:
                print(f"    {i}: {lines[i - 1]}")
            print(f"    ... {count - max_lines_per_cut} lines omitted ...")
            for i in list(block)[-half:]:
                print(f"    {i}: {lines[i - 1]}")
        else:
            for i in block:
                print(f"    {i}: {lines[i - 1]}")


def apply_cuts(lines, cuts):
    """
    cuts: list of (start_line, end_line), 1-indexed inclusive. Returns (kept_lines, removed_count).
    A blank line is inserted between two kept segments wherever a cut joined them, so removing a
    metadata block doesn't visually weld two unrelated paragraphs together with no spacing.
    """
    if not cuts:
        return list(lines), 0

    merged = merge_ranges(cuts)

    keep = []
    removed = 0
    cursor = 1
    for start, end in merged:
        if cursor < start:
            if keep:
                keep.append("")
            keep.extend(lines[cursor - 1:start - 1])
        removed += end - start + 1
        cursor = end + 1
    if cursor <= len(lines):
        if keep:
            keep.append("")
        keep.extend(lines[cursor - 1:])

    return keep, removed


# =============================================================
# File I/O
# =============================================================

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1252") as f:
            return f.read()


def scan_pass(model_id, lines, debug=False, pass_label="", force=False):
    """
    Run one full start/scan/resume pass over `lines` (0-indexed list, fresh numbering — not
    necessarily the original file's line numbers if this is a second pass over already-cleaned
    text). Returns (cuts, gave_up_reason): cuts is a list of 1-indexed inclusive ranges relative to
    THIS input; gave_up_reason is None on success or a string describing why the pass bailed.

    With force=True the pass never bails: it cuts whatever the model judged to be non-story and
    always returns cuts, and gave_up_reason instead lists every point where a normal run would
    have bailed (None if there were none).
    """
    total_lines = len(lines)
    prefix = f"[{pass_label}] " if pass_label else ""
    overridden = []

    start_idx, gave_up = scan_forward_for_resumption(model_id, lines, 0, debug, force)
    if gave_up:
        reason = f"{prefix}gave up looking for the story start"
        if not force:
            return None, reason
        overridden.append(reason)
    if start_idx is None:
        reason = f"{prefix}no story prose found anywhere in the text"
        if not force:
            return None, reason
        return [(1, total_lines)], "; ".join(overridden + [reason])
    if debug:
        print(f"  {prefix}[start] story begins at line {start_idx + 1}")

    cuts = []
    if start_idx > 0:
        cuts.append((1, start_idx))

    cursor = start_idx
    while cursor < total_lines:
        window_end = min(cursor + WINDOW_LINES, total_lines)
        if window_end < total_lines:
            snap = window_end
            while snap > cursor and lines[snap - 1].strip() != "":
                snap -= 1
            if snap > cursor:
                window_end = snap

        boundary_idx = find_metadata_boundary(model_id, lines, cursor, window_end, debug)
        if boundary_idx is None:
            cursor = window_end
            continue

        resume_idx, gave_up = scan_forward_for_resumption(model_id, lines, boundary_idx + 1, debug, force)
        if gave_up:
            reason = f"{prefix}gave up searching for where the story resumes after line {boundary_idx + 2}"
            if not force:
                return None, reason
            overridden.append(reason)
        if resume_idx is None:
            # metadata runs to end of text — this is how a trailing footer gets handled
            cuts.append((boundary_idx + 2, total_lines))
            cursor = total_lines
        else:
            if resume_idx > boundary_idx + 1:
                cuts.append((boundary_idx + 2, resume_idx))
            cursor = resume_idx

    if debug:
        print(f"  {prefix}[cuts] {len(cuts)} range(s):")
        print_cut_preview(lines, cuts)

    return cuts, "; ".join(overridden) or None


def scan_and_verify(model_id, lines, debug=False, pass_label="", force=False):
    """One full scan_pass over `lines` plus the sandwiched-gap verification, returning
    (cuts, gave_up_reason) the same way scan_pass does. `cuts` is 1-indexed inclusive ranges
    relative to `lines`."""
    cuts, gave_up_reason = scan_pass(model_id, lines, debug, pass_label=pass_label, force=force)
    if cuts is None:
        return None, gave_up_reason

    extra_cuts = verify_sandwiched_gaps(model_id, lines, cuts, debug)
    if extra_cuts:
        if debug:
            prefix = f"[{pass_label}] " if pass_label else ""
            print(f"  {prefix}[verify] {len(extra_cuts)} additional range(s) confirmed as metadata:")
            print_cut_preview(lines, extra_cuts)
        cuts = cuts + extra_cuts

    return cuts, gave_up_reason


def clean_file(input_path, output_dir, review_dir, model_id, debug=False, force=False):
    """Clean one file into cleaned/, or copy it untouched into review/ if a safety guard trips.

    With force=True a tripped guard is overridden instead: the cuts are applied anyway and the
    result is written to review/ in place of the untouched original, so it can be diffed against
    the input file to see what the guard was protecting against. Files that trip no guard are
    unaffected by force and go to cleaned/ as usual."""
    name = os.path.basename(input_path)
    print(f"Processing {name}...")

    text = read_text(input_path)
    lines = text.replace("\r", "").split("\n")
    total_lines = len(lines)
    forced = []  # guards overridden by force; non-empty means the output belongs in review/

    def route_to_review(reason):
        out_path = os.path.join(review_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  -> review/ ({reason})")

    cuts, gave_up_reason = scan_and_verify(model_id, lines, debug, pass_label="scan", force=force)
    if gave_up_reason:
        if not force:
            route_to_review(gave_up_reason)
            return
        forced.append(gave_up_reason)

    if debug:
        print(f"  [cuts] {len(cuts)} total range(s):")
        print_cut_preview(lines, cuts)

    kept_lines, removed_count = apply_cuts(lines, cuts)

    # Heavily-cut files often keep a few residual metadata lines the windowed first pass walked
    # past. Re-scan the cleaned text (fresh numbering) until a pass removes nothing, capped at
    # RESCAN_MAX_PASSES. Any pass that wants to remove more than a small margin is not trusted:
    # the file goes to review/ instead.
    needs_rescan = (removed_count / total_lines if total_lines else 0) >= RESCAN_TRIGGER_RATIO \
        or len(merge_ranges(cuts)) >= RESCAN_TRIGGER_CUTS
    if needs_rescan:
        for pass_num in range(2, RESCAN_MAX_PASSES + 1):
            rescan_cuts, gave_up_reason = scan_and_verify(
                model_id, kept_lines, debug, pass_label=f"rescan {pass_num}")
            if gave_up_reason:
                if debug:
                    print(f"  [rescan {pass_num}] {gave_up_reason} — keeping previous result")
                break

            prev_len = len(kept_lines)
            rescanned_lines, rescan_removed = apply_cuts(kept_lines, rescan_cuts)
            if rescan_removed == 0:
                if debug:
                    print(f"  [rescan {pass_num}] no residual metadata found — converged")
                break

            guard = max(RESCAN_SAFE_MIN_LINES, int(prev_len * RESCAN_SAFE_RATIO))
            if rescan_removed > guard:
                reason = (f"re-scan pass {pass_num} wanted to remove {rescan_removed} more line(s) "
                          f"({rescan_removed / prev_len:.0%} of the cleaned text) — too much to trust")
                if not force:
                    route_to_review(reason)
                    return
                forced.append(reason)

            if debug:
                print(f"  [rescan {pass_num}] removed {rescan_removed} residual line(s):")
                print_cut_preview(kept_lines, rescan_cuts)
            kept_lines = rescanned_lines
            removed_count += rescan_removed

    removed_ratio = removed_count / total_lines if total_lines else 0

    if removed_ratio > REVIEW_MAX_REMOVED_RATIO:
        reason = f"{removed_ratio:.0%} of lines flagged for removal, looked too aggressive"
        if not force:
            route_to_review(reason)
            return
        forced.append(reason)

    out_path = os.path.join(review_dir if forced else output_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines))

    print(f"  Lines: {total_lines} -> {len(kept_lines)} kept "
          f"({removed_count} removed, {removed_ratio:.1%})")
    for reason in forced:
        print(f"  -> review/ FORCED, cuts applied anyway ({reason})")


# =============================================================
# Main
# =============================================================

def normalize_dropped_path(raw):
    """Clean up a path pasted or drag-and-dropped into the terminal.

    Linux file managers wrap the dropped path in single quotes (and shell-escape
    spaces), or hand over a file:// URI. Strip all of that back to a plain path."""
    s = raw.strip()
    if not s:
        return s

    # file:///home/user/My%20Dir -> /home/user/My Dir
    if s.startswith(("file://", "file:")):
        parsed = urlparse(s)
        return unquote(parsed.path)

    # Surrounding matching quotes, or shell-escaped tokens like 'My\ Dir'.
    try:
        parts = shlex.split(s)
        if len(parts) == 1:
            return parts[0]
    except ValueError:
        pass

    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]

    return s


def main():
    parser = argparse.ArgumentParser(description="Remove archive metadata from story text files using an LLM.")
    parser.add_argument("input_file", nargs="?", help="Path to a .txt file to clean. If omitted, you'll be "
                                                        "prompted for a directory to clean every .txt file in.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print scan/LLM debug info")
    parser.add_argument("-f", "--force", action="store_true",
                        help="When a file would be sent to review/ untouched, apply the cuts anyway and "
                             "write the cleaned result to review/ instead, to inspect what went wrong")
    args = parser.parse_args()

    if args.input_file:
        args.input_file = os.path.expanduser(normalize_dropped_path(args.input_file))
        if not os.path.exists(args.input_file):
            print(f"Error: file not found: {args.input_file}")
            sys.exit(1)

        model_id = get_model_info(debug=args.debug)
        print(f"Model: {model_id}")

        base_dir = os.path.dirname(os.path.abspath(args.input_file))
        output_dir = os.path.join(base_dir, "cleaned")
        review_dir = os.path.join(base_dir, "review")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(review_dir, exist_ok=True)

        clean_file(args.input_file, output_dir, review_dir, model_id, debug=args.debug, force=args.force)

    else:
        directory = normalize_dropped_path(input("Enter directory to process: "))
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            print(f"Error: '{directory}' is not a valid directory.")
            sys.exit(1)

        txt_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".txt"))
        if not txt_files:
            print("No .txt files found in that directory.")
            return

        model_id = get_model_info(debug=args.debug)
        print(f"Model: {model_id}")

        output_dir = os.path.join(directory, "cleaned")
        review_dir = os.path.join(directory, "review")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(review_dir, exist_ok=True)

        print(f"Found {len(txt_files)} file(s). Output -> {output_dir}")
        for filename in txt_files:
            clean_file(os.path.join(directory, filename), output_dir, review_dir, model_id,
                       debug=args.debug, force=args.force)

        print(f"\n=== All done. {len(txt_files)} file(s) processed. ===")


if __name__ == "__main__":
    main()
