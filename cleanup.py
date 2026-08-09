import argparse
import json
import os
import sys

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
    """Confirm a model is loaded and its context length covers MIN_CONTEXT_LENGTH."""
    resp = requests.get(MODELS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    loaded = [m for m in models if m.get("state") == "loaded"]
    if not loaded:
        raise RuntimeError("No model is currently loaded in LM Studio.")
    model = loaded[0]
    model_id = model["id"]
    context_length = model.get("loaded_context_length")

    if debug:
        print(f"[model] {model_id} | loaded_context_length={context_length}")

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


def scan_forward_for_resumption(model_id, lines, start_pos, debug=False):
    """
    Scan forward from start_pos in WINDOW_LINES chunks looking for where story resumes.
    Returns (index, gave_up):
      - (idx, False)  — story resumes at 0-indexed idx
      - (None, False) — legitimately reached end of file with no story found (e.g. a footer)
      - (None, True)  — gave up after MAX_RESUME_WINDOWS without reaching EOF or finding story
    """
    n = len(lines)
    pos = start_pos
    for _ in range(MAX_RESUME_WINDOWS):
        if pos >= n:
            return None, False
        window_end = min(pos + WINDOW_LINES, n)
        idx = find_story_line(model_id, lines, pos, window_end, debug)
        if idx is not None:
            return idx, False
        pos = window_end
    if pos >= n:
        return None, False
    if debug:
        print(f"    [scan_forward_for_resumption] gave up after {MAX_RESUME_WINDOWS} windows, "
              f"still at line {pos + 1}")
    return None, True


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


def scan_pass(model_id, lines, debug=False, pass_label=""):
    """
    Run one full start/scan/resume pass over `lines` (0-indexed list, fresh numbering — not
    necessarily the original file's line numbers if this is a second pass over already-cleaned
    text). Returns (cuts, gave_up_reason): cuts is a list of 1-indexed inclusive ranges relative to
    THIS input; gave_up_reason is None on success or a string describing why the pass bailed.
    """
    total_lines = len(lines)
    prefix = f"[{pass_label}] " if pass_label else ""

    start_idx, gave_up = scan_forward_for_resumption(model_id, lines, 0, debug)
    if gave_up:
        return None, f"{prefix}gave up looking for the story start"
    if start_idx is None:
        return None, f"{prefix}no story prose found anywhere in the text"
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

        resume_idx, gave_up = scan_forward_for_resumption(model_id, lines, boundary_idx + 1, debug)
        if gave_up:
            return None, f"{prefix}gave up searching for where the story resumes after line {boundary_idx + 2}"
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

    return cuts, None


def clean_file(input_path, output_dir, review_dir, model_id, debug=False):
    name = os.path.basename(input_path)
    print(f"Processing {name}...")

    text = read_text(input_path)
    lines = text.replace("\r", "").split("\n")
    total_lines = len(lines)

    def route_to_review(reason):
        out_path = os.path.join(review_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  -> review/ ({reason})")

    cuts, gave_up_reason = scan_pass(model_id, lines, debug, pass_label="scan")
    if gave_up_reason:
        route_to_review(gave_up_reason)
        return

    extra_cuts = verify_sandwiched_gaps(model_id, lines, cuts, debug)
    if extra_cuts:
        if debug:
            print(f"  [verify] {len(extra_cuts)} additional range(s) confirmed as metadata:")
            print_cut_preview(lines, extra_cuts)
        cuts = cuts + extra_cuts

    if debug:
        print(f"  [cuts] {len(cuts)} total range(s):")
        print_cut_preview(lines, cuts)

    kept_lines, removed_count = apply_cuts(lines, cuts)
    removed_ratio = removed_count / total_lines if total_lines else 0

    if removed_ratio > REVIEW_MAX_REMOVED_RATIO:
        route_to_review(f"{removed_ratio:.0%} of lines flagged for removal, looked too aggressive")
        return

    out_path = os.path.join(output_dir, name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_lines))

    print(f"  Lines: {total_lines} -> {len(kept_lines)} kept "
          f"({removed_count} removed, {removed_ratio:.1%})")


# =============================================================
# Main
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="Remove archive metadata from story text files using an LLM.")
    parser.add_argument("input_file", nargs="?", help="Path to a .txt file to clean. If omitted, you'll be "
                                                        "prompted for a directory to clean every .txt file in.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print scan/LLM debug info")
    args = parser.parse_args()

    if args.input_file:
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

        clean_file(args.input_file, output_dir, review_dir, model_id, debug=args.debug)

    else:
        directory = input("Enter directory to process: ").strip()
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
            clean_file(os.path.join(directory, filename), output_dir, review_dir, model_id, debug=args.debug)

        print(f"\n=== All done. {len(txt_files)} file(s) processed. ===")


if __name__ == "__main__":
    main()
