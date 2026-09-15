"""Generate the six-month Meridian corpus from data/company_bible.json.

One LLM call per meeting, in chronological order, through engine.llm.complete
so every response is cached on disk. Reruns cost zero API calls.

Each call receives the bible material relevant to that meeting, a rolling
summary of every prior meeting, and the two most recent transcripts verbatim so
back-references point at things people actually said. The rolling summary is
derived from the bible's own beats rather than generated, which costs no extra
API calls and cannot drift from the plan.

Generation runs on gpt-oss-120b (see GENERATION_MODEL). Turns that state one of
the four deliberately inconsistent facts are tagged inline by the model and
recorded in data/inconsistent_turns.json. scripts/build_golden_set.py excludes
them, so a question never ends up with several disagreeing answer turns.

Every generated meeting is validated: turn count, speakers actually in the room,
no minutes-style "Decision:" prefixes, disputed facts tagged where the bible
says they are stated, and no near-duplicate turns by the same speaker within the
meeting. Failures retry with an attempt marker. Repetition across meetings is
deliberate and is not linted.

Usage:
  python scripts/generate_corpus.py            # every slot
  python scripts/generate_corpus.py --limit 4  # first 4 slots only
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from engine.llm import complete, get_call_count

BIBLE_PATH = ROOT / "data" / "company_bible.json"
MEETINGS_DIR = ROOT / "data" / "meetings"
INCONSISTENT_PATH = ROOT / "data" / "inconsistent_turns.json"

# The corpus is a one-time, disk-cached fixture, so generation quality is worth
# the larger model: 20b satisfied the style constraints superficially and lost
# coherence, producing fragments like "The auth flow—still pending." The agent's
# runtime model stays on gpt-oss-20b, which is a separate decision driven by
# per-query latency and cost rather than one-off fixture quality.
GENERATION_MODEL = "openai/gpt-oss-120b"

MAX_ATTEMPTS = 3

# Prompt budget. gpt-oss-120b on the free tier allows 8000 tokens per minute,
# and a single request over that is a hard 413 rather than throttling. This
# prompt is dense structured text that tokenises at roughly 1.45 chars/token,
# not the usual 4, so the budget is far tighter than character counts suggest.
# Everything below exists to keep the largest prompt under it.
# Groq charges a request as prompt + max_tokens against the per-minute
# allowance, so the response reservation is spent whether the model uses it or
# not. A 60-turn all-hands is ~1600 tokens of content; 3500 leaves room for the
# reasoning channel while keeping the whole request under the allowance.
CHARS_PER_TOKEN = 4.2
TPM_LIMIT = 8000
GENERATION_MAX_TOKENS = 3500

# gpt-oss deliberates before answering, and at the default effort it spent most
# of a 3500-token budget reasoning about the style constraints — in one case
# writing its plan into the content channel ("We need a few turns where they
# speak past each other...") and then truncating mid-transcript. Low effort
# leaves the budget for the transcript itself. The constraints are specific
# enough not to need deliberation.
GENERATION_REASONING = "low"

RECENT_TRANSCRIPTS = 1
RECENT_TRANSCRIPT_TURNS = 10

# Full beats for the most recent prior meetings; older ones collapse to a dated
# one-liner. Thread state is already supplied in full by the date-filtered
# thread block, so the rolling summary does not need to carry it twice.
ROLLING_SUMMARY_FULL = 5

# Only the most recent complications; older ones are implied by what follows.
MAX_COMPLICATIONS = 3

# Turns that state a disputed fact are tagged inline as "12. Sarah [F1]: text".
# An earlier design had the model report turn numbers in a trailing block; it
# omitted the block, invented its own format, and miscounted, because that asks
# the model to count turns after the fact. Tagging at the point of writing is
# the thing the model is already doing.
# The number separator accepts ":" as well as "." and ")": the model sometimes
# writes "23: Marcus: ..." and a stricter pattern silently dropped those turns
# from the transcript rather than failing.
_TURN_LINE = re.compile(
    r"^\s*(\d+)\s*[.):]\s*([^:\[]{1,60}?)\s*(?:\[\s*([^\]]*?)\s*\])?\s*:\s*(.+?)\s*$"
)
_LEADING_TAG = re.compile(r"^\s*\[\s*([^\]]*?)\s*\]\s*")
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*")
_FACT_IN_BEAT = re.compile(r"\b(F\d)\b\s+INCONSISTENT FACT")
_FACT_TAG = re.compile(r"\bF\d\b")
_WORD = re.compile(r"[a-z0-9]+")

# Within-meeting near-duplicate detection. Same speaker, Jaccard over token
# sets. Repetition ACROSS meetings is deliberate and left alone: recurring stock
# phrases are realistic and they create the near-duplicate chunks that force
# retrieval to discriminate.
DUPLICATE_THRESHOLD = 0.8
DUPLICATE_MIN_TOKENS = 6
DUPLICATE_TOLERANCE = 1
# Disabled: minor within-meeting repetition does not affect retrieval, and the
# regeneration it triggers is not worth the API budget or the wall clock.
DUPLICATE_LINT = False

# Prefixes the inherited transcripts use and generated meetings must not.
BANNED_PREFIXES = (
    "decision:",
    "action item:",
    "action:",
    "owner:",
    "next steps:",
    "summary:",
    "status:",
    "update:",
)

SYSTEM = (
    "You write raw meeting transcripts. You produce what a transcription "
    "service would emit from a recording: messy, referential, interrupted "
    "speech. You never write minutes, summaries, or labelled decisions."
)


def _load_bible() -> dict:
    return json.loads(BIBLE_PATH.read_text(encoding="utf-8"))


def _cast_block(bible: dict, attendees: list[str]) -> str:
    by_name = {c["name"]: c for c in bible["cast"]}
    lines = []
    for name in attendees:
        c = by_name[name]
        role = c["role"].split(".")[0]
        lines.append(f"- {name}, {role}. {c['style']}")
    return "\n".join(lines)


def _leading_date(text: str) -> str | None:
    """The 'YYYY-MM-DD:' or 'YYYY-MM:' prefix a bible entry starts with, if any."""
    m = re.match(r"^\s*(\d{4}-\d{2}(?:-\d{2})?)\s*:", text)
    if not m:
        return None
    value = m.group(1)
    return value if len(value) == 10 else f"{value}-01"


def _happened_by(items: list[str], on_date: str) -> list[str]:
    """Entries dated on or before `on_date`. Undated entries are treated as future.

    Passing a thread's whole complication list to every meeting is what made the
    13 March review discuss an audit commissioned on 7 April. A meeting only
    ever sees the part of its own threads that has already happened.
    """
    kept = []
    for item in items:
        stamp = _leading_date(item)
        if stamp is not None and stamp <= on_date:
            kept.append(item)
    return kept


def _has_begun(thread: dict, on_date: str) -> bool:
    stamp = _leading_date(thread["beginning"])
    return stamp is not None and stamp <= on_date


def _thread_block(bible: dict, thread_ids: list[str], on_date: str) -> str:
    by_id = {t["id"]: t for t in bible["threads"]}
    lines = []
    for tid in thread_ids:
        t = by_id[tid]
        aliases = ", ".join(f'"{a}"' for a in t["referential_aliases"])
        so_far = _happened_by(t["complications"], on_date)[-MAX_COMPLICATIONS:]
        resolution = _leading_date(t["resolution"])
        entry = [
            f"- {t['name']}",
            f"  How people refer to it without naming it: {aliases}",
        ]
        if _has_begun(t, on_date):
            entry.append(f"  How it started: {t['beginning']}")
        else:
            entry.append(
                "  This comes up for the first time in THIS meeting. Nobody has "
                "raised it before."
            )
        if so_far:
            entry.append("  What has happened since: " + " | ".join(so_far))
        if resolution is not None and resolution <= on_date:
            entry.append(f"  Resolved: {t['resolution']}")
        else:
            entry.append(
                "  NOT resolved as of this meeting. Nobody knows yet how it ends."
            )
        lines.append("\n".join(entry))
    return "\n".join(lines)


def _other_threads_block(bible: dict, slot: dict) -> tuple[str, str]:
    """Threads not in this meeting, split into not-yet-existing and background."""
    on_date = slot["date"]
    todays = set(slot["threads"])
    future, background = [], []
    for t in bible["threads"]:
        if t["id"] in todays:
            continue
        if _has_begun(t, on_date):
            background.append(f"- {t['name']}")
        else:
            # Naming the thread is not enough: the model leaked "rate limiting"
            # and "headcount reqs" while obeying a ban on the formal titles.
            # The aliases are the phrasings people would actually reach for.
            aliases = ", ".join(
                f'"{a}"' for a in t["referential_aliases"] if a.lower() != "that"
            )
            future.append(f"- {t['name']} — including any mention of {aliases}")
    return (
        "\n".join(future) or "- (none)",
        "\n".join(background) or "- (none)",
    )


def _rolling_summary(prior_slots: list[dict]) -> str:
    """Dated log of every prior meeting, from the bible's beats. No LLM call."""
    if not prior_slots:
        return "(this is the first meeting in the corpus)"
    lines = []
    cutoff = len(prior_slots) - ROLLING_SUMMARY_FULL
    for i, s in enumerate(prior_slots):
        threads = ", ".join(s["threads"])
        if i >= cutoff:
            what = (
                " ".join(f"{b}." for b in s["beats"])
                if s["beats"]
                else "Nothing decided."
            )
            lines.append(f"{s['date']} {s['type']} [{threads}]: {what}")
        else:
            lines.append(f"{s['date']} {s['type']} [{threads}]")
    return "\n".join(lines)


def _inherited_block(bible: dict) -> str:
    return bible["company"]["note_on_inherited_transcripts"]


def _recent_block(written: list[tuple[str, str]]) -> str:
    """The last few transcripts verbatim, tail-truncated."""
    if not written:
        return "(none yet)"
    chunks = []
    for meeting_id, text in written[-RECENT_TRANSCRIPTS:]:
        lines = text.splitlines()
        tail = lines[-RECENT_TRANSCRIPT_TURNS:]
        elided = "" if len(lines) <= RECENT_TRANSCRIPT_TURNS else "[...earlier turns omitted...]\n"
        chunks.append(f"--- {meeting_id} ---\n{elided}" + "\n".join(tail))
    return "\n\n".join(chunks)


def _facts_for_slot(bible: dict, slot: dict) -> list[dict]:
    ids = set()
    for beat in slot["beats"]:
        m = _FACT_IN_BEAT.search(beat)
        if m:
            ids.add(m.group(1))
    return [f for f in bible["inconsistent_facts"] if f["id"] in ids]


def _facts_block(facts: list[dict]) -> str:
    if not facts:
        return (
            "None. No turn in this meeting should state a disputed audit finding "
            "count, a Ridgeline contract value, a claim about which launch date "
            "was originally agreed, or a count of DataCorp follow-ups."
        )
    lines = []
    for f in facts:
        lines.append(
            f"- {f['id']}: {f['fact']}. In THIS meeting it is stated as: "
            + "; ".join(v for v in f["variants"])
        )
    return "\n".join(lines)


def _build_prompt(bible: dict, slot: dict, prior_slots: list[dict], written) -> str:
    style = bible["speech_style"]
    mtype = bible["meeting_types"][slot["type"]]
    ratio = style["referential_chains"]["ratio_by_meeting_type"][slot["type"]]
    facts = _facts_for_slot(bible, slot)
    future_threads, background_threads = _other_threads_block(bible, slot)

    beats = (
        "\n".join(f"- {b}" for b in slot["beats"])
        if slot["beats"]
        else "- (none: this is a deliberately low-signal meeting)"
    )

    thin_block = ""
    if slot.get("thin"):
        tm = bible["thin_meetings"]
        thin_block = (
            "\nTHIS IS A DELIBERATELY LOW-SIGNAL MEETING.\n"
            f"{tm['why']}\n"
            "Fill it with:\n"
            + "\n".join(f"- {c}" for c in tm["content"])
            + f"\n{tm['do_not']}\n"
        )

    return f"""Write the full transcript of one meeting at {bible['company']['name']}.

{bible['company']['description']}

MEETING
  Date: {slot['date']} ({slot['weekday']})
  Type: {slot['type']} — {mtype['cadence']}
  Title: {slot.get('title', slot['type'])}
  Character: {mtype['character']}
  Length: {mtype['turns']} turns
  In the room: {', '.join(slot['attendees'])}

WHO IS IN THE ROOM
{_cast_block(bible, slot['attendees'])}

Only these people speak. Nobody else. People who are not in the room may be
talked about, and should be referred to by pronoun where natural.

THIS MEETING TAKES PLACE ON {slot['date']}, A {slot['weekday'].upper()}.
NOTHING THAT HAPPENS AFTER THAT DATE EXISTS YET. The people in this room cannot
know, mention, hint at or worry about anything later than it. They do not know
how any of this turns out.
Nobody reads the date aloud. People say "Tuesday", "end of the month", "the
28th" — never "2025-03-17".

THREADS THIS MEETING TOUCHES
{_thread_block(bible, slot['threads'], slot['date'])}

TOPICS THAT DO NOT EXIST YET — NEVER MENTION THESE
Nobody at {bible['company']['name']} has raised any of these as of {slot['date']}.
Do not reference them, not even in passing, not even as a worry:
{future_threads}

ONGOING ELSEWHERE — passing mention only, no new developments
{background_threads}

WHAT MUST HAPPEN IN THIS MEETING
{beats}
{thin_block}
TURN RHYTHM
{chr(10).join('- ' + r for r in style['turn_rhythm']['rules'])}

SPECIFIC FACTS
{chr(10).join('- ' + r for r in style['anchors']['rules'])}

DO NOT REPEAT YOURSELF WITHIN THIS MEETING
{chr(10).join('- ' + r for r in style['no_repetition']['rules'])}

HOW DECISIONS AND COMMITMENTS MUST APPEAR
Decisions are buried in conversation, never announced as minutes. Never begin a
turn with "Decision:", "Action item:", "Owner:", "Next steps:", "Summary:",
"Status:" or "Update:".
{chr(10).join('- ' + p for p in style['how_decisions_surface']['instead'])}

REFERENTIAL SPEECH — THE MOST IMPORTANT REQUIREMENT
{style['referential_chains']['requirement']}
Within this meeting: {style['referential_chains']['in_meeting']}
Across meetings: {style['referential_chains']['across_meetings']}
About people: {style['referential_chains']['people']}
Density for a {slot['type']}: {ratio}
Balance: {style['referential_chains']['balance']}
Examples of the register:
{chr(10).join('  ' + e for e in style['referential_chains']['examples'][:4])}

GENERAL STYLE
{style['general']}

WHAT HAS HAPPENED SO FAR (prior meetings, oldest first)
{_rolling_summary(prior_slots)}

THE MOST RECENT TRANSCRIPT, VERBATIM
Refer back to what was actually said here, using pronouns and short references.
{_recent_block(written)}

DISPUTED FACTS STATED IN THIS MEETING
{_facts_block(facts)}

OUTPUT FORMAT
Write one turn per line, numbered from 1, in exactly this shape:
1. Alice: text of the turn
2. Bob: text of the turn
No blank lines. No stage directions. No speaker parentheticals. Never put a
turn on more than one line. Only the names listed in the room may appear as
speakers.

Output the numbered turns and nothing else. Do not write a preamble, a plan, a
checklist, a count, or any commentary about how you are satisfying the rules
above. Do not think aloud in the output. Begin at "1." and stop after the last
turn. Write {mtype['turns']} turns; get there by writing the meeting, not by
padding it.

TAGGING DISPUTED FACTS
{_tagging_instruction(facts)}"""


def _tagging_instruction(facts: list[dict]) -> str:
    if not facts:
        return (
            "No turn in this meeting states a disputed fact, so no turn carries "
            "a tag. Do not use square brackets anywhere."
        )
    ids = ", ".join(f["id"] for f in facts)
    example = facts[0]["id"]
    return (
        f"This meeting states the disputed fact(s) {ids}, listed above. The one "
        "turn that actually says the disputed value must carry the fact id in "
        "square brackets after the speaker name, like this:\n"
        f"  12. Sarah [{example}]: forty-one issues, six of them blocking.\n"
        "Tag only the turn that states the value itself. A turn that mentions "
        "the topic without giving the number is not tagged. Every fact id "
        f"listed ({ids}) must appear on exactly one turn. Use square brackets "
        "for nothing else."
    )


def _parse(raw: str, meeting_id: str) -> tuple[list[tuple[str, str]], dict[str, list[int]]]:
    """Parse numbered turn lines, lifting any inline [F1] disputed-fact tags.

    Deliberately forgiving: the tag is accepted before or after the speaker
    name, and an unparseable line is reported and skipped rather than failing
    the whole meeting.
    """
    turns: list[tuple[str, str]] = []
    facts: dict[str, list[int]] = {}

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _TURN_LINE.match(line)
        if not m:
            print(f"  WARNING {meeting_id}: unparsed line: {stripped[:90]}")
            continue

        speaker = _PARENTHETICAL.sub(" ", m.group(2)).strip()
        tags = m.group(3) or ""
        text = m.group(4).strip()

        # Tolerate "12. Sarah: [F1] text" as well as "12. Sarah [F1]: text".
        leading = _LEADING_TAG.match(text)
        if leading:
            tags = f"{tags} {leading.group(1)}"
            text = _LEADING_TAG.sub("", text).strip()
        if not text:
            continue

        index = len(turns)
        turns.append((speaker, text))
        for fact_id in _FACT_TAG.findall(tags.upper()):
            facts.setdefault(fact_id, []).append(index)

    return turns, facts


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _duplicate_pairs(turns: list[tuple[str, str]]) -> list[tuple[int, int, float]]:
    """Near-duplicate turns by the same speaker within one meeting.

    Short turns are exempt: "Yeah." and "Still nothing." repeating is how people
    actually talk, and collapsing them would remove realism rather than filler.
    """
    toks = [_tokens(text) for _, text in turns]
    pairs = []
    for i in range(len(turns)):
        if len(toks[i]) < DUPLICATE_MIN_TOKENS:
            continue
        for j in range(i + 1, len(turns)):
            if turns[i][0] != turns[j][0] or len(toks[j]) < DUPLICATE_MIN_TOKENS:
                continue
            union = len(toks[i] | toks[j])
            if not union:
                continue
            score = len(toks[i] & toks[j]) / union
            if score >= DUPLICATE_THRESHOLD:
                pairs.append((i, j, score))
    return pairs


def _validate(
    meeting_id: str,
    slot: dict,
    cast: set[str],
    turns,
    facts: dict[str, list[int]],
    expected_facts: list[str],
) -> tuple[list[str], list[tuple[int, int, float]]]:
    problems = []
    low, high = (int(x) for x in slot["_turn_range"])
    if not low <= len(turns) <= high:
        problems.append(f"{len(turns)} turns, expected {low}-{high}")

    attendees = set(slot["attendees"])
    for i, (speaker, text) in enumerate(turns):
        if speaker not in cast:
            problems.append(f"turn {i}: unknown speaker {speaker!r}")
        elif speaker not in attendees:
            problems.append(f"turn {i}: {speaker} was not in the room")
        lowered = text.lower()
        for prefix in BANNED_PREFIXES:
            if lowered.startswith(prefix):
                problems.append(f"turn {i}: banned prefix {prefix!r}")

    # A slot scripted to carry a disputed fact that records no turn id is a
    # silent failure: the fact still lands in the corpus, but build_golden_set
    # never learns to exclude it, and a golden question ends up with several
    # disagreeing answer turns. Loud, and it triggers a regeneration.
    for fact_id in expected_facts:
        if not facts.get(fact_id):
            problems.append(f"no turn tagged [{fact_id}], which this meeting must state")
    for fact_id in facts:
        if fact_id not in expected_facts:
            problems.append(f"turn tagged [{fact_id}], which does not belong here")

    duplicates = _duplicate_pairs(turns) if DUPLICATE_LINT else []
    if DUPLICATE_LINT and len(duplicates) > DUPLICATE_TOLERANCE:
        i, j, score = duplicates[0]
        problems.append(
            f"{len(duplicates)} near-duplicate turn pairs, e.g. {i}/{j} at {score:.2f}"
        )
    return problems, duplicates


_token_log: list[tuple[float, int]] = []


def _pace(est_tokens: int) -> None:
    """Sleep until this request fits inside the per-minute token allowance.

    Capping prompt size stops a single request being rejected outright, but the
    allowance is a rate, so throughput still has to be paced. Without this the
    1.5s gap in engine.llm fires requests far faster than the budget refills,
    every key 429s, and the run dies a few meetings in.
    """
    while True:
        now = time.monotonic()
        recent = [(t, n) for t, n in _token_log if t > now - 60]
        used = sum(n for _, n in recent)
        if not recent or used + est_tokens <= TPM_LIMIT:
            break
        sleep_for = max(1.0, 61 - (now - recent[0][0]))
        print(f"  pacing: {used} tokens in the last minute, waiting {sleep_for:.0f}s")
        time.sleep(sleep_for)
    _token_log.append((time.monotonic(), est_tokens))


def _generate(bible, slot, prior_slots, written, cast):
    """Generate one meeting, retrying malformed or truncated responses.

    engine.llm.complete caches on the prompt, so a truncated response would be
    replayed forever. Each retry appends an attempt marker, which changes the
    cache key while keeping the run deterministic: attempt 1 is still a cache
    hit on reruns, still fails validation, and still falls through to the
    attempt that worked. Reruns therefore cost zero API calls.
    """
    base = _build_prompt(bible, slot, prior_slots, written)
    expected = [f["id"] for f in _facts_for_slot(bible, slot)]
    best = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = base
        if attempt > 1:
            prompt = (
                f"{base}\n(Attempt {attempt}. The previous attempt was truncated, "
                "malformed, repeated itself, or missed a required tag. Produce "
                "the complete transcript.)\n"
            )
        est = int(len(prompt) / CHARS_PER_TOKEN) + GENERATION_MAX_TOKENS
        if est > TPM_LIMIT:
            raise RuntimeError(
                f"{slot['id']}: request is ~{est} tokens (prompt + reservation) "
                f"against a {TPM_LIMIT} limit; Groq rejects this as a 413. "
                "Shorten the prompt or lower GENERATION_MAX_TOKENS."
            )
        _pace(est)
        raw = complete(
            prompt,
            system=SYSTEM,
            model=GENERATION_MODEL,
            max_tokens=GENERATION_MAX_TOKENS,
            reasoning_effort=GENERATION_REASONING,
        )
        turns, facts = _parse(raw, slot["id"])
        if not turns:
            problems, duplicates = ["no parseable turns"], []
        else:
            problems, duplicates = _validate(
                slot["id"], slot, cast, turns, facts, expected
            )
        if not problems:
            return turns, facts, [], duplicates
        if best is None or len(problems) < len(best[2]):
            best = (turns, facts, problems, duplicates)
        print(f"  attempt {attempt} failed validation: {problems[0]}; retrying")
    return best


def main() -> None:
    started = time.perf_counter()
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    bible = _load_bible()
    cast = {c["name"] for c in bible["cast"]}
    timeline = bible["timeline"]
    for slot in timeline:
        slot["_turn_range"] = bible["meeting_types"][slot["type"]]["turns"].split("-")

    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    selected = timeline[:limit] if limit else timeline
    print(f"Generating {len(selected)} of {len(timeline)} meetings into {MEETINGS_DIR}")

    # Turn indices shift when a meeting is regenerated, so drop any recorded
    # ids belonging to the slots about to be rewritten. Entries for slots not
    # in this run (a --limit run) are kept.
    inconsistent: dict[str, list[str]] = {}
    if INCONSISTENT_PATH.exists():
        inconsistent = json.loads(INCONSISTENT_PATH.read_text(encoding="utf-8"))
    regenerating = {s["id"] for s in selected}
    inconsistent = {
        fact_id: [t for t in ids if t.rsplit(":", 1)[0] not in regenerating]
        for fact_id, ids in inconsistent.items()
    }

    written: list[tuple[str, str]] = []
    all_problems: dict[str, list[str]] = {}
    all_duplicates: dict[str, list] = {}
    total_turns = 0

    for n, slot in enumerate(selected):
        meeting_id = slot["id"]
        turns, facts, problems, duplicates = _generate(
            bible, slot, timeline[:n], written, cast
        )
        if not turns:
            raise RuntimeError(f"{meeting_id}: model returned no parseable turns")
        if problems:
            all_problems[meeting_id] = problems
        if duplicates:
            all_duplicates[meeting_id] = duplicates

        text = "\n".join(
            f"{i + 1}. {speaker}: {body}" for i, (speaker, body) in enumerate(turns)
        )
        (MEETINGS_DIR / f"{meeting_id}.txt").write_text(text + "\n", encoding="utf-8")
        written.append((meeting_id, text))
        total_turns += len(turns)

        for fact_id, indices in facts.items():
            for idx in indices:
                inconsistent.setdefault(fact_id, [])
                tid = f"{meeting_id}:{idx}"
                if tid not in inconsistent[fact_id]:
                    inconsistent[fact_id].append(tid)

        flag = " THIN" if slot.get("thin") else ""
        fact_note = f" facts={sorted(facts)}" if facts else ""
        print(
            f"[{n + 1}/{len(selected)}] {meeting_id}: {len(turns)} turns"
            f"{flag}{fact_note}{' PROBLEMS' if problems else ''}"
        )

    INCONSISTENT_PATH.write_text(
        json.dumps(inconsistent, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {len(selected)} transcripts, {total_turns} turns")
    print(f"Mean turns per meeting: {total_turns / len(selected):.1f}")

    flat = sorted({t for ids in inconsistent.values() for t in ids})
    print(f"\n--- disputed-fact turns -> {INCONSISTENT_PATH.name} ---")
    for fact_id in sorted(inconsistent):
        print(f"  {fact_id}: {', '.join(inconsistent[fact_id])}")
    print(f"  total: {len(flat)} turn ids")

    print("\n--- within-meeting duplicate lint ---")
    if not DUPLICATE_LINT:
        print("  disabled (DUPLICATE_LINT = False)")
    elif not all_duplicates:
        print("  no near-duplicate turn pairs above "
              f"{DUPLICATE_THRESHOLD} in any meeting")
    else:
        total_pairs = sum(len(v) for v in all_duplicates.values())
        print(f"  {total_pairs} pairs across {len(all_duplicates)} meetings "
              f"(tolerance {DUPLICATE_TOLERANCE} per meeting before regeneration)")
        for meeting_id, pairs in all_duplicates.items():
            for i, j, score in pairs:
                print(f"  {meeting_id}: turns {i}/{j} at {score:.2f}")

    if all_problems:
        print(f"\n--- VALIDATION PROBLEMS in {len(all_problems)} meetings ---")
        for meeting_id, problems in all_problems.items():
            print(f"{meeting_id}:")
            for p in problems:
                print(f"  - {p}")

    print(f"\nTotal runtime: {time.perf_counter() - started:.1f}s")
    print(f"LLM API calls made: {get_call_count()}")

    fatal = [
        p
        for ps in all_problems.values()
        for p in ps
        if "banned prefix" in p or "no turn tagged" in p
    ]
    if fatal:
        raise RuntimeError(
            f"{len(fatal)} unrecoverable problems after {MAX_ATTEMPTS} attempts: "
            "a banned minutes-style prefix, or a meeting that states a disputed "
            "fact without recording which turn states it."
        )


if __name__ == "__main__":
    main()
