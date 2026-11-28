"""
Task 3, step 1: find phrases that point at something without naming it.

Rule (option 3): find every "this / that / these / those / here / there",
then drop the uses that are known NOT to point at anything (see EXCLUSIONS).

Every hit is saved, including the excluded ones and the reason, so you can
check whether the exclusions are right.

Output: task3/outputs/candidates.csv
    python task3/detect_deixis.py
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "task1"))
import config                                    # noqa: E402
from ami_reference import load_reference         # noqa: E402
from elevenlabs_client import response_to_words  # noqa: E402

MODEL = "scribe_v2"          # which ElevenLabs transcript to search
OUT = ROOT / "task3" / "outputs" / "candidates.csv"
TIMELINE_DIR = ROOT / "task2" / "outputs"
NOTES = ROOT / "task2" / "slide_notes.csv"

# The pointing words we look for ("that's" and "there's" count too).
TRIGGERS = {"this", "that", "these", "those", "here", "there", "that's", "there's", "here's"}

# ---------------------------------------------------------------------------
# Exclusions: uses of the trigger words that do NOT point at anything.
# Each rule looks at the word before (prev) and the word after (nxt).
# ---------------------------------------------------------------------------
SUBJECT_PRONOUNS = {"i", "we", "you", "he", "she", "they"}
SAYING_VERBS = {"think", "thought", "say", "said", "know", "mean", "guess", "sure", "so", "hope"}
TIME_WORDS = {"week", "morning", "afternoon", "evening", "year", "time", "meeting",
              "moment", "stage", "point", "project"}
VERB_AFTER_THERE = {"is", "are", "was", "were", "be", "will", "might", "could", "isn't", "aren't"}
AGREEMENT = {"right", "it", "fine", "true", "good", "cool", "all", "why", "great", "okay", "ok"}

EXCLUSIONS = [
    # "I think that we...", "so that they..."  -> 'that' starts a clause
    ("that + clause", lambda w, prev, nxt: w == "that" and (nxt in SUBJECT_PRONOUNS or prev in SAYING_VERBS)),
    # "there is a problem", "there's no way"   -> 'there' means 'exists'
    ("there is/are", lambda w, prev, nxt: (w == "there" and nxt in VERB_AFTER_THERE) or w == "there's"),
    # "this week", "at that point"              -> time, not a thing
    ("time phrase", lambda w, prev, nxt: w in {"this", "that"} and nxt in TIME_WORDS),
    # "that's right", "that's it"               -> agreeing, not pointing
    ("agreement", lambda w, prev, nxt: w == "that's" and nxt in AGREEMENT),
]


def clean(text):
    """Lowercase and strip punctuation, keeping apostrophes ("that's")."""
    return re.sub(r"[^\w']", "", text.lower())


def why_excluded(word, prev, nxt):
    """Name of the first exclusion rule that matches, or '' if none does."""
    for name, rule in EXCLUSIONS:
        if rule(word, prev, nxt):
            return name
    return ""


def load_screen(meeting):
    """Timeline rows for this meeting, plus hand notes, to look up what was on screen."""
    with open(TIMELINE_DIR / f"{meeting}_timeline.csv") as f:
        timeline = list(csv.DictReader(f))
    notes = {}
    if NOTES.exists():
        with open(NOTES) as f:
            notes = {(r["meeting"], r["image_file"]): r["content_type"] for r in csv.DictReader(f)}
    return timeline, notes


def screen_at(t, meeting, timeline, notes):
    """(screen_status, slide_id, content_type) at time t."""
    for row in timeline:
        if float(row["start"]) <= t < float(row["end"]):
            if row["status"] == "no capture":
                return "no capture", "", ""
            return "captured", row["slide_id"], notes.get((meeting, row["image_file"]), "slide")
    return "outside transcript", "", ""


def find_candidates(meeting):
    raw = json.loads((config.OUTPUT_DIR / "raw" / f"{meeting}__{MODEL}.json").read_text())
    words = response_to_words(raw)

    # Only search the part of the meeting AMI transcribed (as in Tasks 1 and 2).
    ref, _ = load_reference(meeting)
    first, last = min(w.start for w in ref), max(w.end for w in ref)
    words = [w for w in words if first <= w.start <= last]

    timeline, notes = load_screen(meeting)
    tokens = [clean(w.text) for w in words]
    rows = []
    for i, word in enumerate(tokens):
        if word not in TRIGGERS:
            continue
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        before = " ".join(w.text for w in words[max(0, i - 6):i])
        after = " ".join(w.text for w in words[i + 1:i + 7])
        status, slide_id, content = screen_at(words[i].start, meeting, timeline, notes)
        rows.append({
            "id": f"{meeting}_{len(rows) + 1:04d}",
            "meeting": meeting,
            "time": f"{words[i].start:.2f}",
            "speaker": words[i].speaker,
            "word": word,
            "context": f"{before} [{words[i].text}] {after}",
            "excluded_by": why_excluded(word, prev, nxt),
            "screen_status": status,
            "slide_id": slide_id,
            "content_type": content,
        })
    return rows


def main():
    all_rows = []
    for meeting in config.TUNE_MEETINGS + config.TEST_MEETINGS:
        if not (config.OUTPUT_DIR / "raw" / f"{meeting}__{MODEL}.json").exists():
            print(f"{meeting}: no {MODEL} transcript in task1/outputs/raw, skipping")
            continue
        rows = find_candidates(meeting)
        kept = [r for r in rows if not r["excluded_by"]]
        print(f"{meeting}: {len(rows)} trigger words, {len(kept)} kept")
        all_rows += rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    kept = [r for r in all_rows if not r["excluded_by"]]
    print(f"\nTotal: {len(all_rows)} trigger words, {len(kept)} kept -> {OUT}")
    print("Excluded by rule:", dict(Counter(r["excluded_by"] for r in all_rows if r["excluded_by"])))
    print("Kept, by word:   ", dict(Counter(r["word"] for r in kept).most_common()))
    print("Kept, by screen: ", dict(Counter(r["screen_status"] for r in kept)))


if __name__ == "__main__":
    main()