"""
Task 3: a helper for hand-labelling task3/labels.csv. It does NOT label anything;
it only puts the evidence in front of you and saves what you type.

For each row you haven't labelled yet, it:
  - shows the transcript from 30 s before to 15 s after, with speakers
  - plays the audio from 10 s before to 5 s after the word (needs sox: `play`)
  - opens the screen capture, if there was one (macOS `open`)
  - asks for each field with short choices, and saves after every row

Keys at the category prompt:  r = replay audio,  s = skip row,  q = save and quit

    python task3/label_helper.py
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "task3" / "labels.csv"
AUDIO = ROOT / "data" / "amicorpus" / "{m}" / "audio" / "{m}.Mix-Headset.wav"
SLIDES = ROOT / "data" / "amicorpus" / "{m}" / "slidesBackUp" / "{f}"
TRANSCRIPT = ROOT / "task1" / "outputs" / "transcripts" / "{m}__scribe_v2.csv"

CATEGORIES = ["screen_whole", "screen_item", "physical_object", "whiteboard",
              "room_or_people", "spoken", "no_referent", "unclear"]
CONFIDENCE = {"h": "high", "m": "medium", "l": "low"}
# What extra evidence would have settled this row (feeds the Task 4 argument).
RESOLVE = {"v": "video/gesture", "c": "cursor/pointer", "s": "screen capture",
           "e": "earlier speech", "n": "nothing needed"}


def play(meeting, t):
    """Play from 10 s before to 5 s after time t, in the background."""
    start = max(0.0, t - 10)
    path = str(AUDIO).format(m=meeting)
    return subprocess.Popen(["play", "-q", path, "trim", f"{start:.1f}", "15"],
                            stderr=subprocess.DEVNULL)


def show_context(meeting, t):
    """Print the transcript turns from 30 s before to 15 s after t."""
    with open(str(TRANSCRIPT).format(m=meeting)) as f:
        for r in csv.DictReader(f):
            s = float(r["start"])
            if t - 30 <= s <= t + 15:
                marker = ">>" if s <= t <= float(r["end"]) + 0.5 else "  "
                print(f"  {marker} {s:7.1f}  {r.get('ami_speaker_best_match') or r['speaker']}: {r['text']}")


def ask(prompt, choices=None, default=""):
    """Ask until the answer is one of the choices (or anything, if no choices)."""
    while True:
        ans = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
        ans = ans or default
        if choices is None or ans in choices:
            return ans
        print(f"   please type one of: {', '.join(choices)}")


def save(rows, columns):
    with open(LABELS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    with open(LABELS) as f:
        reader = csv.DictReader(f)
        rows, columns = list(reader), list(reader.fieldnames)
    if "would_resolve" not in columns:
        columns.append("would_resolve")
        for r in rows:
            r["would_resolve"] = ""

    todo = [r for r in rows if not r["category"]]
    print(f"{len(rows) - len(todo)} of {len(rows)} labelled; {len(todo)} to go.\n")

    for n, r in enumerate(todo, 1):
        t, m = float(r["time"]), r["meeting"]
        print("=" * 80)
        print(f"[{n}/{len(todo)}] {r['id']}  {m} at {r['mm:ss']}  word: '{r['word']}'")
        print(f"screen: {r['screen_status']}  {r['content_type']}  {r['slide_id']}  {r['image_file']}")
        print(f"context: {r['context']}\n")
        show_context(m, t)
        if r["image_file"]:
            subprocess.run(["open", str(SLIDES).format(m=m, f=r["image_file"])])
        player = play(m, t)

        print("\n" + "  ".join(f"{i}={c}" for i, c in enumerate(CATEGORIES, 1)))
        while True:
            c = ask("category (r=replay, s=skip, q=quit)", [str(i) for i in range(1, 9)] + ["r", "s", "q"])
            if c == "r":
                player.kill()
                player = play(m, t)
                continue
            break
        player.kill()
        if c == "q":
            break
        if c == "s":
            continue

        r["category"] = CATEGORIES[int(c) - 1]
        r["referent"] = ask("referent (a few words)")
        on_screen = r["category"] in ("screen_whole", "screen_item")
        r["target_capture"] = ask("target_capture (slide_id, blank if none)",
                                  default=r["slide_id"] if on_screen else "")
        r["shown_not_captured"] = "yes" if ask("shown but not captured? (y/n)", ["y", "n"], "n") == "y" else "no"
        r["confidence"] = CONFIDENCE[ask("confidence (h/m/l)", list(CONFIDENCE))]
        print("  ".join(f"{k}={v}" for k, v in RESOLVE.items()))
        r["would_resolve"] = RESOLVE[ask("what would have settled it?", list(RESOLVE), "n")]
        r["notes"] = ask("notes (optional)")
        save(rows, columns)
        print("saved.\n")

    save(rows, columns)
    done = sum(1 for r in rows if r["category"])
    print(f"\n{done} of {len(rows)} labelled. Saved to {LABELS}")


if __name__ == "__main__":
    main()