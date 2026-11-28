"""
Task 3, step 3 (preparation): draw the sample to hand-label and create the sheet.

Sampling choices (see write-up):
  - Only candidates the detection rule kept.
  - One example per phrase: "this one here" gives two hits, so a hit from the same
    speaker within PHRASE_GAP seconds of the previous kept hit is skipped.
  - Mostly this/these/those/here/there, fewer "that": the precision check found
    only ~9% of "that" hits point at anything visual, against 35-71% for the others.
  - Spread evenly across meetings and screen states (captured / no capture),
    so failures can be compared between them.
  - Fixed random seed, so the same sample comes out every time.

Output: task3/labels.csv  (columns to fill in are at the end, see labelling_scheme.md)
It will NOT overwrite an existing labels.csv, to protect your labels.

    python task3/make_label_sheet.py
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "outputs" / "candidates.csv"
SLIDES = HERE.parent / "task2" / "outputs" / "slides.csv"
OUT = HERE / "labels.csv"

N_STRONG = 40        # from this / these / those / here / there
N_THAT = 10          # from that / that's
PHRASE_GAP = 1.5     # seconds: closer hits from one speaker count as the same phrase
SEED = 1

STRONG_WORDS = {"this", "these", "those", "here", "there", "here's"}
THAT_WORDS = {"that", "that's"}
LABEL_COLUMNS = ["category", "referent", "target_capture", "shown_not_captured", "confidence", "notes"]


def one_per_phrase(rows):
    """Drop a hit if the same speaker had a kept hit less than PHRASE_GAP seconds earlier."""
    rows = sorted(rows, key=lambda r: (r["meeting"], float(r["time"])))
    kept, last = [], {}
    for r in rows:
        key = (r["meeting"], r["speaker"])
        t = float(r["time"])
        if key in last and t - last[key] < PHRASE_GAP:
            continue
        kept.append(r)
        last[key] = t
    return kept


def stratified_sample(rows, n, rng):
    """Take n rows, spread evenly over (meeting, screen_status) groups: round-robin
    through the groups, one random row at a time, until n are picked."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["meeting"], r["screen_status"])].append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    picked = []
    while len(picked) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


def mmss(seconds):
    """1552.3 -> '25:52', handy for jumping to the right place in the audio."""
    s = int(float(seconds))
    return f"{s // 60}:{s % 60:02d}"


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; not overwriting your labels. Rename it to make a new sample.")

    with open(CANDIDATES) as f:
        rows = [r for r in csv.DictReader(f) if not r["excluded_by"]]
    rows = one_per_phrase(rows)

    image_of = {}
    if SLIDES.exists():
        with open(SLIDES) as f:
            image_of = {r["slide_id"]: r["image_file"] for r in csv.DictReader(f)}

    rng = random.Random(SEED)
    sample = (stratified_sample([r for r in rows if r["word"] in STRONG_WORDS], N_STRONG, rng)
              + stratified_sample([r for r in rows if r["word"] in THAT_WORDS], N_THAT, rng))
    sample.sort(key=lambda r: (r["meeting"], float(r["time"])))

    columns = ["id", "meeting", "time", "mm:ss", "speaker", "word", "context",
               "screen_status", "content_type", "slide_id", "image_file"] + LABEL_COLUMNS
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in sample:
            w.writerow({"id": r["id"], "meeting": r["meeting"], "time": r["time"], "mm:ss": mmss(r["time"]),
                        "speaker": r["speaker"], "word": r["word"], "context": r["context"],
                        "screen_status": r["screen_status"], "content_type": r["content_type"],
                        "slide_id": r["slide_id"], "image_file": image_of.get(r["slide_id"], ""),
                        **{c: "" for c in LABEL_COLUMNS}})

    print(f"{len(rows)} phrases after one-per-phrase -> sampled {len(sample)} -> {OUT}")
    print("by meeting:", dict(Counter(r["meeting"] for r in sample)))
    print("by screen: ", dict(Counter(r["screen_status"] for r in sample)))
    print("by word:   ", dict(Counter(r["word"] for r in sample)))


if __name__ == "__main__":
    main()