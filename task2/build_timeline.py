"""
Task 2, step 2: turn the slide files into a clean timeline.

What it does, per meeting:
  1. Merge consecutive captures that are the same image into one interval
     (about half the files are repeats, so a new file is not a new slide).
  2. Give each distinct slide an id. If a slide comes back later in the
     meeting, it gets its earlier id again, so "the one we saw before" can be
     traced in Task 3.
  3. Clip to the part of the meeting AMI transcribed.
  4. Fill every moment that has no slide with an explicit "no capture" row.
     That means "we don't know what was on screen", not "nothing was shown"
     (ES2002b shows a whole presentation was never captured).

Outputs (task2/outputs/):
  <meeting>_timeline.csv   start, end, status, slide_id, image_file, n_captures
  slides.csv               one row per distinct slide: slide_id, meeting, image_file
                           (the input for the OCR / description step)

    python task2/build_timeline.py
"""
import argparse
import csv

import numpy as np

# inspect_slides must be imported first: it adds task1/ to the import path.
from inspect_slides import OUT_DIR, PIXEL_CHANGE, SAME_IMAGE_SHARE, load_slides, slide_dir, thumbnail
import config                          # noqa: E402  (task1's config)
from ami_reference import load_reference  # noqa: E402


def same_image(a, b) -> bool:
    return np.mean(np.abs(a - b) > PIXEL_CHANGE) < SAME_IMAGE_SHARE


def build(meeting):
    slides, _ = load_slides(meeting)
    if not slides:
        return [], []
    ref, _ = load_reference(meeting)
    m_start, m_end = min(w.start for w in ref), max(w.end for w in ref)

    # 1 + 2: merge consecutive repeats; reuse ids for slides shown again later.
    distinct = []          # [(slide_id, thumb, first image file)]
    intervals = []         # [dict] one per run of identical captures
    for start, end, path in slides:
        thumb = thumbnail(path)
        if intervals and same_image(thumb, intervals[-1]["thumb"]):
            intervals[-1]["end"] = end
            intervals[-1]["n_captures"] += 1
            continue
        match = next((sid for sid, t, _ in distinct if same_image(thumb, t)), None)
        if match is None:
            match = f"{meeting}_s{len(distinct) + 1:02d}"
            distinct.append((match, thumb, path.name))
        intervals.append({"start": start, "end": end, "slide_id": match,
                          "image_file": path.name, "n_captures": 1, "thumb": thumb})

    # 3: clip to the transcribed region.
    rows = []
    for iv in intervals:
        s, e = max(iv["start"], m_start), min(iv["end"], m_end)
        if e > s:
            rows.append({"start": s, "end": e, "status": "slide", "slide_id": iv["slide_id"],
                         "image_file": iv["image_file"], "n_captures": iv["n_captures"]})

    # 4: fill the gaps with explicit "no capture" rows.
    timeline, t = [], m_start
    for r in rows:
        if r["start"] > t + 0.01:
            timeline.append({"start": t, "end": r["start"], "status": "no capture",
                             "slide_id": "", "image_file": "", "n_captures": 0})
        timeline.append(r)
        t = max(t, r["end"])
    if m_end > t + 0.01:
        timeline.append({"start": t, "end": m_end, "status": "no capture",
                         "slide_id": "", "image_file": "", "n_captures": 0})

    slide_rows = [{"slide_id": sid, "meeting": meeting, "image_file": f} for sid, _, f in distinct]
    return timeline, slide_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="+", default=config.TUNE_MEETINGS + config.TEST_MEETINGS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_slides = []
    for meeting in args.meetings:
        if not slide_dir(meeting).exists():
            print(f"{meeting}: no slides folder, skipping")
            continue
        timeline, slide_rows = build(meeting)
        all_slides += slide_rows

        out = OUT_DIR / f"{meeting}_timeline.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["start", "end", "status", "slide_id", "image_file", "n_captures"])
            w.writeheader()
            for r in timeline:
                w.writerow({**r, "start": f"{r['start']:.2f}", "end": f"{r['end']:.2f}"})

        shown = [r for r in timeline if r["status"] == "slide"]
        span = timeline[-1]["end"] - timeline[0]["start"] if timeline else 0
        covered = sum(r["end"] - r["start"] for r in shown)
        revisits = len(shown) - len({r["slide_id"] for r in shown})
        print(f"{meeting}: {sum(r['n_captures'] for r in shown)} captures -> {len(shown)} intervals, "
              f"{len(slide_rows)} distinct slides, {revisits} revisits; "
              f"slide on screen {covered / span:.0%} of transcribed time -> {out.name}")

    with open(OUT_DIR / "slides.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide_id", "meeting", "image_file"])
        w.writeheader()
        w.writerows(all_slides)
    print(f"{len(all_slides)} distinct slides -> slides.csv")


if __name__ == "__main__":
    main()