"""
Task 2, step 4 of 4: link the transcript to what was on screen.

For every row of the Task 1 transcript, attach the screen capture shown while it
was said, plus everything known about it (OCR text, and your hand notes).

Choices (see write-up):
  - Transcript: the ElevenLabs transcript from Task 1 (what a real system would have).
  - A row that spans a screen change gets the screen shown for MOST of the row.
    n_screens says how many screens the row spanned, so Task 3 can check those
    rows at the exact moment of the word instead.
  - screen_status:
        captured           a capture was on screen (see content_type for what kind)
        no capture         nothing captured: we do NOT know what was shown
        outside transcript before/after the part of the meeting AMI transcribed
  - content_type comes from slide_notes.csv (your hand notes). A capture with no
    note is a plain text slide: "slide".

Inputs:  task1/outputs/transcripts/<meeting>__<model>.csv
         task2/outputs/<meeting>_timeline.csv       (build_timeline.py)
         task2/outputs/slide_ocr.csv                (ocr_slides.py)
         task2/slide_notes.csv                      (hand-written)
Output:  task2/outputs/<meeting>_transcript_with_screen.csv

    python task2/join_transcript.py
    python task2/join_transcript.py --model scribe_v3
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

# inspect_slides first: it adds task1/ to the import path.
from inspect_slides import OUT_DIR
import config  # noqa: E402

NOTES = Path(__file__).resolve().parent / "slide_notes.csv"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def screen_for(row_start, row_end, timeline):
    """Pick the timeline interval covering most of [row_start, row_end].
    Returns (interval or None, number of intervals the row touches)."""
    if row_end <= row_start:                         # zero-length row: use its start time
        row_end = row_start + 1e-6
    best, best_overlap, touched = None, 0.0, 0
    for iv in timeline:
        ov = min(row_end, iv["end"]) - max(row_start, iv["start"])
        if ov > 0:
            touched += 1
            if ov > best_overlap:
                best, best_overlap = iv, ov
    return best, touched


def join(meeting, model, ocr_by_id, notes):
    transcript = read_csv(config.OUTPUT_DIR / "transcripts" / f"{meeting}__{model}.csv")
    timeline = read_csv(OUT_DIR / f"{meeting}_timeline.csv")
    for iv in timeline:
        iv["start"], iv["end"] = float(iv["start"]), float(iv["end"])
    t_start, t_end = timeline[0]["start"], timeline[-1]["end"]

    out = []
    for r in transcript:
        s, e = float(r["start"]), float(r["end"])
        row = {"start": r["start"], "end": r["end"], "speaker": r["speaker"],
               "ami_speaker_best_match": r.get("ami_speaker_best_match", ""), "text": r["text"],
               "screen_status": "", "content_type": "", "slide_id": "", "image_file": "",
               "description": "", "same_as": "", "ocr_text": "", "n_screens": 0}

        if e <= t_start or s >= t_end:
            row["screen_status"] = "outside transcript"
            out.append(row)
            continue

        iv, touched = screen_for(s, e, timeline)
        row["n_screens"] = touched
        if iv is None or iv["status"] == "no capture":
            row["screen_status"] = "no capture"
        else:
            note = notes.get((meeting, iv["image_file"]), {})
            row.update({
                "screen_status": "captured",
                "content_type": note.get("content_type", "slide"),
                "slide_id": iv["slide_id"],
                "image_file": iv["image_file"],
                "description": note.get("description", ""),
                "same_as": note.get("same_as", ""),
                "ocr_text": ocr_by_id.get(iv["slide_id"], "").replace("\n", " | "),
            })
        out.append(row)
    return out


def summarise(meeting, rows):
    """Share of transcript rows and of speaking time by what was on screen."""
    time_by, rows_by = defaultdict(float), defaultdict(int)
    for r in rows:
        key = r["screen_status"] if r["screen_status"] != "captured" else f"captured: {r['content_type']}"
        time_by[key] += float(r["end"]) - float(r["start"])
        rows_by[key] += 1
    total_t, total_r = sum(time_by.values()) or 1, len(rows) or 1
    print(f"\n{meeting}: {len(rows)} transcript rows")
    for k in sorted(time_by, key=lambda k: -time_by[k]):
        print(f"  {k:<26} {rows_by[k] / total_r:6.1%} of rows   {time_by[k] / total_t:6.1%} of speaking time")
    multi = sum(1 for r in rows if r["n_screens"] > 1)
    print(f"  rows spanning a screen change: {multi}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="+", default=config.TUNE_MEETINGS + config.TEST_MEETINGS)
    ap.add_argument("--model", default="scribe_v2")
    args = ap.parse_args()

    ocr_by_id = {r["slide_id"]: r["ocr_text"] for r in read_csv(OUT_DIR / "slide_ocr.csv")}
    notes = {(r["meeting"], r["image_file"]): r for r in read_csv(NOTES)} if NOTES.exists() else {}
    if not notes:
        print(f"note: no hand notes found at {NOTES}; every capture will be treated as a text slide")

    for meeting in args.meetings:
        if not (OUT_DIR / f"{meeting}_timeline.csv").exists():
            print(f"{meeting}: no timeline, run build_timeline.py first")
            continue
        rows = join(meeting, args.model, ocr_by_id, notes)
        out = OUT_DIR / f"{meeting}_transcript_with_screen.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summarise(meeting, rows)
        print(f"  -> {out.name}")


if __name__ == "__main__":
    main()