"""
Task 3, step 4: score the baseline against the hand labels.

Baseline: a pointing phrase refers to whatever was on screen when the word was
said (the capture in the row's slide_id; no capture = no answer).

Scoring rule:
  A prediction is correct when the label says the phrase refers to a capture
  (category screen_whole or screen_item) AND the predicted capture is that
  capture (target_capture). Pointing at the right screen counts as correct even
  for screen_item, where the speaker meant one part of it: the baseline only
  works at the level of whole screens, and the report shows how many such cases
  there are.

  "No answer" (no capture on screen) is counted as wrong, because the phrase
  still needed resolving.

Every wrong answer gets one reason, so the report explains WHY it fails, not
just how often.

    python task3/evaluate.py
"""
import csv
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABELS = HERE / "labels.csv"
OUT = HERE / "outputs" / "baseline_results.csv"

SCREEN = {"screen_whole", "screen_item"}
VISIBLE_NOT_SCREEN = {"physical_object", "whiteboard", "room_or_people"}


def predict_baseline(row):
    """Whatever capture was on screen when the word was said, or None."""
    return row["slide_id"] or None


def failure_reason(row, prediction):
    """Why a prediction is wrong (or '' if it is right)."""
    cat = row["category"]
    if cat in SCREEN and prediction == row["target_capture"]:
        return ""
    if cat == "unclear":
        return "label unclear"
    if prediction is None:
        if row["shown_not_captured"] == "yes":
            return "no capture: referent was on screen but not captured"
        if cat in SCREEN:
            return "no capture: referent on a different capture"
        return "no capture: referent not on screen anyway"
    if cat in SCREEN:
        return "wrong capture (e.g. an earlier one)"
    if cat in VISIBLE_NOT_SCREEN:
        return "points at something in the room, not the screen"
    return "not pointing at anything visible (spoken / no referent)"


def wilson(k, n, z=1.96):
    """95% confidence interval for k successes out of n (honest for small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def accuracy_line(name, rows):
    k = sum(1 for r in rows if r["correct"])
    n = len(rows)
    lo, hi = wilson(k, n)
    return f"  {name:<44} {k:3d}/{n:<3d} = {k / n if n else 0:5.0%}   (95% CI {lo:.0%}-{hi:.0%})"


def evaluate(predict, name="baseline", out=OUT, verbose=True):
    """Score any predictor (a function row -> capture id or None) on the labels."""
    with open(LABELS) as f:
        rows = [r for r in csv.DictReader(f) if r["category"]]   # skip anything unlabelled

    for r in rows:
        r["prediction"] = predict(r) or ""
        r["failure_reason"] = failure_reason(r, r["prediction"] or None)
        r["correct"] = r["failure_reason"] == ""

    if verbose:
        print(f"\n=== {name}: {len(rows)} labelled examples ===")
        print(accuracy_line("all examples", rows))
        print(accuracy_line("excluding low-confidence labels", [r for r in rows if r["confidence"] != "low"]))
        print(accuracy_line("when something was captured", [r for r in rows if r["screen_status"] == "captured"]))
        print(accuracy_line("when nothing was captured", [r for r in rows if r["screen_status"] != "captured"]))
        print(accuracy_line("when the label says 'on a screen'", [r for r in rows if r["category"] in SCREEN]))

        print("\nWhat the phrases referred to (your labels):")
        for cat, n in Counter(r["category"] for r in rows).most_common():
            right = sum(1 for r in rows if r["category"] == cat and r["correct"])
            print(f"  {cat:<18} {n:3d}   baseline right on {right}")

        print("\nWhy it was wrong:")
        for reason, n in Counter(r["failure_reason"] for r in rows if not r["correct"]).most_common():
            print(f"  {n:3d}  {reason}")

        print("\nWhat would have settled uncertain labels (medium/low confidence):")
        unsure = [r for r in rows if r["confidence"] in ("medium", "low")]
        for what, n in Counter(r.get("would_resolve", "") or "(blank)" for r in unsure).most_common():
            print(f"  {n:3d}  {what}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["id", "meeting", "time", "word", "context", "screen_status", "category",
            "target_capture", "confidence", "prediction", "correct", "failure_reason"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    if verbose:
        print(f"\nPer-example results -> {out}")
    return rows


if __name__ == "__main__":
    evaluate(predict_baseline)