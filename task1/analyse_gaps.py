"""
Choose SEGMENT_MAX_GAP from the data, not from the score.

For every pair of consecutive words from the same speaker in the AMI reference,
measure the pause between them, and sort each pause into one of two kinds:

    "pause"      nobody else started a word during it: the same person just paused
    "turn break" someone else started a word during it: the floor passed to them

A good gap threshold merges most pauses and splits most turn breaks. The two
distributions overlap (someone saying "yeah" mid-sentence counts as a turn
break), so there is no perfect value. The script shows you where the trade-off is.

    python analyse_gaps.py                       # tuning meeting(s) from config
    python analyse_gaps.py --meetings ES2002a ES2002b ES2002c ES2002d
"""
import argparse
import bisect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from ami_reference import load_reference

CANDIDATES = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def classify_gaps(words):
    """Return (pauses, turn_breaks): two arrays of gap lengths in seconds."""
    # Skip zero-length words (e.g. ones with missing times): their position is a guess.
    words = [w for w in words if w.end > w.start]
    speakers = sorted({w.speaker for w in words})
    starts_by_spk = {s: sorted(w.start for w in words if w.speaker == s) for s in speakers}

    pauses, breaks = [], []
    for spk in speakers:
        own = sorted((w for w in words if w.speaker == spk), key=lambda w: w.start)
        others = sorted(t for s in speakers if s != spk for t in starts_by_spk[s])
        for prev, nxt in zip(own, own[1:]):
            gap = nxt.start - prev.end
            if gap <= 0:
                continue                      # touching or overlapping words: no pause
            # Did anyone else start a word strictly inside the gap?
            i = bisect.bisect_right(others, prev.end)
            interrupted = i < len(others) and others[i] < nxt.start
            (breaks if interrupted else pauses).append(gap)
    return np.array(pauses), np.array(breaks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="+", default=config.TUNE_MEETINGS)
    args = ap.parse_args()

    pauses, breaks = [], []
    for m in args.meetings:
        words, _ = load_reference(m)
        p, b = classify_gaps(words)
        pauses.append(p)
        breaks.append(b)
        print(f"{m}: {len(p)} pauses, {len(b)} turn breaks")
    pauses, breaks = np.concatenate(pauses), np.concatenate(breaks)

    print("\nPercentiles (seconds)      10%    25%    50%    75%    90%")
    for name, arr in [("pauses", pauses), ("turn breaks", breaks)]:
        q = np.percentile(arr, [10, 25, 50, 75, 90])
        print(f"  {name:<22}" + "".join(f"{v:7.2f}" for v in q))

    print("\nIf SEGMENT_MAX_GAP = t:")
    print("     t   pauses merged   turn breaks split   (higher is better for both)")
    for t in CANDIDATES:
        print(f"  {t:4.2f}   {np.mean(pauses <= t):12.0%}   {np.mean(breaks > t):17.0%}")

    # Plot: both distributions on a log scale (gaps run from milliseconds to minutes).
    lo = max(0.01, min(pauses.min(), breaks.min()))
    hi = max(pauses.max(), breaks.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(pauses, bins=bins, histtype="step", linewidth=2, color="#257de0", label="pause (same speaker continues)")
    ax.hist(breaks, bins=bins, histtype="step", linewidth=2, color="#f47230", label="turn break (someone else spoke)")
    ax.axvline(config.SEGMENT_MAX_GAP, color="#555555", linestyle="--", linewidth=1)
    ax.text(config.SEGMENT_MAX_GAP, ax.get_ylim()[1] * 0.95, f"  current setting {config.SEGMENT_MAX_GAP} s",
            color="#555555", va="top", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Gap between consecutive words from the same speaker (seconds, log scale)")
    ax.set_ylabel("Number of gaps")
    ax.set_title(f"Within-speaker gaps in the AMI reference ({', '.join(args.meetings)})", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = config.OUTPUT_DIR / "gap_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
