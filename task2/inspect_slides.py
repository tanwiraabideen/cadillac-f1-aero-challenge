"""
Task 2, step 1: how far can the slide timeline from the file names be trusted?

Each file in slidesBackUp/ is named <start>__<end>.jpg (seconds). This script
builds the timeline from those names and checks it, rather than assuming it is
right. For each meeting it reports:

  - files whose names don't parse
  - how much of the meeting has a slide on screen, and the gaps where nothing does
  - overlaps (two slides claiming the same moment) and zero/negative durations
  - very short slides (someone flicking through, or capture glitches)
  - consecutive captures that look identical (a new file without a new slide)
  - "next slide"-type phrases in the AMI reference, and how far each is from the
    nearest slide change: an independent check on the timings

and saves a timeline plot per meeting.

    python task2/inspect_slides.py
    python task2/inspect_slides.py --meetings ES2002b
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Reuse Task 1's paths and AMI reader.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task1"))
import config                               # noqa: E402
from ami_reference import load_reference    # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
NAME = re.compile(r"^(\d+(?:\.\d+)?)__(\d+(?:\.\d+)?)\.(jpe?g|png)$", re.I)

SHORT_SLIDE_S = 2.0          # shown for less than this: flag it
# Two captures count as the same slide if fewer than this share of pixels changed
# clearly (by more than PIXEL_CHANGE grey levels). Slides share a white template,
# so an average difference would call most slides "the same"; counting changed
# pixels catches a change in text. Calibrate by eye: the script prints the values.
PIXEL_CHANGE = 40
SAME_IMAGE_SHARE = 0.001
# Phrases that suggest a slide change is happening. Matched on the reference text.
CHANGE_PHRASES = ["next slide", "next one", "previous slide", "go back", "the last slide",
                  "move on", "this slide", "on the slide"]


def slide_dir(meeting):
    return config.AUDIO_ROOT / meeting / "slidesBackUp"


def load_slides(meeting):
    """[(start, end, path)] sorted by start, plus names that didn't parse."""
    slides, bad = [], []
    for p in sorted(slide_dir(meeting).iterdir()):
        if p.name.startswith("."):
            continue
        m = NAME.match(p.name)
        if m:
            slides.append((float(m.group(1)), float(m.group(2)), p))
        else:
            bad.append(p.name)
    slides.sort(key=lambda s: (s[0], s[1]))
    return slides, bad


def thumbnail(path):
    """Greyscale at a fixed size, so captures of different sizes can be compared."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((320, 240)), dtype=float)


def phrase_times(ref_words):
    """Start time of each change-phrase occurrence in the reference."""
    toks = [(w.text.lower().strip(".,?!"), w.start) for w in ref_words]
    hits = []
    for phrase in CHANGE_PHRASES:
        parts = phrase.split()
        for i in range(len(toks) - len(parts) + 1):
            if all(toks[i + j][0] == parts[j] for j in range(len(parts))):
                hits.append((toks[i][1], phrase))
    return sorted(hits)


def inspect(meeting):
    print(f"\n=== {meeting} ===")
    if not slide_dir(meeting).exists():
        print(f"  no folder {slide_dir(meeting)}, skipping")
        return None
    slides, bad = load_slides(meeting)
    print(f"  {len(slides)} slide files" + (f", {len(bad)} unparseable: {bad[:5]}" if bad else ""))
    if not slides:
        return None

    ref, _ = load_reference(meeting)
    m_start, m_end = min(w.start for w in ref), max(w.end for w in ref)
    span = m_end - m_start

    durs = np.array([e - s for s, e, _ in slides])
    print(f"  first slide starts {slides[0][0]:.1f} s, last ends {slides[-1][1]:.1f} s "
          f"(transcript covers {m_start:.1f}-{m_end:.1f} s)")
    print(f"  durations: min {durs.min():.1f} s, median {np.median(durs):.1f} s, max {durs.max():.1f} s")

    bad_dur = [(s, e) for s, e, _ in slides if e <= s]
    short = [(s, e) for s, e, _ in slides if 0 < e - s < SHORT_SLIDE_S]
    print(f"  zero/negative durations: {len(bad_dur)} {bad_dur[:5]}")
    print(f"  shorter than {SHORT_SLIDE_S} s: {len(short)} {[(round(s,1), round(e,1)) for s, e in short[:5]]}")

    # Gaps and overlaps between consecutive slides
    gaps, overlaps = [], []
    for (s1, e1, _), (s2, e2, _) in zip(slides, slides[1:]):
        if s2 > e1 + 0.01:
            gaps.append((e1, s2))
        elif s2 < e1 - 0.01:
            overlaps.append((s2, min(e1, e2)))
    print(f"  gaps between slides: {len(gaps)}, total {sum(b - a for a, b in gaps):.1f} s; "
          f"largest {[(round(a,1), round(b,1)) for a, b in sorted(gaps, key=lambda g: g[0]-g[1])[:3]]}")
    print(f"  overlapping slides: {len(overlaps)} {[(round(a,1), round(b,1)) for a, b in overlaps[:5]]}")

    # Coverage of the transcribed part of the meeting
    covered = 0.0
    for s, e, _ in slides:
        covered += max(0.0, min(e, m_end) - max(s, m_start))
    print(f"  slide on screen for {covered:.0f} s of {span:.0f} s transcribed ({covered / span:.0%})"
          + ("  [overlaps double-count]" if overlaps else ""))

    # Consecutive captures that look the same
    pairs = []
    thumbs = [thumbnail(p) for _, _, p in slides]
    for i in range(1, len(slides)):
        changed = np.mean(np.abs(thumbs[i] - thumbs[i - 1]) > PIXEL_CHANGE)
        pairs.append((changed, slides[i - 1][2].name, slides[i][2].name))
    same = [p for p in pairs if p[0] < SAME_IMAGE_SHARE]
    print(f"  consecutive near-identical captures: {len(same)} "
          f"(threshold: <{SAME_IMAGE_SHARE:.1%} of pixels changed)")
    print("    most similar consecutive pairs (open these to check the threshold):")
    for changed, a, b in sorted(pairs)[:5]:
        print(f"      {changed:6.2%} changed   {a}  ->  {b}")

    # Independent check: do "next slide" phrases happen near slide changes?
    changes = np.array(sorted({s for s, _, _ in slides} | {e for _, e, _ in slides}))
    hits = phrase_times(ref)
    print(f"  change-phrases in transcript: {len(hits)}")
    dists = []
    for t, phrase in hits:
        d = t - changes[np.argmin(np.abs(changes - t))]     # + means phrase after the change
        dists.append(d)
        print(f"    {t:7.1f} s  '{phrase}'  nearest slide change {d:+.1f} s")
    if dists:
        print(f"  median |distance| to nearest change: {np.median(np.abs(dists)):.1f} s")

    plot_timeline(meeting, slides, ref, hits, m_start, m_end)
    return {"meeting": meeting, "slides": len(slides), "gaps": len(gaps), "overlaps": len(overlaps),
            "short": len(short), "same": len(same), "coverage": covered / span}


def plot_timeline(meeting, slides, ref, hits, m_start, m_end):
    fig, ax = plt.subplots(figsize=(12, 2.6))
    # Slides: alternating greys so adjacent slides are distinguishable
    for i, (s, e, _) in enumerate(slides):
        ax.barh(1, e - s, left=s, height=0.6, color="#9a9a9a" if i % 2 else "#cfcfcf",
                edgecolor="white", linewidth=1)
    # Speech activity from the reference
    for w in ref:
        if w.end > w.start:
            ax.barh(0, w.end - w.start, left=w.start, height=0.6, color="#257de0", linewidth=0)
    # Change phrases
    for t, _ in hits:
        ax.axvline(t, color="#f47230", linewidth=1.2, alpha=0.9)
    ax.axvspan(0, m_start, color="#f2f2f2", zorder=-1)
    ax.axvspan(m_end, max(m_end, slides[-1][1]) + 10, color="#f2f2f2", zorder=-1)
    ax.set_yticks([0, 1], ["speech (AMI)", "slides"])
    ax.set_xlabel("time (s)")
    ax.set_title(f"{meeting}: slide timeline from file names. Orange lines: 'next slide'-type phrases; "
                 f"shaded: outside the transcribed region", loc="left", fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{meeting}_slide_timeline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="+", default=config.TUNE_MEETINGS + config.TEST_MEETINGS)
    args = ap.parse_args()
    rows = [r for m in args.meetings if (r := inspect(m))]
    if rows:
        print("\nSummary")
        print(f"  {'meeting':<9}{'slides':>7}{'gaps':>6}{'overlap':>8}{'short':>7}{'same':>6}{'coverage':>10}")
        for r in rows:
            print(f"  {r['meeting']:<9}{r['slides']:>7}{r['gaps']:>6}{r['overlaps']:>8}"
                  f"{r['short']:>7}{r['same']:>6}{r['coverage']:>10.0%}")


if __name__ == "__main__":
    main()