"""
Read the AMI reference transcript (NXT words XML) into a list of Words.

AMI stores one file per speaker, e.g. ES2002a.A.words.xml ... ES2002a.D.words.xml.
Each spoken word is a <w> element with starttime / endtime attributes:

    <w nite:id="ES2002a.A.words5" starttime="53.56" endtime="53.96">Okay</w>
    <w nite:id="ES2002a.A.words6" starttime="53.96" endtime="53.96" punc="true">.</w>
    <vocalsound nite:id="..." starttime="..." endtime="..." type="laugh"/>

We keep only <w> elements that are real words (not punctuation), then merge the
speakers and sort everything by start time.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import config
from words import Word


def _local_name(tag: str) -> str:
    """'{namespace}w' -> 'w' (AMI files use the nite namespace on the root)."""
    return tag.rsplit("}", 1)[-1]


def read_speaker_file(path: Path, speaker: str) -> tuple[list[Word], dict]:
    """Parse one speaker's words file. Returns the words and some counts
    (useful evidence for the write-up: how much did we drop, and why)."""
    stats = {"w_elements": 0, "punctuation": 0, "truncated": 0,
             "truncated_dropped": 0, "missing_times": 0, "other_elements": 0}
    words: list[Word] = []
    last_time = 0.0

    for el in ET.parse(path).getroot():
        if _local_name(el.tag) != "w":
            stats["other_elements"] += 1          # vocalsound, disfmarker, gap, ...
            continue
        stats["w_elements"] += 1

        if el.get("punc") == "true":
            stats["punctuation"] += 1
            continue
        if el.get("trunc") == "true":
            stats["truncated"] += 1
            if config.DROP_TRUNCATED_WORDS:
                stats["truncated_dropped"] += 1
                continue

        text = (el.text or "").strip()
        if not text:
            continue

        start, end = el.get("starttime"), el.get("endtime")
        if start is None or end is None:
            # A few AMI words have no timing. We keep them for WER (they were
            # spoken) and give them the previous word's time so ordering still
            # works. They are zero-length, so they barely affect DER.
            stats["missing_times"] += 1
            start = end = last_time
        start, end = float(start), float(end)
        last_time = end
        words.append(Word(speaker=speaker, start=start, end=end, text=text))

    return words, stats


def load_reference(meeting: str, words_dir: Path = config.WORDS_DIR) -> tuple[list[Word], dict]:
    """All speakers for one meeting, merged and sorted by time."""
    files = sorted(words_dir.glob(f"{meeting}.*.words.xml"))
    if not files:
        raise FileNotFoundError(f"No words files for {meeting} in {words_dir}")

    all_words: list[Word] = []
    stats_by_speaker = {}
    for f in files:
        speaker = f.name.split(".")[1]            # "ES2002a.A.words.xml" -> "A"
        words, stats = read_speaker_file(f, speaker)
        all_words.extend(words)
        stats_by_speaker[speaker] = stats

    # Sort by start time. Where two people overlap, their words interleave by
    # start time, which is one (imperfect) choice; see the write-up note on
    # overlapping speech.
    all_words.sort(key=lambda w: (w.start, w.end))
    return all_words, stats_by_speaker
