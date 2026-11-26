"""
The two metrics.

WER (word error rate): align the reference and hypothesis word sequences with
the fewest edits, then
    WER = (substitutions + deletions + insertions) / number of reference words
Speaker labels are ignored here; it only asks "were the right words written?".

DER (diarisation error rate): ignore the words, look only at "who was speaking
when". Turn each transcript into speaker segments, then measure the fraction of
reference speaking time that is
    missed       (someone spoke, hypothesis says nobody did)
    false alarm  (hypothesis says someone spoke, nobody did)
    confusion    (right time, wrong speaker)
Speaker names don't have to match ("speaker_2" vs "C"): pyannote finds the
best one-to-one mapping between the two sets of labels before scoring.
"""
from collections import Counter

import jiwer
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

import config
from words import Word


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------
def compute_wer(ref: list[Word], hyp: list[Word], top_n: int = 30) -> dict:
    """Both inputs must already be normalised. Returns the score, its parts,
    and the most common errors (look at these before changing any settings)."""
    ref_tokens = [w.text for w in ref]
    hyp_tokens = [w.text for w in hyp]
    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))

    subs, dels, ins = Counter(), Counter(), Counter()
    for chunk in out.alignments[0]:
        r = ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx]
        h = hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type == "substitute":
            for a, b in zip(r, h):
                subs[(a, b)] += 1
        elif chunk.type == "delete":
            dels.update(r)
        elif chunk.type == "insert":
            ins.update(h)

    n_ref = len(ref_tokens)
    return {
        "ref_words": n_ref,
        "hyp_words": len(hyp_tokens),
        "wer": out.wer,
        "sub_rate": out.substitutions / n_ref,
        "del_rate": out.deletions / n_ref,
        "ins_rate": out.insertions / n_ref,
        "top_substitutions": subs.most_common(top_n),
        "top_deletions": dels.most_common(top_n),
        "top_insertions": ins.most_common(top_n),
    }


# ---------------------------------------------------------------------------
# DER
# ---------------------------------------------------------------------------
def words_to_segments(words: list[Word], max_gap: float = config.DER_SEGMENT_MAX_GAP) -> list[tuple[str, float, float]]:
    """Merge each speaker's consecutive words into (speaker, start, end) segments,
    starting a new segment when the pause is longer than max_gap seconds."""
    by_speaker: dict[str, list[Word]] = {}
    for w in words:
        by_speaker.setdefault(w.speaker, []).append(w)

    segments = []
    for spk, ws in by_speaker.items():
        ws = sorted(ws, key=lambda w: w.start)
        seg_start, seg_end = ws[0].start, ws[0].end
        for w in ws[1:]:
            if w.start - seg_end <= max_gap:
                seg_end = max(seg_end, w.end)
            else:
                segments.append((spk, seg_start, seg_end))
                seg_start, seg_end = w.start, w.end
        segments.append((spk, seg_start, seg_end))
    return sorted(segments, key=lambda s: s[1])


def to_annotation(segments, uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    for i, (spk, start, end) in enumerate(segments):
        if end > start:                      # zero-length segments carry no time
            ann[Segment(start, end), i] = spk
    return ann


def compute_der(ref: list[Word], hyp: list[Word], uri: str, region=None) -> dict:
    """Diarisation error rate plus its three components (as fractions of
    reference speech time) and the speaker mapping pyannote chose."""
    ref_ann = to_annotation(words_to_segments(ref), uri)
    hyp_ann = to_annotation(words_to_segments(hyp), uri)


    metric = DiarizationErrorRate(
        collar=2 * config.DER_COLLAR_EACH_SIDE,     # pyannote wants the total width
        skip_overlap=config.DER_SKIP_OVERLAP,
    )
    uem = Timeline([Segment(*region)]) if region else None
    d = metric(ref_ann, hyp_ann, uem=uem, detailed=True)
    total = d["total"]
    return {
        "der": d["diarization error rate"],
        "missed": d["missed detection"] / total,
        "false_alarm": d["false alarm"] / total,
        "confusion": d["confusion"] / total,
        "scored_speech_seconds": total,
        "n_hyp_speakers": len(hyp_ann.labels()),
        # e.g. {"speaker_0": "B", ...}. Uses the reference, so it is for
        # analysis only: a real system would not have it.
        "speaker_mapping": metric.optimal_mapping(ref_ann, hyp_ann),
    }
