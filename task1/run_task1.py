"""
Task 1 pipeline: transcribe -> normalise -> score -> write results.

    python run_task1.py                    # all meetings, all models
    python run_task1.py --meetings ES2002a --models scribe_v2
    python run_task1.py --no-transcribe    # score cached responses only (free)

Outputs (in outputs/):
    raw/<meeting>__<model>.json         ElevenLabs response, exactly as returned
    transcripts/<meeting>__<model>.csv  readable timestamped transcript (the deliverable)
    errors/<meeting>__<model>.txt       most common substitutions / deletions / insertions
    results.csv, results.md             the results table
"""
import argparse
import csv
import json
import wave

import config
from ami_reference import load_reference
from elevenlabs_client import raw_path, response_to_words, transcribe
from normalise import normalise_words
from scoring import compute_der, compute_wer
from words import Word


def audio_path(meeting):
    return config.AUDIO_ROOT / meeting / "audio" / f"{meeting}.{config.AUDIO_CHANNEL}.wav"

def audio_seconds(path) -> float:
    with wave.open(str(path)) as f:
        return f.getnframes() / f.getframerate()


def write_transcript(words: list[Word], mapping: dict, path):
    """One row per turn: consecutive words from the same speaker with no long pause."""
    rows, cur = [], None
    for w in words:
        if cur and w.speaker == cur["speaker"] and w.start - cur["end"] <= config.TRANSCRIPT_TURN_GAP:
            cur["end"] = w.end
            cur["text"].append(w.text)
        else:
            if cur:
                rows.append(cur)
            cur = {"start": w.start, "end": w.end, "speaker": w.speaker, "text": [w.text]}
    if cur:
        rows.append(cur)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["start", "end", "speaker", "ami_speaker_best_match", "text"])
        for r in rows:
            out.writerow([f"{r['start']:.2f}", f"{r['end']:.2f}", r["speaker"],
                          mapping.get(r["speaker"], ""), " ".join(r["text"])])


def write_errors(wer: dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Top substitutions (reference -> hypothesis):"]
    lines += [f"  {n:4d}  {a} -> {b}" for (a, b), n in wer["top_substitutions"]]
    lines += ["", "Top deletions (in reference, missing from hypothesis):"]
    lines += [f"  {n:4d}  {w}" for w, n in wer["top_deletions"]]
    lines += ["", "Top insertions (in hypothesis, not in reference):"]
    lines += [f"  {n:4d}  {w}" for w, n in wer["top_insertions"]]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="+", default=config.TUNE_MEETINGS + config.TEST_MEETINGS)
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--no-transcribe", action="store_true", help="only score cached responses")
    ap.add_argument("--force", action="store_true", help="re-call the API even if cached")
    args = ap.parse_args()

    results = []
    for meeting in args.meetings:
        print(f"\n=== {meeting} ===")
        ref_raw, ref_stats = load_reference(meeting)
        ref = normalise_words(ref_raw)
        print(f"  reference: {len(ref_raw)} words -> {len(ref)} tokens after normalisation")
        print(f"  per-speaker parse counts: {json.dumps(ref_stats)}")

        wav = audio_path(meeting)
        if wav.exists():
            secs = audio_seconds(wav)
            last = max(w.end for w in ref_raw)
            print(f"  audio {secs:.1f} s, last reference word ends {last:.1f} s")
            if last > secs + 1:
                print("  WARNING: reference runs past the end of the audio. Check the files match.")

        for model in args.models:
            if args.no_transcribe:
                if not raw_path(meeting, model).exists():
                    print(f"  [{model}] no cached response, skipping")
                    continue
                response = json.loads(raw_path(meeting, model).read_text())
            else:
                response = transcribe(meeting, model, wav, force=args.force)

            hyp_raw = response_to_words(response)
            hyp = normalise_words(hyp_raw)

            # Score only where AMI has a transcript. Outside that region (before the
            # meeting starts / after it ends) there is no answer key, so any words
            # ElevenLabs finds there can't be judged. Words are kept if their midpoint
            # falls inside the region.
            region = (min(w.start for w in ref_raw), max(w.end for w in ref_raw))
            hyp_in = [w for w in hyp if region[0] <= (w.start + w.end) / 2 <= region[1]]

            wer = compute_wer(ref, hyp_in)
            der = compute_der(ref, hyp_in, uri=meeting, region=region)
            wer_full = compute_wer(ref, hyp)               # whole file, for comparison
            der_full = compute_der(ref, hyp, uri=meeting)

            tag = f"{meeting}__{model}"
            write_transcript(hyp_raw, der["speaker_mapping"], config.OUTPUT_DIR / "transcripts" / f"{tag}.csv")
            write_errors(wer, config.OUTPUT_DIR / "errors" / f"{tag}.txt")

            row = {
                "meeting": meeting,
                "split": "tune" if meeting in config.TUNE_MEETINGS else "test",
                "model": model,
                "ref_words": wer["ref_words"],
                "WER": wer["wer"], "sub": wer["sub_rate"], "del": wer["del_rate"], "ins": wer["ins_rate"],
                "DER": der["der"], "missed": der["missed"], "false_alarm": der["false_alarm"],
                "confusion": der["confusion"], "hyp_speakers": der["n_hyp_speakers"],
                "speech_s": round(der["scored_speech_seconds"], 1),
                "WER_full": wer_full["wer"], "DER_full": der_full["der"],
            }
            results.append(row)
            print(f"  [{model}] WER {row['WER']:.1%}  DER {row['DER']:.1%}  "
                  f"(speakers found: {row['hyp_speakers']})")

    if results:
        write_results(results)


def write_results(results):
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    keys = list(results[0])
    with open(config.OUTPUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)

    pct = {"WER", "sub", "del", "ins", "DER", "missed", "false_alarm", "confusion"}
    fmt = lambda k, v: f"{v:.1%}" if k in pct else str(v)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(k, r[k]) for k in keys) + " |" for r in results]

    # Test-set summary per model. WER is weighted by reference words and DER by
    # speech time, so a long meeting counts more than a short one (the same as
    # pooling all test meetings together).
    lines += ["", "Test meetings pooled (WER weighted by words, DER by speech time):", "", "| model | WER | DER |", "|---|---|---|"]
    for model in dict.fromkeys(r["model"] for r in results):
        test = [r for r in results if r["model"] == model and r["split"] == "test"]
        if test:
            n = sum(r["ref_words"] for r in test)
            wer = sum(r["WER"] * r["ref_words"] for r in test) / n
            t = sum(r["speech_s"] for r in test)
            der = sum(r["DER"] * r["speech_s"] for r in test) / t
            lines.append(f"| {model} | {wer:.1%} | {der:.1%} |")

    (config.OUTPUT_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {config.OUTPUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
