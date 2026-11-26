# Task 1: baseline transcript (ElevenLabs Scribe on AMI ES2002)

Transcribes the AMI ES2002 meetings with ElevenLabs, compares `scribe_v2` with
`scribe_v3`, and scores each transcript against the AMI reference using WER
(were the words right?) and DER (was the speaker right?).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ELEVENLABS_API_KEY=...        # never commit this
```

Put the data here (or change the paths in `config.py`):

```
data/ami_annotations/words/ES2002a.A.words.xml ...   # from "AMI manual annotations v1.6.2"
data/audio/ES2002a.Mix-Headset.wav ...               # from the AMI mirror, <meeting>/audio/
```

## Run

```bash
python -m pytest tests -q                              # check the scoring on toy data first
python run_task1.py --meetings ES2002a --models scribe_v2   # one call, check it looks right
python run_task1.py                                    # everything (8 API calls)
python run_task1.py --no-transcribe                    # re-score cached responses, no API cost
```

Responses are cached in `outputs/raw/`, so changing normalisation or DER
settings and re-scoring is free.

## Files

| File | What it does |
|---|---|
| `config.py` | Every design decision, in one place, with the reason for each |
| `ami_reference.py` | AMI words XML → time-sorted words (drops punctuation and non-word markup) |
| `elevenlabs_client.py` | API call + caching; response → words |
| `normalise.py` | Text normalisation, applied the same way to both sides |
| `scoring.py` | WER (jiwer) with error breakdown; DER (pyannote.metrics) |
| `run_task1.py` | Runs the pipeline and writes transcripts, error lists and the results table |

## Outputs

- `outputs/transcripts/<meeting>__<model>.csv`: timestamped, speaker-labelled transcript.
  `ami_speaker_best_match` comes from the DER mapping, so it uses the reference
  and is for analysis only.
- `outputs/errors/<meeting>__<model>.txt`: most common substitutions, deletions and insertions.
- `outputs/results.md` / `.csv`: the results table, plus pooled test-set scores per model.
