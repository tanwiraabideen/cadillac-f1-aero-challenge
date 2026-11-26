"""
Send audio to ElevenLabs Speech-to-Text and cache the raw JSON response.

Why cache: every call costs credits and the output can vary slightly between
runs. Saving the raw response means scoring can be re-run (e.g. after changing
normalisation rules) without calling the API again, and anyone checking your
work scores exactly the same transcript you did.

Needs the environment variable ELEVENLABS_API_KEY. Never put the key in code.
"""
import json
import os
import time
from pathlib import Path

import requests

import config
from words import Word


def raw_path(meeting: str, model: str) -> Path:
    return config.OUTPUT_DIR / "raw" / f"{meeting}__{model}.json"


def transcribe(meeting: str, model: str, audio_path: Path, force: bool = False) -> dict:
    """Return the ElevenLabs response for this meeting/model, using the cache if present."""
    out = raw_path(meeting, model)
    if out.exists() and not force:
        print(f"  using cached {out}")
        return json.loads(out.read_text())

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ELEVENLABS_API_KEY in your environment first.")

    params = {"model_id": model, **config.ELEVENLABS_PARAMS}
    print(f"  calling ElevenLabs: {meeting} with {model} ...")
    t0 = time.time()
    with open(audio_path, "rb") as f:
        resp = requests.post(
            config.ELEVENLABS_URL,
            headers={"xi-api-key": api_key},
            data=params,
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=1800,       # a 30-minute meeting can take a while
        )
    if resp.status_code != 200:
        # Print the API's message: if a setting isn't supported by a model
        # (e.g. temperature on one of them), this is where you'll find out.
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text[:1000]}")

    result = resp.json()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    # Record exactly what was asked for, next to the result.
    meta = {"request": params, "audio": str(audio_path), "seconds_taken": round(time.time() - t0, 1)}
    out.with_suffix(".request.json").write_text(json.dumps(meta, indent=1))
    print(f"  saved {out} ({meta['seconds_taken']} s)")
    return result


def response_to_words(result: dict) -> list[Word]:
    """Turn the response's word list into Words.

    The response's "words" list mixes three types: "word", "spacing" (the gaps
    between words) and "audio_event". We keep only "word".
    """
    words = []
    for item in result.get("words", []):
        if item.get("type") != "word":
            continue
        words.append(Word(
            speaker=item.get("speaker_id") or "unknown",
            start=float(item["start"]),
            end=float(item["end"]),
            text=item["text"].strip(),
        ))
    words.sort(key=lambda w: (w.start, w.end))
    return words
