"""
Every design decision for Task 1 lives in this file, so you can see and change
them in one place (and point to them in the write-up).

Decisions marked  # DECIDED  are ones you have already made.
Decisions marked  # TODO-DECIDE  are placeholder defaults: read the comment,
pick a value, and note the reason in your write-up.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Expected layout (change to match where you unzipped things):
#   data/ami_annotations/words/ES2002a.A.words.xml   (from the 1.6.2 annotations zip)
#   data/audio/ES2002a.Mix-Headset.wav               (from the AMI mirror)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WORDS_DIR = DATA_DIR / "ami_public_manual_1" / "words"
# Audio is stored one folder per meeting: data/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav
AUDIO_ROOT = DATA_DIR / "amicorpus"
OUTPUT_DIR = PROJECT_ROOT / "task1" / "outputs"

# ---------------------------------------------------------------------------
# Which meetings, and the tune / test split
# ---------------------------------------------------------------------------
# DECIDED: tune on one meeting, test on the other three.
# TODO-DECIDE: which one is the tuning meeting. 'a' is the kick-off meeting and
# tends to be shorter and different in style (introductions), so you might
# prefer b. Whatever you pick, never choose settings by looking at test scores.
TUNE_MEETINGS = ["ES2002a"]
TEST_MEETINGS = ["ES2002b", "ES2002c", "ES2002d"]

# DECIDED: Mix-Headset (realistic single track). Per-headset is a write-up idea.
AUDIO_CHANNEL = "Mix-Headset"

# ---------------------------------------------------------------------------
# ElevenLabs request settings
# ---------------------------------------------------------------------------
# DECIDED: compare these two models.
MODELS = ["scribe_v2", "scribe_v3"]

ELEVENLABS_URL = "https://api.elevenlabs.io/v1/speech-to-text"

ELEVENLABS_PARAMS = {
    "language_code": "en",             # fixed rather than auto-detected
    "diarize": "true",
    "num_speakers": "4",               # DECIDED: 4 participants per ES meeting
    "tag_audio_events": "false",       # "(laughter)" would count as inserted words
    "no_verbatim": "false",            # DECIDED: keep fillers, as AMI does
    "timestamps_granularity": "word",
    "temperature": "0",                # as deterministic as the API allows
    "seed": "42",                      # best-effort reproducibility
    # keyterms: DECIDED not to use (see write-up)
}

# ---------------------------------------------------------------------------
# Building the reference from AMI words XML
# ---------------------------------------------------------------------------
# Elements other than <w> (<vocalsound>, <disfmarker>, <gap>, ...) are always
# ignored, and <w punc="true"> tokens are always dropped: they are not spoken words.

# Keeping them is more faithful to what was said; dropping them avoids
# penalising the ASR for fragments almost no system writes down.
DROP_TRUNCATED_WORDS = False

# ---------------------------------------------------------------------------
# Text normalisation (applied identically to reference and hypothesis)
# ---------------------------------------------------------------------------
LOWERCASE = True                  # TODO-DECIDE (almost everyone does this)
STRIP_PUNCTUATION = True          # TODO-DECIDE (keeps apostrophes inside words)
HYPHEN_TO_SPACE = True            # TODO-DECIDE "mm-hmm" -> "mm hmm", "remote-control" -> 2 words
DIGITS_TO_WORDS = True            # TODO-DECIDE "42" -> "forty two" (AMI spells numbers out)

KEEP_FILLERS = True
FILLERS = {"um", "uh", "hmm", "mm", "erm", "er", "ah", "oh"}

# Spelling variants mapped to one form before scoring. Add to this when you
# see the same word spelled two ways in your error lists, and record each
# addition: every rule here lowers WER, so each one needs a reason.
SPELLING_MAP = {
    "umm": "um",
    "uhh": "uh",
    "hmmm": "hmm",
    "mmm": "mm",
    "okay": "ok",
}

# ---------------------------------------------------------------------------
# Diarisation scoring
# ---------------------------------------------------------------------------
# DER: only bridge the tiny gaps between words. Real pauses (AMI's shortest
# is ~0.9 s on ES2002a) should stay non-speech.
DER_SEGMENT_MAX_GAP = 0.5

# Readable transcript rows / Task 2 segments: the same speaker carrying on
# after a pause stays in one row.
TRANSCRIPT_TURN_GAP = 2.0

# EACH SIDE. The common convention is 0.25. (pyannote's own `collar` argument
# is the TOTAL width, so scoring.py passes 2x this value.)
DER_COLLAR_EACH_SIDE = 0.25

# False = score everything (harder, more honest for meetings).
DER_SKIP_OVERLAP = False
