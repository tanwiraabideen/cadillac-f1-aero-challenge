"""The one data type the whole pipeline passes around: a timed, speaker-labelled word."""
from dataclasses import dataclass


@dataclass
class Word:
    speaker: str   # AMI letter ("A".."D") for the reference, "speaker_0".. for ElevenLabs
    start: float   # seconds from the start of the audio file
    end: float
    text: str
