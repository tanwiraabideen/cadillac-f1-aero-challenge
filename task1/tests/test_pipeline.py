"""
Sanity tests on tiny made-up data. Each test checks one thing you'd want to be
sure of before trusting a number in your results table.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from ami_reference import load_reference
from elevenlabs_client import response_to_words
from normalise import normalise_text, normalise_words
from scoring import compute_der, compute_wer, words_to_segments
from words import Word

NS = 'xmlns:nite="http://nite.sourceforge.net/"'


def write_xml(folder: Path, meeting: str, speaker: str, body: str):
    (folder / f"{meeting}.{speaker}.words.xml").write_text(
        f'<?xml version="1.0"?>\n<nite:root nite:id="{meeting}.{speaker}.words" {NS}>\n{body}\n</nite:root>')


@pytest.fixture
def ami_dir(tmp_path):
    write_xml(tmp_path, "FAKE", "A", """
  <w nite:id="a1" starttime="1.00" endtime="1.40">Okay</w>
  <w nite:id="a2" starttime="1.40" endtime="1.40" punc="true">,</w>
  <w nite:id="a3" starttime="1.50" endtime="1.90">let's</w>
  <w nite:id="a4" starttime="1.90" endtime="2.30">start</w>
  <vocalsound nite:id="a5" starttime="2.40" endtime="2.90" type="laugh"/>
  <w nite:id="a6" starttime="2.30" endtime="2.30" punc="true">.</w>""")
    write_xml(tmp_path, "FAKE", "B", """
  <w nite:id="b1" starttime="3.00" endtime="3.30">Um</w>
  <disfmarker nite:id="b2" starttime="3.30" endtime="3.30"/>
  <w nite:id="b3" starttime="3.40" endtime="3.80">remote-control</w>
  <w nite:id="b4" starttime="3.90" endtime="4.10" trunc="true">th</w>
  <w nite:id="b5">yeah</w>""")
    return tmp_path


def test_reference_drops_punctuation_and_non_words(ami_dir):
    words, stats = load_reference("FAKE", ami_dir)
    assert [w.text for w in words] == ["Okay", "let's", "start", "Um", "remote-control", "th", "yeah"]
    assert stats["A"]["punctuation"] == 2
    assert stats["A"]["other_elements"] == 1          # the vocalsound
    assert stats["B"]["missing_times"] == 1            # "yeah" had no times


def test_reference_is_time_sorted_across_speakers(ami_dir):
    words, _ = load_reference("FAKE", ami_dir)
    assert [w.speaker for w in words] == ["A", "A", "A", "B", "B", "B", "B"]
    starts = [w.start for w in words]
    assert starts[:6] == sorted(starts[:6])


def test_normalisation_rules():
    assert normalise_text("Okay,") == ["ok"]
    assert normalise_text("Remote-control") == ["remote", "control"]
    assert normalise_text("42") == ["forty", "two"]
    assert normalise_text("don't") == ["don't"]
    assert normalise_text("Umm...") == ["um"]
    assert normalise_text(".") == []


def test_both_sides_normalise_to_the_same_thing():
    # Same speech, different writing conventions -> zero WER.
    ref = normalise_words([Word("A", 0, 1, "Okay"), Word("A", 1, 2, "forty-two"), Word("A", 2, 3, "mm-hmm")])
    hyp = normalise_words([Word("s0", 0, 1, "OK."), Word("s0", 1, 2, "42"), Word("s0", 2, 3, "Mm-hmm.")])
    assert compute_wer(ref, hyp)["wer"] == 0


def test_wer_counts_each_error_type():
    ref = [Word("A", i, i + 1, t) for i, t in enumerate("the red big button is here".split())]
    hyp = [Word("s", i, i + 1, t) for i, t in enumerate("a red button is here now".split())]
    r = compute_wer(ref, hyp)
    # the->a (sub), "big" deleted, "now" inserted: 3 errors / 6 ref words
    assert r["wer"] == pytest.approx(3 / 6)
    assert r["top_substitutions"] == [(("the", "a"), 1)]
    assert r["top_deletions"] == [("big", 1)]
    assert r["top_insertions"] == [("now", 1)]


def test_segments_merge_small_gaps_only():
    ws = [Word("A", 0.0, 0.5, "x"), Word("A", 0.8, 1.2, "x"), Word("A", 3.0, 3.5, "x")]
    assert words_to_segments(ws, max_gap=0.5) == [("A", 0.0, 1.2), ("A", 3.0, 3.5)]


def test_der_is_zero_when_only_label_names_differ():
    ref = [Word("A", 0, 2, "x"), Word("B", 3, 5, "x"), Word("A", 6, 8, "x")]
    hyp = [Word("speaker_1", 0, 2, "x"), Word("speaker_0", 3, 5, "x"), Word("speaker_1", 6, 8, "x")]
    r = compute_der(ref, hyp, "t")
    assert r["der"] == pytest.approx(0)
    assert r["speaker_mapping"] == {"speaker_1": "A", "speaker_0": "B"}


def test_der_counts_confusion():
    # 2 s of B's 6 s total speech is labelled as A's speaker: confusion = 2/6.
    ref = [Word("A", 0, 2, "x"), Word("B", 10, 12, "x"), Word("A", 20, 22, "x")]
    hyp = [Word("s1", 0, 2, "x"), Word("s1", 10, 12, "x"), Word("s1", 20, 22, "x")]
    r = compute_der(ref, hyp, "t")
    assert r["confusion"] == pytest.approx(2 / 6)
    assert r["missed"] == pytest.approx(0) and r["false_alarm"] == pytest.approx(0)


def test_elevenlabs_response_keeps_only_words():
    resp = {"words": [
        {"text": "Hello", "start": 0.1, "end": 0.4, "type": "word", "speaker_id": "speaker_0"},
        {"text": " ", "start": 0.4, "end": 0.5, "type": "spacing", "speaker_id": "speaker_0"},
        {"text": "(laughs)", "start": 0.5, "end": 1.0, "type": "audio_event", "speaker_id": "speaker_1"},
        {"text": "there", "start": 0.5, "end": 0.8, "type": "word", "speaker_id": "speaker_0"},
    ]}
    assert [w.text for w in response_to_words(resp)] == ["Hello", "there"]
