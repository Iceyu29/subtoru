from __future__ import annotations

import json

import pysubs2

from subtoru.transcriber import _segments_to_cues, cues_to_srt, load_srt_to_cues, save_raw_json


def test_cues_to_srt(tmp_path, sample_cues):
    output_path = tmp_path / "test.srt"
    cues_to_srt(sample_cues, output_path)

    assert output_path.exists()

    # round-trip with pysubs2
    subs = pysubs2.load(str(output_path))
    assert len(subs) == len(sample_cues)
    assert subs[0].text == sample_cues[0].text


def test_save_raw_json(tmp_path):
    output_path = tmp_path / "test.json"
    data = {"test": 123}
    save_raw_json(data, output_path)

    assert output_path.exists()
    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_load_srt_to_cues(fixtures_dir):
    srt_path = fixtures_dir / "sample.srt"
    cues = load_srt_to_cues(srt_path)

    assert len(cues) == 3
    assert cues[0].start == 0.5
    assert cues[0].end == 2.8
    assert cues[0].text == "Hello and welcome."


def test_segments_to_cues():
    segments = [
        {
            "start": 0.5,
            "end": 2.8,
            "text": " Hello and welcome. ",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "Hello", "start": 0.5, "end": 0.9, "score": 0.98},
            ],
        }
    ]

    cues = _segments_to_cues(segments)
    assert len(cues) == 1
    assert cues[0].index == 1
    assert cues[0].start == 0.5
    assert cues[0].end == 2.8
    assert cues[0].text == "Hello and welcome."
    assert cues[0].speaker == "SPEAKER_00"
    assert len(cues[0].words) == 1
    assert cues[0].words[0].word == "Hello"
    assert cues[0].words[0].start == 0.5
    assert cues[0].words[0].end == 0.9
