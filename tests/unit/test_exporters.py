import csv
import json
import re

import pysubs2
import pytest

# Make sure all exporters are imported and registered
from subtoru.exporters.base import get_exporter, list_supported_formats


def test_registry():
    exporter = get_exporter("srt")
    assert exporter is not None
    assert exporter.format_name == "srt"

    formats = list_supported_formats()
    expected_formats = {
        "srt",
        "vtt",
        "ass",
        "json",
        "lrc",
        "txt",
        "csv",
        "sbv",
        "bilingual-srt",
        "bilingual-ass",
    }
    assert expected_formats.issubset(set(formats))


@pytest.mark.parametrize("fmt", ["srt", "vtt", "ass"])
def test_pysubs2_formats_roundtrip(fmt, sample_cues, tmp_path):
    exporter = get_exporter(fmt)
    out_path = tmp_path / f"test.{fmt}"

    res = exporter.export(sample_cues, out_path)
    assert res == out_path
    assert out_path.exists()

    # Reload with pysubs2
    subs = pysubs2.load(str(out_path), format_=fmt)
    assert len(subs) == len(sample_cues)

    # Check text survives round-trip (substring check handles minor formatting diffs)
    # The cues text is "Hello and welcome.", "This is a test.", "Goodbye!"
    assert "Hello and welcome." in subs[0].text
    assert "This is a test." in subs[1].text
    assert "Goodbye!" in subs[2].text


def test_json_export(sample_cues, sample_translated_cues, sample_metadata, tmp_path):
    exporter = get_exporter("json")
    out_path = tmp_path / "test.json"

    res = exporter.export(
        sample_cues, out_path, translated_cues=sample_translated_cues, metadata=sample_metadata
    )

    assert res == out_path
    assert out_path.exists()

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "metadata" in data
    assert data["metadata"]["source_language"] == "en"

    assert "cues" in data
    assert len(data["cues"]) == 3

    # Verify word timings
    assert "words" in data["cues"][0]
    assert len(data["cues"][0]["words"]) > 0

    # Verify translated_text
    assert "translated_text" in data["cues"][0]
    assert data["cues"][0]["translated_text"] == "Bonjour et bienvenue."


def test_lrc_standard(cues_no_words, tmp_path):
    exporter = get_exporter("lrc")
    out_path = tmp_path / "standard.lrc"

    res = exporter.export(cues_no_words, out_path)
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    assert "[00:00.50] Hello and welcome." in content or "[00:00.50]Hello and welcome." in content
    # Standard format shouldn't have inline timestamps
    assert "<00:" not in content


def test_lrc_enhanced(sample_cues, tmp_path):
    exporter = get_exporter("lrc")
    out_path = tmp_path / "enhanced.lrc"

    res = exporter.export(sample_cues, out_path)
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    # Enhanced format should have inline timestamps like <00:00.50>
    assert "<00:00.50>" in content or "<00:00.50" in content
    # Should have words
    assert "Hello" in content


@pytest.mark.parametrize("style", ["lines", "paragraph"])
def test_txt_export(sample_cues, tmp_path, style):
    exporter = get_exporter("txt")
    out_path = tmp_path / f"test_{style}.txt"

    res = exporter.export(sample_cues, out_path, style=style)
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    # Verify speaker labels
    assert "SPEAKER_00:" in content
    assert "Hello and welcome." in content


def test_csv_export(sample_cues, tmp_path):
    exporter = get_exporter("csv")
    out_path = tmp_path / "test.csv"

    res = exporter.export(sample_cues, out_path)
    assert res == out_path
    assert out_path.exists()

    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    headers = rows[0]
    assert headers == ["index", "start", "end", "original_text", "translated_text", "speaker"]

    assert len(rows) == 4
    assert rows[1][0] == "1"
    assert rows[1][3] == "Hello and welcome."
    assert rows[1][5] == "SPEAKER_00"


def test_sbv_export(sample_cues, tmp_path):
    exporter = get_exporter("sbv")
    out_path = tmp_path / "test.sbv"

    res = exporter.export(sample_cues, out_path)
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    # Verify timestamp format H:MM:SS.mmm,H:MM:SS.mmm
    # 0.5 -> 0:00:00.500
    timestamp_pattern = re.compile(r"\d:\d{2}:\d{2}\.\d{3},\d:\d{2}:\d{2}\.\d{3}")
    assert timestamp_pattern.search(content) is not None


def test_bilingual_srt_original_first(sample_cues, sample_translated_cues, tmp_path):
    exporter = get_exporter("bilingual-srt")
    out_path = tmp_path / "bilingual_orig.srt"

    res = exporter.export(
        sample_cues,
        out_path,
        translated_cues=sample_translated_cues,
        bilingual_order="original-first",
    )
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    # Should see both languages
    assert "Hello and welcome." in content
    assert "Bonjour et bienvenue." in content

    # Check order manually
    idx_orig = content.find("Hello and welcome.")
    idx_trans = content.find("Bonjour et bienvenue.")
    assert idx_orig < idx_trans


def test_bilingual_srt_translated_first(sample_cues, sample_translated_cues, tmp_path):
    exporter = get_exporter("bilingual-srt")
    out_path = tmp_path / "bilingual_trans.srt"

    res = exporter.export(
        sample_cues,
        out_path,
        translated_cues=sample_translated_cues,
        bilingual_order="translated-first",
    )
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")

    idx_orig = content.find("Hello and welcome.")
    idx_trans = content.find("Bonjour et bienvenue.")
    assert idx_trans < idx_orig


def test_bilingual_ass(sample_cues, sample_translated_cues, tmp_path):
    exporter = get_exporter("bilingual-ass")
    out_path = tmp_path / "bilingual.ass"

    res = exporter.export(sample_cues, out_path, translated_cues=sample_translated_cues)
    assert res == out_path
    assert out_path.exists()

    content = out_path.read_text(encoding="utf-8")
    assert "Style: Original," in content
    assert "Style: Translated," in content
    assert "Hello and welcome." in content
    assert "Bonjour et bienvenue." in content
