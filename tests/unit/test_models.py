"""Unit tests for subtoru.models (Cue, WordTiming, TranslatedCue, RunManifest, ExportMetadata)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from subtoru.models import Cue, ExportMetadata, RunManifest, TranslatedCue, WordTiming


class TestWordTiming:
    """Tests for the WordTiming dataclass."""

    def test_word_timing_initialization(self) -> None:
        wt = WordTiming(word="hello", start=1.0, end=1.5, score=0.95)
        assert wt.word == "hello"
        assert wt.start == 1.0
        assert wt.end == 1.5
        assert wt.score == 0.95

    def test_word_timing_score_defaults_to_none(self) -> None:
        wt = WordTiming(word="world", start=2.0, end=2.4)
        assert wt.word == "world"
        assert wt.start == 2.0
        assert wt.end == 2.4
        assert wt.score is None


class TestCue:
    """Tests for the Cue dataclass and its properties."""

    def test_cue_initialization_defaults(self) -> None:
        cue = Cue(index=1, start=0.0, end=2.5, text="Hello world.")
        assert cue.index == 1
        assert cue.start == 0.0
        assert cue.end == 2.5
        assert cue.text == "Hello world."
        assert cue.speaker is None
        assert cue.words == []

    def test_cue_start_ms(self) -> None:
        cue1 = Cue(index=1, start=0.0, end=1.0, text="a")
        assert cue1.start_ms == 0

        cue2 = Cue(index=2, start=0.5, end=2.8, text="b")
        assert cue2.start_ms == 500

        cue3 = Cue(index=3, start=1.234, end=3.0, text="c")
        assert cue3.start_ms == 1234

        cue4 = Cue(index=4, start=65.4321, end=70.0, text="d")
        assert cue4.start_ms == 65432

    def test_cue_end_ms(self) -> None:
        cue1 = Cue(index=1, start=0.0, end=2.8, text="a")
        assert cue1.end_ms == 2800

        cue2 = Cue(index=2, start=1.0, end=5.5, text="b")
        assert cue2.end_ms == 5500

        cue3 = Cue(index=3, start=0.0, end=0.005, text="c")
        assert cue3.end_ms == 5

        cue4 = Cue(index=4, start=0.0, end=120.9876, text="d")
        assert cue4.end_ms == 120987

    def test_cue_duration(self) -> None:
        cue1 = Cue(index=1, start=0.5, end=2.8, text="a")
        assert cue1.duration == pytest.approx(2.3)

        cue2 = Cue(index=2, start=3.0, end=5.5, text="b")
        assert cue2.duration == pytest.approx(2.5)

        cue3 = Cue(index=3, start=10.0, end=10.0, text="zero")
        assert cue3.duration == pytest.approx(0.0)

    def test_cue_with_speaker_and_words(self, sample_cues: list[Cue]) -> None:
        cue = sample_cues[0]
        assert cue.speaker == "SPEAKER_00"
        assert len(cue.words) == 3
        assert cue.words[0].word == "Hello"
        assert cue.words[0].score == 0.98
        assert cue.start_ms == 500
        assert cue.end_ms == 2800
        assert cue.duration == pytest.approx(2.3)

    def test_sample_srt_fixture_matches_sample_cues(
        self, fixtures_dir: Path, sample_cues: list[Cue]
    ) -> None:
        srt_path = fixtures_dir / "sample.srt"
        assert srt_path.is_file()
        content = srt_path.read_text(encoding="utf-8")
        for cue in sample_cues:
            assert str(cue.index) in content
            assert cue.text in content


class TestTranslatedCue:
    """Tests for the TranslatedCue dataclass."""

    def test_translated_cue_attributes(self, sample_cues: list[Cue]) -> None:
        source_cue = sample_cues[0]
        trans = TranslatedCue(
            source=source_cue,
            translated_text="Bonjour et bienvenue.",
            target_language="fr",
        )
        assert trans.source is source_cue
        assert trans.translated_text == "Bonjour et bienvenue."
        assert trans.target_language == "fr"
        assert trans.source.start_ms == 500
        assert trans.source.speaker == "SPEAKER_00"


class TestRunManifest:
    """Tests for RunManifest state tracking and status helper methods."""

    def test_run_manifest_defaults(self) -> None:
        manifest = RunManifest(input_source="video.mp4")
        assert manifest.input_source == "video.mp4"
        assert manifest.media_path is None
        assert manifest.source_srt_path is None
        assert manifest.source_json_path is None
        assert manifest.translated_srt_paths == {}
        assert manifest.export_paths == {}
        assert manifest.config_used == {}
        assert isinstance(manifest.created_at, str)
        # Verify created_at is valid ISO format
        dt = datetime.fromisoformat(manifest.created_at)
        assert dt is not None

    def test_is_downloaded(self) -> None:
        manifest = RunManifest(input_source="https://youtube.com/watch?v=abc")
        assert manifest.is_downloaded() is False

        manifest.media_path = "/path/to/media.mp4"
        assert manifest.is_downloaded() is True

    def test_is_transcribed(self) -> None:
        manifest = RunManifest(input_source="video.mp4")
        assert manifest.is_transcribed() is False

        manifest.source_srt_path = "/path/to/video.srt"
        assert manifest.is_transcribed() is True

    def test_is_translated(self) -> None:
        manifest = RunManifest(input_source="video.mp4")
        assert manifest.is_translated("fr") is False
        assert manifest.is_translated("de") is False

        manifest.translated_srt_paths["fr"] = "/path/to/video.fr.srt"
        assert manifest.is_translated("fr") is True
        assert manifest.is_translated("de") is False

        manifest.translated_srt_paths["de"] = "/path/to/video.de.srt"
        assert manifest.is_translated("de") is True
        assert manifest.is_translated("es") is False

    def test_is_exported(self) -> None:
        manifest = RunManifest(input_source="video.mp4")
        assert manifest.is_exported("srt") is False
        assert manifest.is_exported("vtt") is False

        # Format key present but list is empty
        manifest.export_paths["srt"] = []
        assert manifest.is_exported("srt") is False

        # Format key present and list has paths
        manifest.export_paths["srt"] = ["/path/to/video.srt"]
        assert manifest.is_exported("srt") is True
        assert manifest.is_exported("vtt") is False

        # Multiple export formats
        manifest.export_paths["vtt"] = ["/path/to/video.vtt", "/path/to/video.fr.vtt"]
        assert manifest.is_exported("vtt") is True
        assert manifest.is_exported("json") is False


class TestExportMetadata:
    """Tests for ExportMetadata dataclass."""

    def test_export_metadata_defaults(self) -> None:
        meta = ExportMetadata()
        assert meta.source_language is None
        assert meta.target_language is None
        assert meta.source_media is None
        assert meta.whisper_model is None
        assert meta.gemini_model is None
        assert isinstance(meta.generated_at, str)
        dt = datetime.fromisoformat(meta.generated_at)
        assert dt is not None

    def test_export_metadata_custom_values(self, sample_metadata: ExportMetadata) -> None:
        assert sample_metadata.source_language == "en"
        assert sample_metadata.target_language == "fr"
        assert sample_metadata.source_media == "test.mp4"
        assert sample_metadata.whisper_model == "large-v2"
        assert sample_metadata.gemini_model == "gemini-2.0-flash"
        assert isinstance(sample_metadata.generated_at, str)
