"""Shared pytest fixtures for Subtoru test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from subtoru.models import Cue, ExportMetadata, TranslatedCue, WordTiming


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the absolute path to the tests/fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_cues() -> list[Cue]:
    """Three sample cues with word timings and speaker labels."""
    return [
        Cue(
            index=1,
            start=0.5,
            end=2.8,
            text="Hello and welcome.",
            speaker="SPEAKER_00",
            words=[
                WordTiming("Hello", 0.5, 0.9, 0.98),
                WordTiming("and", 1.0, 1.2, 0.95),
                WordTiming("welcome.", 1.3, 2.8, 0.97),
            ],
        ),
        Cue(
            index=2,
            start=3.0,
            end=5.5,
            text="This is a test.",
            speaker="SPEAKER_01",
            words=[
                WordTiming("This", 3.0, 3.3, 0.99),
                WordTiming("is", 3.4, 3.5, 0.96),
                WordTiming("a", 3.6, 3.7, 0.94),
                WordTiming("test.", 3.8, 5.5, 0.92),
            ],
        ),
        Cue(
            index=3,
            start=6.0,
            end=8.0,
            text="Goodbye!",
            speaker="SPEAKER_00",
            words=[WordTiming("Goodbye!", 6.0, 8.0, 0.99)],
        ),
    ]


@pytest.fixture
def sample_translated_cues(sample_cues: list[Cue]) -> list[TranslatedCue]:
    return [
        TranslatedCue(
            source=sample_cues[0], translated_text="Bonjour et bienvenue.", target_language="fr"
        ),
        TranslatedCue(
            source=sample_cues[1], translated_text="Ceci est un test.", target_language="fr"
        ),
        TranslatedCue(source=sample_cues[2], translated_text="Au revoir!", target_language="fr"),
    ]


@pytest.fixture
def sample_metadata() -> ExportMetadata:
    return ExportMetadata(
        source_language="en",
        target_language="fr",
        source_media="test.mp4",
        whisper_model="large-v2",
        gemini_model="gemini-2.0-flash",
    )


@pytest.fixture
def cues_no_words() -> list[Cue]:
    """Cues without word-level timing."""
    return [
        Cue(index=1, start=0.5, end=2.8, text="Hello and welcome."),
        Cue(index=2, start=3.0, end=5.5, text="This is a test."),
    ]
