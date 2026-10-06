"""Core data model: Cue, WordTiming, RunManifest.

These dataclasses form the canonical in-memory representation that all pipeline stages
produce and all exporters consume. Keeping a single shared model means exporters never
need to know whether data came from whisperX, a loaded SRT, or a JSON fixture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class WordTiming:
    """A single word with its time-alignment and optional confidence score."""

    word: str
    start: float  # seconds
    end: float  # seconds
    score: float | None = None


@dataclass
class Cue:
    """One subtitle cue — the unit that every exporter works with.

    Design notes:
    - ``speaker`` is tracked separately and NEVER baked into ``text`` before
      translation, because gemini-srt-translator would mangle or literally
      translate labels like "[Speaker A]:".
    - ``words`` is empty when alignment was skipped or unavailable. Exporters
      that need word-level data (enhanced LRC, JSON) check for it and degrade
      gracefully.
    """

    index: int
    start: float  # seconds
    end: float  # seconds
    text: str
    speaker: str | None = None
    words: list[WordTiming] = field(default_factory=list)

    @property
    def start_ms(self) -> int:
        """Start time in milliseconds (what pysubs2 expects)."""
        return int(self.start * 1000)

    @property
    def end_ms(self) -> int:
        """End time in milliseconds (what pysubs2 expects)."""
        return int(self.end * 1000)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end - self.start


@dataclass
class TranslatedCue:
    """Pairs a source cue with its translated text for bilingual exports."""

    source: Cue
    translated_text: str
    target_language: str


@dataclass
class RunManifest:
    """Tracks what each pipeline stage has produced so ``--resume`` can skip completed work.

    Written as ``manifest.json`` into the output directory after each stage completes.
    """

    input_source: str  # original URL or file path
    media_path: str | None = None
    source_srt_path: str | None = None
    source_json_path: str | None = None  # raw whisperX result for word-level exports
    translated_srt_paths: dict[str, str] = field(default_factory=dict)  # {lang_code: path}
    export_paths: dict[str, list[str]] = field(default_factory=dict)
    config_used: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_downloaded(self) -> bool:
        """Check whether the download stage has completed."""
        return self.media_path is not None

    def is_transcribed(self) -> bool:
        """Check whether the transcription stage has completed."""
        return self.source_srt_path is not None

    def is_translated(self, lang: str) -> bool:
        """Check whether translation exists for a given target language."""
        return lang in self.translated_srt_paths

    def is_exported(self, fmt: str) -> bool:
        """Check whether a given export format has been produced."""
        return fmt in self.export_paths and len(self.export_paths[fmt]) > 0


@dataclass
class ExportMetadata:
    """Metadata passed to exporters for inclusion in rich output formats (JSON, etc.)."""

    source_language: str | None = None
    target_language: str | None = None
    source_media: str | None = None
    whisper_model: str | None = None
    gemini_model: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
