"""Exporter protocol and format registry.

All exporters implement the ``Exporter`` protocol and register themselves
via ``register_exporter``. This allows ``subtoru export`` and the pipeline
to resolve format names to concrete exporter instances at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from subtoru.models import Cue, ExportMetadata, TranslatedCue


@runtime_checkable
class Exporter(Protocol):
    """Protocol that all format exporters must implement."""

    @property
    def format_name(self) -> str:
        """Short name used in --export-formats (e.g. 'srt', 'vtt', 'json')."""
        ...

    @property
    def file_extension(self) -> str:
        """File extension including the dot (e.g. '.srt')."""
        ...

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Write cues to a file and return the path written.

        Args:
            cues: Source-language cues.
            output_path: Where to write the file.
            translated_cues: Optional translated cue pairs (for bilingual/JSON formats).
            metadata: Optional metadata for inclusion in the output.
            **kwargs: Format-specific options.
        """
        ...


# --- Format registry ---

_registry: dict[str, Exporter] = {}


def register_exporter(exporter: Exporter) -> None:
    """Register an exporter instance by its format_name."""
    _registry[exporter.format_name.lower()] = exporter


def get_exporter(format_name: str) -> Exporter | None:
    """Look up a registered exporter by format name (case-insensitive)."""
    return _registry.get(format_name.lower())


def get_all_exporters() -> dict[str, Exporter]:
    """Return a copy of the full registry."""
    return dict(_registry)


def list_supported_formats() -> list[str]:
    """Return sorted list of all supported format names."""
    return sorted(_registry.keys())
