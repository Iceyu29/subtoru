from __future__ import annotations

from pathlib import Path

import pysubs2

from subtoru.exporters.base import Exporter, register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


def cues_to_pysubs2(cues: list[Cue]) -> pysubs2.SSAFile:
    """Convert a list of Cues to a pysubs2.SSAFile for reuse by other exporters."""
    subs = pysubs2.SSAFile()
    for cue in cues:
        # Use \N as line separator for multi-line text (pysubs2/ASS convention)
        text = cue.text.replace("\n", "\\N")
        event = pysubs2.SSAEvent(start=cue.start_ms, end=cue.end_ms, text=text)
        if cue.speaker:
            event.name = cue.speaker
        subs.append(event)
    return subs


class Pysubs2Exporter(Exporter):
    """Exporter for formats natively supported by pysubs2."""

    def __init__(self, format_name: str, ext: str, format_: str):
        self._format_name = format_name
        self._file_extension = ext
        self._format_ = format_

    @property
    def format_name(self) -> str:
        return self._format_name

    @property
    def file_extension(self) -> str:
        return self._file_extension

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        fps: float = 23.976,
        **kwargs,
    ) -> Path:
        out = Path(output_path)
        subs = cues_to_pysubs2(cues)
        subs.save(str(out), format_=self._format_, fps=fps, **kwargs)
        return out


# Mapping: (format_name, file_extension, pysubs2_format_string)
_FORMATS = [
    ("srt", ".srt", "srt"),
    ("vtt", ".vtt", "vtt"),
    ("ass", ".ass", "ass"),
    ("ssa", ".ssa", "ssa"),
    ("microdvd", ".sub", "microdvd"),
    ("mpl2", ".mpl2", "mpl2"),
    ("tmp", ".tmp", "tmp"),
    ("ttml", ".ttml", "ttml"),
    ("sami", ".smi", "sami"),
]

for name, ext, fmt in _FORMATS:
    register_exporter(Pysubs2Exporter(name, ext, fmt))
