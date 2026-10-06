from __future__ import annotations

from pathlib import Path

from subtoru.exporters.base import register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


def format_sbv_time(seconds: float) -> str:
    """Format time in H:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hours}:{mins:02d}:{secs:02d}.{millis:03d}"


class SBVExporter:
    """SubViewer (YouTube legacy) format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "sbv"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".sbv"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export cues to SBV format."""
        out = Path(output_path)
        with out.open("w", encoding="utf-8") as f:
            for i, cue in enumerate(cues):
                start = format_sbv_time(cue.start)
                end = format_sbv_time(cue.end)

                if i > 0:
                    f.write("\n")

                f.write(f"{start},{end}\n")
                f.write(f"{cue.text}\n")

        return out


register_exporter(SBVExporter())
