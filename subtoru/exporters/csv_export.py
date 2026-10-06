from __future__ import annotations

import csv
from pathlib import Path

from subtoru.exporters.base import register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


class CSVExporter:
    """CSV format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "csv"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".csv"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export cues to CSV format."""
        out = Path(output_path)

        translated_map = {}
        if translated_cues:
            for tc in translated_cues:
                translated_map[tc.source.index] = tc.translated_text

        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                ["index", "start", "end", "original_text", "translated_text", "speaker"]
            )

            for cue in cues:
                start_str = f"{cue.start:.3f}"
                end_str = f"{cue.end:.3f}"
                translated = translated_map.get(cue.index, "")
                speaker = cue.speaker or ""

                writer.writerow([cue.index, start_str, end_str, cue.text, translated, speaker])

        return out


register_exporter(CSVExporter())
