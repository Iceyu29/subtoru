from __future__ import annotations

from pathlib import Path

from subtoru.exporters.base import register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


def format_lrc_time(seconds: float) -> str:
    """Format time in mm:ss.xx format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 100))
    if millis >= 100:
        secs += 1
        millis = 0
    return f"{mins:02d}:{secs:02d}.{millis:02d}"


class LRCExporter:
    """LRC format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "lrc"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".lrc"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export cues to LRC format.

        Supports Standard LRC and Enhanced LRC (when word-level timings are present).
        """
        out = Path(output_path)
        with out.open("w", encoding="utf-8") as f:
            for cue in cues:
                time_tag = f"[{format_lrc_time(cue.start)}]"

                if cue.words:
                    # Enhanced LRC
                    line = ""
                    for word in cue.words:
                        line += f"<{format_lrc_time(word.start)}>{word.word} "
                    line = line.strip()
                    f.write(f"{time_tag} {line}\n")
                else:
                    # Standard LRC
                    f.write(f"{time_tag} {cue.text}\n")
        return out


register_exporter(LRCExporter())
