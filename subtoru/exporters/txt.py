from __future__ import annotations

from pathlib import Path

from subtoru.exporters.base import register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


class TXTExporter:
    """Plain text format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "txt"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".txt"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export cues to plain text format.

        Supported kwargs:
            txt_style: 'lines' (default) or 'paragraph'
        """
        out = Path(output_path)
        txt_style = kwargs.get("txt_style", "lines")

        with out.open("w", encoding="utf-8") as f:
            if txt_style == "paragraph":
                texts = []
                for cue in cues:
                    text = f"{cue.speaker}: {cue.text}" if cue.speaker else cue.text
                    # Replace internal newlines with space to form a flowing paragraph
                    texts.append(text.replace("\n", " "))
                # Join chunks with spaces, ensuring one long paragraph string
                f.write(" ".join(texts) + "\n")
            else:
                # 'lines' format
                for cue in cues:
                    text = f"{cue.speaker}: {cue.text}" if cue.speaker else cue.text
                    f.write(text.replace("\n", " ") + "\n")

        return out


register_exporter(TXTExporter())
