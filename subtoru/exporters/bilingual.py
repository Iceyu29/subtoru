from __future__ import annotations

from pathlib import Path

import pysubs2

from subtoru.exporters.base import register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


class BilingualSRTExporter:
    """Bilingual SRT format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "bilingual-srt"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".srt"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export source and translated cues merged into a single SRT."""
        out = Path(output_path)

        bilingual_order = kwargs.get("bilingual_order", "original-first")

        translated_map = {}
        if translated_cues:
            for tc in translated_cues:
                translated_map[tc.source.index] = tc.translated_text

        subs = pysubs2.SSAFile()
        for cue in cues:
            translated = translated_map.get(cue.index, "")

            if bilingual_order == "original-first":
                text = f"{cue.text}\\N{translated}" if translated else cue.text
            else:
                text = f"{translated}\\N{cue.text}" if translated else cue.text

            event = pysubs2.SSAEvent(start=cue.start_ms, end=cue.end_ms, text=text)
            subs.append(event)

        subs.save(str(out), format_="srt")
        return out


class BilingualASSExporter:
    """Bilingual ASS format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "bilingual-ass"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".ass"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export source and translated cues as styled ASS."""
        out = Path(output_path)

        translated_map = {}
        if translated_cues:
            for tc in translated_cues:
                translated_map[tc.source.index] = tc.translated_text

        subs = pysubs2.SSAFile()

        subs.styles["Original"] = pysubs2.SSAStyle(
            fontsize=20, primarycolor=pysubs2.Color(255, 255, 255)
        )
        subs.styles["Translated"] = pysubs2.SSAStyle(
            fontsize=24, primarycolor=pysubs2.Color(255, 255, 0)
        )

        for cue in cues:
            translated = translated_map.get(cue.index, "")

            event_orig = pysubs2.SSAEvent(
                start=cue.start_ms, end=cue.end_ms, text=cue.text, style="Original"
            )
            subs.append(event_orig)

            if translated:
                event_trans = pysubs2.SSAEvent(
                    start=cue.start_ms, end=cue.end_ms, text=translated, style="Translated"
                )
                subs.append(event_trans)

        subs.save(str(out), format_="ass")
        return out


register_exporter(BilingualSRTExporter())
register_exporter(BilingualASSExporter())
