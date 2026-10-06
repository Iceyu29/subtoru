from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from subtoru.exporters.base import register_exporter
from subtoru.models import Cue, ExportMetadata, TranslatedCue


class JSONExporter:
    """Rich JSON format exporter."""

    @property
    def format_name(self) -> str:
        """Format name for registry."""
        return "json"

    @property
    def file_extension(self) -> str:
        """File extension."""
        return ".json"

    def export(
        self,
        cues: list[Cue],
        output_path: Path | str,
        *,
        translated_cues: list[TranslatedCue] | None = None,
        metadata: ExportMetadata | None = None,
        **kwargs,
    ) -> Path:
        """Export cues and metadata to a rich JSON format."""
        out = Path(output_path)

        meta_dict = {}
        if metadata:
            meta_dict = {k: v for k, v in asdict(metadata).items() if v is not None}

        translated_map = {}
        if translated_cues:
            for tc in translated_cues:
                translated_map[tc.source.index] = tc.translated_text

        cues_list = []
        for cue in cues:
            cue_dict = {
                "index": cue.index,
                "start": cue.start,
                "end": cue.end,
                "original_text": cue.text,
            }
            if cue.index in translated_map:
                cue_dict["translated_text"] = translated_map[cue.index]

            if cue.speaker:
                cue_dict["speaker"] = cue.speaker

            if cue.words:
                cue_dict["words"] = [asdict(w) for w in cue.words]

            cues_list.append(cue_dict)

        result = {"metadata": meta_dict, "cues": cues_list}

        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return out


register_exporter(JSONExporter())
