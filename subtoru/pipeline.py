"""Pipeline orchestrator — wires together download → transcribe → translate → export.

Manages the RunManifest for resumability and implements --dry-run / --resume logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from subtoru.config import SubxtConfig
from subtoru.models import Cue, ExportMetadata, RunManifest, TranslatedCue
from subtoru.utils import (
    EXIT_DOWNLOAD_FAILED,
    EXIT_OK,
    EXIT_TRANSCRIBE_FAILED,
    EXIT_TRANSLATE_FAILED,
    is_url,
    logger,
    make_output_dir,
    print_error,
    print_info,
    print_success,
    print_warning,
)


def _load_manifest(output_dir: Path) -> RunManifest | None:
    """Load an existing manifest from the output directory, if present."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path) as f:
            data = json.load(f)
        return RunManifest(**data)
    except Exception as e:
        logger.debug(f"Could not load manifest: {e}")
        return None


def _save_manifest(manifest: RunManifest, output_dir: Path) -> None:
    """Write the manifest to disk."""
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    logger.debug(f"Manifest saved to {manifest_path}")


def _resolve_plan(
    input_source: str,
    config: SubxtConfig,
    skip_download: bool,
    no_translate: bool,
    resume: bool,
    manifest: RunManifest | None,
) -> dict:
    """Build a plan dict describing what each stage will do."""
    needs_download = is_url(input_source) and not skip_download
    target_langs = config.get_target_languages() if not no_translate else []
    export_fmts = config.get_export_formats()

    plan = {
        "input": input_source,
        "needs_download": needs_download,
        "skip_download": skip_download,
        "no_translate": no_translate,
        "stages": [],
        "target_languages": target_langs,
        "export_formats": export_fmts,
        "bilingual": config.bilingual,
        "bilingual_format": config.bilingual_format,
        "output_dir": config.output_dir,
    }

    # Download
    if needs_download:
        skip = resume and manifest and manifest.is_downloaded()
        plan["stages"].append(
            {"name": "download", "skip": skip, "reason": "already downloaded" if skip else ""}
        )

    # Transcribe
    skip_transcribe = resume and manifest and manifest.is_transcribed()
    plan["stages"].append(
        {
            "name": "transcribe",
            "skip": skip_transcribe,
            "reason": "already transcribed" if skip_transcribe else "",
        }
    )

    # Translate
    if not no_translate:
        for lang in target_langs:
            skip_lang = resume and manifest and manifest.is_translated(lang)
            plan["stages"].append(
                {
                    "name": f"translate ({lang})",
                    "skip": skip_lang,
                    "reason": f"already translated to {lang}" if skip_lang else "",
                }
            )

    # Export
    for fmt in export_fmts:
        skip_fmt = resume and manifest and manifest.is_exported(fmt)
        plan["stages"].append(
            {
                "name": f"export ({fmt})",
                "skip": skip_fmt,
                "reason": f"{fmt} already exported" if skip_fmt else "",
            }
        )

    if config.bilingual and target_langs:
        plan["stages"].append(
            {"name": f"bilingual ({config.bilingual_format})", "skip": False, "reason": ""}
        )

    return plan


def _print_plan(plan: dict) -> None:
    """Print a human-readable dry-run plan."""
    from rich.table import Table

    from subtoru.utils import console

    table = Table(title="Subtoru run plan", show_header=True)
    table.add_column("Stage", style="bold")
    table.add_column("Action", style="dim")
    table.add_column("Notes")

    for stage in plan["stages"]:
        action = "[yellow]SKIP[/yellow]" if stage["skip"] else "[green]RUN[/green]"
        table.add_row(stage["name"], action, stage.get("reason", ""))

    console.print(table)
    console.print(f"\n[dim]Input:[/dim] {plan['input']}")
    console.print(f"[dim]Target languages:[/dim] {', '.join(plan['target_languages']) or '(none)'}")
    console.print(f"[dim]Export formats:[/dim] {', '.join(plan['export_formats'])}")
    if plan["bilingual"]:
        console.print(f"[dim]Bilingual:[/dim] {plan['bilingual_format']}")


def run_pipeline(
    input_source: str,
    config: SubxtConfig,
    skip_download: bool = False,
    no_translate: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> int:
    """Execute the full pipeline, returning an exit code.

    Each stage's output feeds into the next. The manifest is saved after
    each stage so ``--resume`` can pick up where we left off.
    """
    # Determine output directory
    source_name = (
        Path(input_source).stem if not is_url(input_source) else input_source.split("/")[-1]
    )
    output_dir = make_output_dir(config.output_dir, source_name)

    # Load existing manifest if resuming
    manifest = _load_manifest(output_dir) if resume else None
    if manifest is None:
        manifest = RunManifest(input_source=input_source, config_used=config.model_dump())

    plan = _resolve_plan(input_source, config, skip_download, no_translate, resume, manifest)

    if dry_run:
        _print_plan(plan)
        return EXIT_OK

    # Pre-compute stage flags
    needs_download = is_url(input_source) and not skip_download
    target_langs = config.get_target_languages() if not no_translate else []
    media_path: Path

    # ---------------------------------------------------------------------------
    # Set up Rich progress display for the full pipeline
    # ---------------------------------------------------------------------------
    from subtoru.utils.progress import (
        make_download_progress,
        make_gst_progress_interceptor,
        make_pipeline_progress,
        make_whisperx_progress_callback,
        make_ytdlp_progress_hook,
    )

    # Count total stages for overall progress
    total_stages = 0
    if needs_download and not (resume and manifest.is_downloaded()):
        total_stages += 1  # download
    if not (resume and manifest.is_transcribed() and Path(manifest.source_srt_path).exists()):
        total_stages += 1  # transcribe
    if target_langs:
        total_stages += len(target_langs)  # translate (one per language)
    extra_fmts_preview = [f for f in config.get_export_formats() if f != "srt"]
    if extra_fmts_preview or config.bilingual:
        total_stages += 1  # export

    overall_progress = make_pipeline_progress()
    overall_task = overall_progress.add_task(
        "[bold cyan]Overall pipeline", total=max(total_stages, 1)
    )
    stages_completed = 0

    with overall_progress:

        # ── Stage 1: Download ──
        if needs_download:
            if resume and manifest.is_downloaded() and Path(manifest.media_path).exists():
                media_path = Path(manifest.media_path)
                print_info(f"Skipping download (resume): {media_path}")
            else:
                try:
                    from subtoru.downloader import download

                    dl_progress = make_download_progress()
                    with dl_progress:
                        dl_task = dl_progress.add_task("[green]Downloading", total=None)
                        hook = make_ytdlp_progress_hook(dl_progress, dl_task)
                        media_path, info = download(
                            input_source,
                            output_dir=output_dir,
                            audio_only=config.audio_only,
                            format_selector=config.download_format,
                            cookies_from_browser=config.cookies_from_browser,
                            progress_hook=hook,
                        )

                    manifest.media_path = str(media_path)
                    if info.get("title"):
                        source_name = info["title"]
                    _save_manifest(manifest, output_dir)
                    print_success(f"Downloaded: {media_path}")
                    stages_completed += 1
                    overall_progress.update(overall_task, completed=stages_completed)
                except Exception as e:
                    print_error(f"Download failed: {e}")
                    return EXIT_DOWNLOAD_FAILED
        else:
            media_path = Path(input_source)
            if not media_path.exists():
                print_error(f"File not found: {input_source}")
                return EXIT_DOWNLOAD_FAILED
            manifest.media_path = str(media_path)

        # ── Stage 2: Transcribe ──
        cues: list[Cue]
        raw_result: dict | None = None

        if resume and manifest.is_transcribed() and Path(manifest.source_srt_path).exists():
            print_info(f"Skipping transcription (resume): {manifest.source_srt_path}")
            from subtoru.transcriber import load_srt_to_cues

            cues = load_srt_to_cues(Path(manifest.source_srt_path))
        else:
            try:
                from subtoru.transcriber import cues_to_srt, save_raw_json, transcribe

                stage_progress = make_pipeline_progress()
                with stage_progress:
                    tx_task = stage_progress.add_task("[yellow]Transcribing", total=100)
                    al_task = stage_progress.add_task(
                        "[yellow]Aligning", total=100, visible=not config.no_align
                    )
                    di_task = stage_progress.add_task(
                        "[yellow]Diarizing", total=100, visible=config.diarize
                    )

                    cues, raw_result = transcribe(
                        media_path,
                        config=config,
                        transcribe_callback=make_whisperx_progress_callback(
                            stage_progress, tx_task
                        ),
                        align_callback=(
                            make_whisperx_progress_callback(stage_progress, al_task)
                            if not config.no_align
                            else None
                        ),
                        diarize_callback=(
                            make_whisperx_progress_callback(stage_progress, di_task)
                            if config.diarize
                            else None
                        ),
                        verbose_display=True,
                    )

                srt_path = output_dir / f"{Path(media_path).stem}.srt"
                json_path = output_dir / f"{Path(media_path).stem}.whisperx.json"

                cues_to_srt(cues, srt_path)
                save_raw_json(raw_result, json_path)

                manifest.source_srt_path = str(srt_path)
                manifest.source_json_path = str(json_path)
                _save_manifest(manifest, output_dir)
                print_success(f"Transcription: {srt_path}")
                stages_completed += 1
                overall_progress.update(overall_task, completed=stages_completed)
            except ImportError as e:
                print_error(f"whisperX not installed. Install with: pip install subtoru\n  ({e})")
                return EXIT_TRANSCRIBE_FAILED
            except Exception as e:
                print_error(f"Transcription failed: {e}")
                return EXIT_TRANSCRIBE_FAILED

        # ── Stage 3: Translate ──
        for lang in target_langs:
            if resume and manifest.is_translated(lang):
                trans_path = manifest.translated_srt_paths[lang]
                if Path(trans_path).exists():
                    print_info(f"Skipping translation to {lang} (resume): {trans_path}")
                    stages_completed += 1
                    overall_progress.update(overall_task, completed=stages_completed)
                    continue

            try:
                from subtoru.translator import translate

                source_srt = Path(manifest.source_srt_path)
                trans_out = output_dir / f"{source_srt.stem}.{lang}.srt"

                stage_progress = make_pipeline_progress()
                with stage_progress:
                    tr_task = stage_progress.add_task(f"[magenta]Translating → {lang}", total=100)
                    interceptor = make_gst_progress_interceptor(stage_progress, tr_task)

                    result_path = translate(
                        source_srt,
                        target_language=lang,
                        config=config,
                        output_path=trans_out,
                        progress_interceptor=interceptor,
                    )

                manifest.translated_srt_paths[lang] = str(result_path)
                _save_manifest(manifest, output_dir)
                print_success(f"Translated ({lang}): {result_path}")
                stages_completed += 1
                overall_progress.update(overall_task, completed=stages_completed)
            except Exception as e:
                print_error(f"Translation to {lang} failed: {e}")
                return EXIT_TRANSLATE_FAILED

        # ── Stage 4: Export ──
        export_fmts = config.get_export_formats()
        extra_fmts = [f for f in export_fmts if f != "srt"]

        if extra_fmts or config.bilingual:
            print_info("Exporting...")

            # Ensure exporters are registered
            import subtoru.exporters  # noqa: F401
            from subtoru.exporters.base import get_exporter

            # Build translated cue pairs for formats that need them
            all_translated: dict[str, list[TranslatedCue]] = {}
            for lang in target_langs:
                if lang in manifest.translated_srt_paths:
                    from subtoru.transcriber import load_srt_to_cues

                    trans_cues = load_srt_to_cues(Path(manifest.translated_srt_paths[lang]))
                    if len(trans_cues) == len(cues):
                        all_translated[lang] = [
                            TranslatedCue(
                                source=src,
                                translated_text=tr.text,
                                target_language=lang,
                            )
                            for src, tr in zip(cues, trans_cues, strict=True)
                        ]
                    else:
                        print_warning(
                            f"Cue count mismatch for {lang}: "
                            f"source={len(cues)}, translated={len(trans_cues)}. "
                            f"Skipping bilingual/merged exports for {lang}."
                        )

            metadata = ExportMetadata(
                source_language=config.source_language,
                target_language=", ".join(target_langs) if target_langs else None,
                source_media=str(media_path),
                whisper_model=config.whisper_model,
                gemini_model=config.gemini_model,
            )

            for fmt in extra_fmts:
                exporter = get_exporter(fmt)
                if exporter is None:
                    print_warning(f"Unknown export format '{fmt}', skipping.")
                    continue

                try:
                    if target_langs and all_translated:
                        for lang, trans_cues in all_translated.items():
                            out_path = (
                                output_dir / f"{Path(media_path).stem}.{lang}"
                                f"{exporter.file_extension}"
                            )
                            exporter.export(
                                cues,
                                out_path,
                                translated_cues=trans_cues,
                                metadata=metadata,
                                txt_style=config.txt_style,
                            )
                            manifest.export_paths.setdefault(fmt, []).append(str(out_path))
                            print_success(f"Exported {fmt} ({lang}): {out_path}")
                    else:
                        out_path = output_dir / f"{Path(media_path).stem}{exporter.file_extension}"
                        exporter.export(
                            cues,
                            out_path,
                            metadata=metadata,
                            txt_style=config.txt_style,
                        )
                        manifest.export_paths.setdefault(fmt, []).append(str(out_path))
                        print_success(f"Exported {fmt}: {out_path}")
                except Exception as e:
                    print_error(f"Export {fmt} failed: {e}")

            # Bilingual export
            if config.bilingual and all_translated:
                bilingual_fmt_name = f"bilingual-{config.bilingual_format}"
                exporter = get_exporter(bilingual_fmt_name)
                if exporter:
                    for lang, trans_cues in all_translated.items():
                        out_path = (
                            output_dir / f"{Path(media_path).stem}.{lang}"
                            f".bilingual{exporter.file_extension}"
                        )
                        try:
                            exporter.export(
                                cues,
                                out_path,
                                translated_cues=trans_cues,
                                metadata=metadata,
                                bilingual_order=config.bilingual_order,
                            )
                            manifest.export_paths.setdefault(bilingual_fmt_name, []).append(
                                str(out_path)
                            )
                            print_success(f"Bilingual ({lang}): {out_path}")
                        except Exception as e:
                            print_error(f"Bilingual export for {lang} failed: {e}")
                else:
                    print_warning(f"Bilingual format '{config.bilingual_format}' not available.")

            stages_completed += 1
            overall_progress.update(overall_task, completed=stages_completed)

    _save_manifest(manifest, output_dir)
    print_success("Pipeline complete!")
    return EXIT_OK
