"""CLI entry point — Typer app with subcommands for each pipeline stage.

Usage:
    subtoru run <input> -t <lang>     # full pipeline
    subtoru download <url>            # just download
    subtoru transcribe <media>        # just transcribe
    subtoru translate <srt> -t <lang> # just translate
    subtoru export <srt> -f <fmts>    # just export
    subtoru list-models               # show available Gemini models
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from subtoru import __version__
from subtoru.config import SubxtConfig
from subtoru.utils import (
    EXIT_CONFIG_ERROR,
    EXIT_DOWNLOAD_FAILED,
    EXIT_MISSING_DEPENDENCY,
    EXIT_TRANSCRIBE_FAILED,
    EXIT_TRANSLATE_FAILED,
    EXIT_VALIDATION_ERROR,
    print_error,
    print_success,
    print_warning,
    setup_logging,
)

app = typer.Typer(
    name="subtoru",
    help="Subtoru — Subtitle generation & translation CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,  # We handle our own errors
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Subtoru {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Subtoru — Subtitle generation & translation CLI."""
    pass


# ──────────────────────────────────────────────
# subtoru run
# ──────────────────────────────────────────────


@app.command()
def run(
    input_source: Annotated[str, typer.Argument(help="A URL or local media file path.")],
    target_language: Annotated[
        str | None,
        typer.Option(
            "--target-language", "-t", help="Target language(s), comma-separated (e.g. 'fr,es,ja')."
        ),
    ] = None,
    skip_download: Annotated[
        bool,
        typer.Option(
            "--skip-download", help="Treat INPUT as a local file even if it looks like a URL."
        ),
    ] = False,
    no_translate: Annotated[
        bool, typer.Option("--no-translate", help="Stop after transcription.")
    ] = False,
    export_formats: Annotated[
        str,
        typer.Option(
            "--export-formats",
            "-f",
            help="Comma-separated output formats (e.g. 'srt,vtt,json,lrc').",
        ),
    ] = "srt",
    bilingual: Annotated[
        bool, typer.Option("--bilingual", help="Also emit a combined original+translated file.")
    ] = False,
    bilingual_format: Annotated[
        str, typer.Option("--bilingual-format", help="Format for bilingual output: 'srt' or 'ass'.")
    ] = "srt",
    bilingual_order: Annotated[
        str,
        typer.Option(
            "--bilingual-order",
            help="Which language on top: 'original-first' or 'translated-first'.",
        ),
    ] = "original-first",
    output_dir: Annotated[
        str | None, typer.Option("--output-dir", "-o", help="Root output directory.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Skip stages whose outputs already exist.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the resolved plan and exit.")
    ] = False,
    config: Annotated[str | None, typer.Option("--config", help="Path to a config file.")] = None,
    # Transcription options (whisper-prefixed)
    whisper_model: Annotated[
        str | None,
        typer.Option(
            "--whisper-model", help="Whisper model: tiny/base/small/medium/large-v2/large-v3."
        ),
    ] = None,
    source_language: Annotated[
        str | None,
        typer.Option(
            "--source-language", help="Source language ISO code, or omit for auto-detect."
        ),
    ] = None,
    device: Annotated[
        str | None, typer.Option("--device", help="'cuda' or 'cpu' (auto-detect by default).")
    ] = None,
    compute_type: Annotated[
        str | None, typer.Option("--compute-type", help="float16/float32/int8.")
    ] = None,
    whisper_batch_size: Annotated[
        int | None, typer.Option("--whisper-batch-size", help="Batched inference size.")
    ] = None,
    no_align: Annotated[
        bool, typer.Option("--no-align", help="Skip forced word-level alignment.")
    ] = False,
    align_model: Annotated[
        str | None, typer.Option("--align-model", help="Override the alignment model.")
    ] = None,
    task: Annotated[
        str | None,
        typer.Option("--task", help="'transcribe' or 'translate' (X→English)."),
    ] = None,
    initial_prompt: Annotated[
        str | None,
        typer.Option(
            "--initial-prompt",
            help="Prompt for first window (domain vocabulary, context).",
        ),
    ] = None,
    hotwords: Annotated[
        str | None,
        typer.Option("--hotwords", help="Rare/technical terms hint."),
    ] = None,
    vad_onset: Annotated[
        float | None,
        typer.Option("--vad-onset", help="VAD onset threshold (lower=more sensitive)."),
    ] = None,
    vad_offset: Annotated[
        float | None,
        typer.Option("--vad-offset", help="VAD offset threshold."),
    ] = None,
    chunk_size: Annotated[
        int | None,
        typer.Option("--chunk-size", help="VAD chunk size in seconds."),
    ] = None,
    suppress_numerals: Annotated[
        bool,
        typer.Option("--suppress-numerals", help="Suppress numeric/currency symbols."),
    ] = False,
    threads: Annotated[
        int | None,
        typer.Option("--threads", help="CPU threads per worker."),
    ] = None,
    diarize: Annotated[bool, typer.Option("--diarize", help="Enable speaker diarization.")] = False,
    hf_token: Annotated[
        str | None, typer.Option("--hf-token", help="Hugging Face token (for diarization).")
    ] = None,
    diarize_model: Annotated[
        str | None,
        typer.Option("--diarize-model", help="Speaker diarization model name."),
    ] = None,
    num_speakers: Annotated[
        int | None, typer.Option("--num-speakers", help="Exact number of speakers.")
    ] = None,
    min_speakers: Annotated[
        int | None, typer.Option("--min-speakers", help="Minimum speakers for diarization.")
    ] = None,
    max_speakers: Annotated[
        int | None, typer.Option("--max-speakers", help="Maximum speakers for diarization.")
    ] = None,
    # Translation options (gemini-prefixed)
    gemini_api_key: Annotated[
        str | None, typer.Option("--gemini-api-key", help="Gemini API key.")
    ] = None,
    gemini_model: Annotated[
        str | None, typer.Option("--gemini-model", help="Gemini model for translation.")
    ] = None,
    translate_batch_size: Annotated[
        int | None, typer.Option("--translate-batch-size", help="Cues per translation batch.")
    ] = None,
    description: Annotated[
        str | None, typer.Option("--description", help="Context hint for the translator.")
    ] = None,
    # Download options
    audio_only: Annotated[
        bool, typer.Option("--audio-only", help="Extract audio only when downloading.")
    ] = False,
    download_format: Annotated[
        str | None, typer.Option("--download-format", help="yt-dlp format selector.")
    ] = None,
    cookies_from_browser: Annotated[
        str | None, typer.Option("--cookies-from-browser", help="Browser for cookie extraction.")
    ] = None,
    # General options
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output.")] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress informational output.")
    ] = False,
    log_file: Annotated[
        str | None, typer.Option("--log-file", help="Write logs to this file.")
    ] = None,
    txt_style: Annotated[
        str | None, typer.Option("--txt-style", help="'lines' or 'paragraph' for TXT export.")
    ] = None,
) -> None:
    """Run the full pipeline: download → transcribe → translate → export."""
    setup_logging(verbose=verbose, quiet=quiet, log_file=log_file)

    # Build config from all sources
    cfg = SubxtConfig.load(
        config_path=config,
        target_language=target_language,
        export_formats=export_formats,
        bilingual=bilingual,
        bilingual_format=bilingual_format,
        bilingual_order=bilingual_order,
        output_dir=output_dir,
        whisper_model=whisper_model,
        source_language=source_language,
        device=device,
        compute_type=compute_type,
        whisper_batch_size=whisper_batch_size,
        no_align=no_align,
        align_model=align_model,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        chunk_size=chunk_size,
        suppress_numerals=suppress_numerals,
        threads=threads,
        diarize=diarize,
        hf_token=hf_token,
        diarize_model=diarize_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        translate_batch_size=translate_batch_size,
        description=description,
        audio_only=audio_only,
        download_format=download_format if download_format else None,
        cookies_from_browser=cookies_from_browser,
        verbose=verbose,
        quiet=quiet,
        log_file=log_file,
        txt_style=txt_style if txt_style else None,
    )

    # Validate required inputs
    if not no_translate and not cfg.get_target_languages():
        print_error("--target-language is required unless --no-translate is set.")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    from subtoru.pipeline import run_pipeline

    exit_code = run_pipeline(
        input_source=input_source,
        config=cfg,
        skip_download=skip_download,
        no_translate=no_translate,
        resume=resume,
        dry_run=dry_run,
    )
    raise typer.Exit(exit_code)


# ──────────────────────────────────────────────
# subtoru download
# ──────────────────────────────────────────────


@app.command()
def download(
    url: Annotated[str, typer.Argument(help="URL to download.")],
    audio_only: Annotated[bool, typer.Option("--audio-only", help="Extract audio only.")] = False,
    format: Annotated[
        str | None, typer.Option("--format", help="yt-dlp format selector (auto-detected).")
    ] = None,
    cookies_from_browser: Annotated[
        str | None, typer.Option("--cookies-from-browser", help="Browser for cookie extraction.")
    ] = None,
    output_dir: Annotated[
        str | None, typer.Option("--output-dir", "-o", help="Output directory.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Download media from a URL using yt-dlp."""
    setup_logging(verbose=verbose, quiet=quiet)

    from subtoru.downloader import download as do_download
    from subtoru.utils.progress import make_download_progress, make_ytdlp_progress_hook

    try:
        out_dir = Path(output_dir) if output_dir else Path.cwd()
        dl_progress = make_download_progress()
        with dl_progress:
            dl_task = dl_progress.add_task("[green]Downloading", total=None)
            hook = make_ytdlp_progress_hook(dl_progress, dl_task)
            media_path, info = do_download(
                url,
                output_dir=out_dir,
                audio_only=audio_only,
                format_selector=format,
                cookies_from_browser=cookies_from_browser,
                progress_hook=hook,
            )
        print_success(f"Downloaded: {media_path}")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(EXIT_DOWNLOAD_FAILED) from None


# ──────────────────────────────────────────────
# subtoru transcribe
# ──────────────────────────────────────────────


@app.command()
def transcribe(
    media: Annotated[str, typer.Argument(help="Path to media/audio file.")],
    whisper_model: Annotated[
        str, typer.Option("--whisper-model", help="Whisper model name.")
    ] = "large-v2",
    source_language: Annotated[
        str | None, typer.Option("--source-language", help="Source language ISO code.")
    ] = None,
    device: Annotated[str | None, typer.Option("--device", help="'cuda' or 'cpu'.")] = None,
    compute_type: Annotated[str | None, typer.Option("--compute-type")] = None,
    whisper_batch_size: Annotated[int, typer.Option("--whisper-batch-size")] = 16,
    no_align: Annotated[bool, typer.Option("--no-align", help="Skip forced alignment.")] = False,
    align_model: Annotated[str | None, typer.Option("--align-model")] = None,
    task: Annotated[
        str, typer.Option("--task", help="'transcribe' or 'translate' (X→English).")
    ] = "transcribe",
    initial_prompt: Annotated[
        str | None,
        typer.Option("--initial-prompt", help="Prompt for first window (domain vocabulary)."),
    ] = None,
    hotwords: Annotated[
        str | None, typer.Option("--hotwords", help="Rare/technical terms hint.")
    ] = None,
    vad_onset: Annotated[float, typer.Option("--vad-onset", help="VAD onset threshold.")] = 0.5,
    vad_offset: Annotated[
        float, typer.Option("--vad-offset", help="VAD offset threshold.")
    ] = 0.363,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="VAD chunk size in seconds.")] = 5,
    suppress_numerals: Annotated[
        bool, typer.Option("--suppress-numerals", help="Suppress numeric/currency symbols.")
    ] = False,
    threads: Annotated[int, typer.Option("--threads", help="CPU threads per worker.")] = 4,
    diarize: Annotated[bool, typer.Option("--diarize", help="Enable speaker diarization.")] = False,
    hf_token: Annotated[str | None, typer.Option("--hf-token")] = None,
    diarize_model: Annotated[
        str | None, typer.Option("--diarize-model", help="Speaker diarization model.")
    ] = None,
    num_speakers: Annotated[int | None, typer.Option("--num-speakers")] = None,
    min_speakers: Annotated[int | None, typer.Option("--min-speakers")] = None,
    max_speakers: Annotated[int | None, typer.Option("--max-speakers")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Transcribe a media file using whisperX."""
    setup_logging(verbose=verbose, quiet=quiet)

    media_path = Path(media)
    if not media_path.exists():
        print_error(f"File not found: {media}")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    cfg = SubxtConfig.load(
        whisper_model=whisper_model,
        source_language=source_language,
        device=device or "auto",
        compute_type=compute_type,
        whisper_batch_size=whisper_batch_size,
        no_align=no_align,
        align_model=align_model,
        task=task,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        chunk_size=chunk_size,
        suppress_numerals=suppress_numerals,
        threads=threads,
        diarize=diarize,
        hf_token=hf_token,
        diarize_model=diarize_model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        output_dir=output_dir,
    )

    try:
        from subtoru.transcriber import cues_to_srt, save_raw_json
        from subtoru.transcriber import transcribe as do_transcribe
        from subtoru.utils.progress import (
            make_stage_progress,
            make_whisperx_progress_callback,
        )

        stage_progress = make_stage_progress()
        with stage_progress:
            tx_task = stage_progress.add_task("[yellow]Transcribing", total=100)
            al_task = stage_progress.add_task(
                "[yellow]Aligning", total=100, visible=not cfg.no_align
            )
            di_task = stage_progress.add_task("[yellow]Diarizing", total=100, visible=cfg.diarize)

            cues, raw_result = do_transcribe(
                media_path,
                config=cfg,
                transcribe_callback=make_whisperx_progress_callback(stage_progress, tx_task),
                align_callback=(
                    make_whisperx_progress_callback(stage_progress, al_task)
                    if not cfg.no_align
                    else None
                ),
                diarize_callback=(
                    make_whisperx_progress_callback(stage_progress, di_task)
                    if cfg.diarize
                    else None
                ),
                verbose_display=True,
            )

        out_dir = Path(output_dir) if output_dir else media_path.parent
        from subtoru.utils.naming import derive_output_path

        srt_path = derive_output_path(media_path, ".srt", output_dir=out_dir)
        json_path = derive_output_path(media_path, ".json", output_dir=out_dir)

        cues_to_srt(cues, srt_path)
        save_raw_json(raw_result, json_path)

        print_success(f"Transcription: {srt_path}")
        print_success(f"Raw JSON: {json_path}")
    except ImportError as e:
        print_error(f"whisperX is not installed. Install with: pip install subtoru\n  ({e})")
        raise typer.Exit(EXIT_MISSING_DEPENDENCY) from None
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(EXIT_TRANSCRIBE_FAILED) from None


# ──────────────────────────────────────────────
# subtoru translate
# ──────────────────────────────────────────────


@app.command()
def translate(
    srt: Annotated[str, typer.Argument(help="Path to source SRT file.")],
    target_language: Annotated[
        str, typer.Option("--target-language", "-t", help="Target language.")
    ],
    gemini_api_key: Annotated[str | None, typer.Option("--gemini-api-key")] = None,
    gemini_model: Annotated[str | None, typer.Option("--gemini-model")] = None,
    translate_batch_size: Annotated[int | None, typer.Option("--translate-batch-size")] = None,
    description: Annotated[str | None, typer.Option("--description")] = None,
    start_line: Annotated[int | None, typer.Option("--start-line")] = None,
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Translate an SRT file using gemini-srt-translator."""
    setup_logging(verbose=verbose, quiet=quiet)

    srt_path = Path(srt)
    if not srt_path.exists():
        print_error(f"File not found: {srt}")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    cfg = SubxtConfig.load(
        gemini_api_key=gemini_api_key,
        gemini_model=gemini_model,
        translate_batch_size=translate_batch_size,
        description=description,
        start_line=start_line,
        target_language=target_language,
    )

    if not cfg.gemini_api_key:
        print_error("Gemini API key required. Set --gemini-api-key or GEMINI_API_KEY env var.")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    try:
        from subtoru.translator import translate as do_translate
        from subtoru.utils.naming import derive_output_path
        from subtoru.utils.progress import (
            make_gst_progress_interceptor,
            make_stage_progress,
        )

        out_dir = Path(output_dir) if output_dir else srt_path.parent
        out_path = derive_output_path(
            srt_path, ".srt", output_dir=out_dir, language=target_language
        )

        stage_progress = make_stage_progress()
        with stage_progress:
            tr_task = stage_progress.add_task(
                f"[magenta]Translating → {target_language}", total=100
            )
            interceptor = make_gst_progress_interceptor(stage_progress, tr_task)

            result_path = do_translate(
                srt_path,
                target_language=target_language,
                config=cfg,
                output_path=out_path,
                progress_interceptor=interceptor,
            )

        print_success(f"Translated: {result_path}")
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(EXIT_TRANSLATE_FAILED) from None


# ──────────────────────────────────────────────
# subtoru export
# ──────────────────────────────────────────────


@app.command(name="export")
def export_cmd(
    srt: Annotated[str, typer.Argument(help="Path to source SRT file.")],
    formats: Annotated[
        str, typer.Option("--formats", "-f", help="Comma-separated output formats.")
    ] = "srt,vtt",
    translated_srt: Annotated[
        str | None,
        typer.Option("--translated-srt", help="Path to translated SRT for bilingual/JSON export."),
    ] = None,
    target_language: Annotated[str | None, typer.Option("--target-language", "-t")] = None,
    bilingual: Annotated[bool, typer.Option("--bilingual")] = False,
    bilingual_format: Annotated[str, typer.Option("--bilingual-format")] = "srt",
    bilingual_order: Annotated[str, typer.Option("--bilingual-order")] = "original-first",
    txt_style: Annotated[str, typer.Option("--txt-style")] = "lines",
    output_dir: Annotated[str | None, typer.Option("--output-dir", "-o")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Export an SRT file to other subtitle/text formats."""
    setup_logging(verbose=verbose, quiet=quiet)

    srt_path = Path(srt)
    if not srt_path.exists():
        print_error(f"File not found: {srt}")
        raise typer.Exit(EXIT_CONFIG_ERROR) from None

    from subtoru.transcriber import load_srt_to_cues

    cues = load_srt_to_cues(srt_path)

    # Load translated cues if provided
    translated_cues = None
    if translated_srt:
        trans_path = Path(translated_srt)
        if not trans_path.exists():
            print_error(f"Translated SRT not found: {translated_srt}")
            raise typer.Exit(EXIT_CONFIG_ERROR) from None
        trans_cue_list = load_srt_to_cues(trans_path)
        if len(trans_cue_list) != len(cues):
            print_error(
                f"Cue count mismatch: source has {len(cues)} cues, "
                f"translated has {len(trans_cue_list)}. Cannot merge."
            )
            raise typer.Exit(EXIT_VALIDATION_ERROR)
        from subtoru.models import TranslatedCue

        lang = target_language or "translated"
        translated_cues = [
            TranslatedCue(source=src, translated_text=tr.text, target_language=lang)
            for src, tr in zip(cues, trans_cue_list, strict=True)
        ]

    out_dir = Path(output_dir) if output_dir else srt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure exporters are registered
    import subtoru.exporters  # noqa: F401
    from subtoru.exporters.base import get_exporter
    from subtoru.utils.naming import derive_output_path

    format_list = [f.strip().lower() for f in formats.split(",") if f.strip()]

    for fmt in format_list:
        exporter = get_exporter(fmt)
        if exporter is None:
            print_warning(f"Unknown format '{fmt}', skipping.")
            continue
        out_path = derive_output_path(srt_path, exporter.file_extension, output_dir=out_dir)
        try:
            result = exporter.export(
                cues,
                out_path,
                translated_cues=translated_cues,
                txt_style=txt_style,
            )
            print_success(f"Exported {fmt}: {result}")
        except Exception as e:
            print_error(f"Failed to export {fmt}: {e}")

    # Bilingual export
    if bilingual and translated_cues:
        bilingual_fmt = f"bilingual-{bilingual_format}"
        exporter = get_exporter(bilingual_fmt)
        if exporter:
            out_path = derive_output_path(
                srt_path, exporter.file_extension, output_dir=out_dir, tag="bilingual"
            )
            try:
                result = exporter.export(
                    cues,
                    out_path,
                    translated_cues=translated_cues,
                    bilingual_order=bilingual_order,
                )
                print_success(f"Exported bilingual: {result}")
            except Exception as e:
                print_error(f"Failed bilingual export: {e}")
        else:
            print_warning(f"Bilingual format '{bilingual_format}' not supported.")


# ──────────────────────────────────────────────
# subtoru list-models
# ──────────────────────────────────────────────


@app.command(name="list-models")
def list_models(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """List available Gemini models for translation."""
    setup_logging(verbose=verbose, quiet=quiet)

    try:
        from subtoru.translator import list_models as do_list

        do_list()
    except ImportError as e:
        print_error(f"gemini-srt-translator is not installed: {e}")
        raise typer.Exit(EXIT_MISSING_DEPENDENCY) from None
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(EXIT_CONFIG_ERROR) from None


# Allow `python -m subtoru`
if __name__ == "__main__":
    app()
