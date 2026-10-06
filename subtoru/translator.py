"""gemini-srt-translator Python API wrapper.

Uses the gemini_srt_translator module's Python API to translate SRT files.
Validates that the translated output preserves the source cue count —
a mismatch would cause bilingual merging to silently misalign subtitles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from subtoru.config import SubxtConfig
from subtoru.utils.logging import logger, print_error


def translate(
    srt_path: str | Path,
    *,
    target_language: str,
    config: SubxtConfig,
    output_path: Path | None = None,
    progress_interceptor: Any | None = None,
) -> Path:
    """Translate an SRT file using gemini-srt-translator.

    Args:
        srt_path: Path to the source-language SRT.
        target_language: Target language name or code (e.g. "French", "fr").
        config: Resolved SubxtConfig with API keys and model settings.
        output_path: Where to write the translated SRT. Auto-derived if None.
        progress_interceptor: Optional callable to replace gst's progress_bar
            for Rich progress tracking. Signature: ``(current, total, **kw)``.

    Returns:
        Path to the translated SRT file.

    Raises:
        ValueError: If GEMINI_API_KEY is missing or cue-count validation fails.
        ImportError: If gemini-srt-translator is not installed.
    """
    try:
        import gemini_srt_translator as gst
    except ImportError as e:
        raise ImportError(
            "gemini-srt-translator is not installed. "
            "Install with: pip install gemini-srt-translator"
        ) from e

    if not config.gemini_api_key:
        raise ValueError(
            "Gemini API key required. Set GEMINI_API_KEY env var, "
            "add it to subtoru.toml, or pass --gemini-api-key."
        )

    # Configure the translator module's globals
    gst.gemini_api_key = config.gemini_api_key
    gst.target_language = target_language
    gst.input_file = str(srt_path)

    if output_path is None:
        output_path = Path(srt_path).with_suffix(f".{target_language}.srt")
    gst.output_file = str(output_path)

    gst.model_name = config.gemini_model
    if config.translate_batch_size:
        gst.batch_size = config.translate_batch_size
    if config.description:
        gst.description = config.description
    if config.start_line:
        gst.start_line = config.start_line

    # Monkey-patch progress reporting if a Rich interceptor is provided
    _original_progress_bar = None
    _original_last_progress = None
    _original_quiet = None
    _original_log_fns: dict[str, Any] = {}
    if progress_interceptor is not None:
        try:
            import gemini_srt_translator.logger as gst_logger

            from subtoru.utils.progress import make_gst_log_interceptor

            _original_progress_bar = getattr(gst_logger, "progress_bar", None)
            _original_last_progress = getattr(gst_logger, "_last_progress", None)
            _original_quiet = getattr(gst, "quiet", None)
            gst_logger.progress_bar = progress_interceptor
            # Do NOT set quiet mode — that blocks *_with_progress helpers
            # from reaching our interceptor.  Instead, replace gst's
            # standalone log functions so their bare print() output goes
            # through Rich console (ANSI-stripped).
            _log_fn_names = [
                "highlight",
                "info",
                "progress",
                "success",
                "warning",
                "error",
            ]
            for fn_name in _log_fn_names:
                orig = getattr(gst_logger, fn_name, None)
                if orig is not None:
                    _original_log_fns[fn_name] = orig
                    setattr(gst_logger, fn_name, make_gst_log_interceptor(style="dim"))
            # gst's helper functions (info_with_progress, warning_with_progress,
            # etc.) call progress_bar(**_last_progress). _last_progress starts as
            # None and only gets set after the first real progress_bar call —
            # but with our interceptor the real call never happens, so
            # **None explodes. Seed it with a safe default dict.
            if gst_logger._last_progress is None:
                gst_logger._last_progress = {
                    "current": 0,
                    "total": 0,
                    "bar_length": 30,
                    "prefix": "",
                    "suffix": "",
                }
        except (ImportError, AttributeError):
            pass  # graceful degradation — no progress tracking

    logger.info(
        f"Translating {Path(srt_path).name} → {target_language} "
        f"(model: {getattr(gst, 'model_name', 'default')})"
    )
    try:
        gst.translate()
    finally:
        # Restore original progress_bar, _last_progress, and log functions
        if _original_progress_bar is not None:
            try:
                import gemini_srt_translator.logger as gst_logger

                gst_logger.progress_bar = _original_progress_bar
                gst_logger._last_progress = _original_last_progress
                if _original_quiet is not None:
                    gst.quiet = _original_quiet
                for fn_name, orig_fn in _original_log_fns.items():
                    setattr(gst_logger, fn_name, orig_fn)
            except (ImportError, AttributeError):
                pass

    # Validate: cue count must match for safe bilingual merging (§9)
    from subtoru.transcriber import load_srt_to_cues

    source_cues = load_srt_to_cues(Path(srt_path))
    target_cues = load_srt_to_cues(Path(output_path))

    if len(source_cues) != len(target_cues):
        msg = (
            f"Translation validation failed: source has {len(source_cues)} cues "
            f"but translated output has {len(target_cues)} cues. "
            f"The bilingual merge will be incorrect. "
            f"Try re-running the translation or adjusting --translate-batch-size."
        )
        print_error(msg)
        raise ValueError(msg)

    logger.info(f"Translation complete: {output_path}")
    return Path(output_path)


def list_models() -> None:
    """List available Gemini models via gemini-srt-translator."""
    try:
        import gemini_srt_translator as gst
    except ImportError as e:
        raise ImportError(
            "gemini-srt-translator is not installed. "
            "Install with: pip install gemini-srt-translator"
        ) from e

    gst.listmodels()
