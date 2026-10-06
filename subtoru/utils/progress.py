"""Subtoru progress tracking — Rich progress bars for all pipeline stages.

Provides a unified progress display for the full pipeline (overall + per-stage)
and for individual subcommands (stage-only). Each stage integration uses the
native progress APIs of the underlying tools:

- yt-dlp: ``progress_hooks`` callback (bytes downloaded / total)
- whisperX: ``progress_callback`` (percentage float 0-100)
- gemini-srt-translator: monkey-patch ``logger.progress_bar``
- pysubs2/export: instant (no granular progress needed)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from subtoru.utils.logging import console

# ---------------------------------------------------------------------------
# Progress display factory
# ---------------------------------------------------------------------------


def make_pipeline_progress() -> Progress:
    """Create a Rich Progress for the full pipeline (overall + stages)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def make_download_progress() -> Progress:
    """Create a Rich Progress optimised for download tracking."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def make_stage_progress() -> Progress:
    """Create a Rich Progress for a single stage (transcribe/translate/etc.)."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# yt-dlp progress hook
# ---------------------------------------------------------------------------


def make_ytdlp_progress_hook(
    progress: Progress, task_id: TaskID
) -> Callable[[dict[str, Any]], None]:
    """Return a yt-dlp ``progress_hook`` that drives a Rich task.

    The hook interprets yt-dlp's callback dict (status, downloaded_bytes,
    total_bytes / total_bytes_estimate) and maps it onto the Rich task.
    """

    def hook(d: dict[str, Any]) -> None:
        status = d.get("status", "")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                progress.update(task_id, total=int(total), completed=int(downloaded))
        elif status == "finished":
            total = progress.tasks[task_id].total or 100
            progress.update(task_id, completed=total)

    return hook


# ---------------------------------------------------------------------------
# whisperX progress callback
# ---------------------------------------------------------------------------


def make_whisperx_progress_callback(progress: Progress, task_id: TaskID) -> Callable[[float], None]:
    """Return a whisperX ``progress_callback`` that drives a Rich task.

    whisperX calls ``callback(percentage)`` where percentage is 0.0–100.0.
    """

    def callback(pct: float) -> None:
        progress.update(task_id, completed=pct, total=100.0)

    return callback


def make_verbose_print_interceptor() -> Callable[..., None]:
    """Return a print-replacement that routes whisperX ``verbose=True`` output
    through Rich console.

    whisperX's ``transcribe(verbose=True)`` calls bare ``print()`` with lines
    like ``Transcript: [0.123 --> 2.456] Hello, world``. By temporarily
    replacing ``builtins.print`` with this interceptor, the live transcript
    text is displayed through Rich while coexisting with progress bars.
    """
    import re

    _TRANSCRIPT_RE = re.compile(r"Transcript:\s*\[[\d.]+ --> [\d.]+\]\s*(.*)")

    def interceptor(*args: Any, **kwargs: Any) -> None:
        text = " ".join(str(a) for a in args)
        m = _TRANSCRIPT_RE.match(text)
        if m:
            segment_text = m.group(1).strip()
            if segment_text:
                console.print(f"  [dim]▸[/dim] {segment_text}", highlight=False)
        # Silently drop non-transcript print() calls from whisperX
        # (e.g. "Progress: XX.XX%..." lines) — our progress bar handles those

    return interceptor


# ---------------------------------------------------------------------------
# gemini-srt-translator progress interception
# ---------------------------------------------------------------------------

# ANSI escape code pattern for stripping gst's colour codes
_ANSI_RE = None


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re

        _ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return _ANSI_RE.sub("", text)


def make_gst_progress_interceptor(progress: Progress, task_id: TaskID) -> Callable[..., None]:
    """Return a replacement for ``gemini_srt_translator.logger.progress_bar``.

    When monkey-patched onto ``gst.logger.progress_bar``, it captures the
    ``current`` and ``total`` line counts that the translator passes on
    every batch and maps them onto our Rich task.  It also displays
    status messages (passed via the ``message`` kwarg by helper functions
    like ``info_with_progress``, ``success_with_progress``, etc.).
    """

    def interceptor(
        current: int = 0,
        total: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        progress.update(task_id, completed=current, total=total)
        message = kwargs.get("message", "")
        if message:
            clean = _strip_ansi(str(message)).strip()
            if clean:
                console.print(f"  [dim]▸[/dim] {clean}", highlight=False)

    return interceptor


def make_gst_log_interceptor(style: str = "dim") -> Callable[..., None]:
    """Return a replacement for gst's standalone log functions.

    gst's ``highlight()``, ``info()``, ``progress()``, ``success()``,
    ``warning()``, and ``error()`` use bare ``print()`` with ANSI colour
    codes.  This interceptor strips the ANSI codes and routes the text
    through Rich console so it coexists with our progress bars.
    """

    def interceptor(message: Any = "", ignore_quiet: bool = False, **kwargs: Any) -> None:
        clean = _strip_ansi(str(message)).strip()
        if clean:
            console.print(f"  [{style}]▸ {clean}[/{style}]", highlight=False)

    return interceptor
