"""Logging and progress utilities built on rich."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Shared console instances
console = Console(stderr=True)
stdout_console = Console()

# Module logger — all Subtoru code should use `from subtoru.utils.logging import logger`
logger = logging.getLogger("subtoru")


def setup_logging(verbose: bool = False, quiet: bool = False, log_file: str | None = None) -> None:
    """Configure the root Subtoru logger.

    - ``verbose``: DEBUG level, show everything.
    - ``quiet``: WARNING level, suppress informational output.
    - default: INFO level.
    - ``log_file``: additionally write all messages to this file (always at DEBUG level).
    """
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)

    handlers: list[logging.Handler] = [
        RichHandler(
            console=console,
            show_time=verbose,
            show_path=verbose,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
        )
    ]

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logger.setLevel(level)


def make_progress(**kwargs) -> Progress:
    """Create a rich Progress bar with the standard Subtoru columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        **kwargs,
    )


def print_error(message: str) -> None:
    """Print a styled error message to stderr."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a styled warning to stderr."""
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_success(message: str) -> None:
    """Print a styled success message to stderr."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_info(message: str) -> None:
    """Print a styled info message to stderr (suppressed in quiet mode)."""
    if logger.isEnabledFor(logging.INFO):
        console.print(f"[dim]ℹ[/dim] {message}")


# --- Exit codes (§9: distinct non-zero exit codes per failure category) ---

EXIT_OK = 0
EXIT_DOWNLOAD_FAILED = 10
EXIT_TRANSCRIBE_FAILED = 20
EXIT_TRANSLATE_FAILED = 30
EXIT_EXPORT_FAILED = 40
EXIT_CONFIG_ERROR = 50
EXIT_MISSING_DEPENDENCY = 60
EXIT_VALIDATION_ERROR = 70
