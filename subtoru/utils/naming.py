"""Naming and path utilities."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path


def slugify(text: str, max_length: int = 80) -> str:
    """Convert text to a filesystem-safe slug.

    >>> slugify("My Video — Episode 3!")
    'my-video-episode-3'
    """
    # Normalize unicode, strip accents
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text).strip("-")
    return text[:max_length]


def sanitize_filename(name: str, *, max_length: int = 200) -> str:
    """Clean up a filename for safe filesystem use while keeping readability.

    Unlike :func:`slugify` (which aggressively lowercases and strips), this
    preserves case, spaces (collapsed), and non-ASCII letters (CJK, accented)
    but removes characters that are problematic on Windows/macOS/Linux:

    - Control characters (``\\x00``–``\\x1f``)
    - Reserved filesystem chars: ``\\ / : * ? " < > |``
    - Leading/trailing dots and spaces (Windows rejects these)
    - Runs of whitespace collapsed to a single space
    - Truncated to *max_length* characters (excluding extension)

    >>> sanitize_filename('My Video — "Episode 3"!.mp4')
    'My Video — Episode 3!.mp4'
    >>> sanitize_filename('  ...hidden  file...  ')
    'hidden  file'
    """
    # Separate extension so we don't mangle it
    p = Path(name)
    stem = p.stem
    ext = p.suffix  # e.g. ".mp4"

    # Strip control characters and reserved FS chars
    stem = re.sub(r'[\x00-\x1f\\/:*?"<>|]', "", stem)

    # Collapse runs of whitespace to a single space
    stem = re.sub(r"\s+", " ", stem).strip()

    # Strip leading/trailing dots and spaces (Windows restriction)
    stem = stem.strip(". ")

    # Truncate stem (not extension)
    if len(stem) > max_length:
        stem = stem[:max_length].rstrip()

    # Fallback for completely empty stems
    if not stem:
        stem = "untitled"

    return f"{stem}{ext}" if ext else stem


def make_output_dir(
    base: str | None,
    source_name: str,
    *,
    default_root: str = "subtoru-output",
) -> Path:
    """Create and return the output directory for a run.

    If ``base`` is provided, use it directly. Otherwise, create
    ``<default_root>/<slug>/`` relative to the current working directory.
    """
    if base:
        out = Path(base)
    else:
        slug = slugify(source_name) or "unnamed"
        out = Path(default_root) / slug
    out.mkdir(parents=True, exist_ok=True)
    return out


def derive_output_path(
    source_path: Path | str,
    suffix: str,
    output_dir: Path | str | None = None,
    language: str | None = None,
    tag: str | None = None,
) -> Path:
    """Build an output filename from a source path, new extension, and optional tags.

    Examples:
        derive_output_path("video.wav", ".srt") -> Path("video.srt")
        derive_output_path("video.wav", ".srt", language="fr") -> Path("video.fr.srt")
        derive_output_path("video.wav", ".srt", tag="bilingual") -> Path("video.bilingual.srt")
    """
    source = Path(source_path)
    stem = source.stem
    parts = [stem]
    if language:
        parts.append(language)
    if tag:
        parts.append(tag)

    filename = ".".join(parts) + suffix
    if output_dir:
        return Path(output_dir) / filename
    return source.parent / filename


def is_url(text: str) -> bool:
    """Heuristic check for whether a string looks like a URL.

    Intentionally broad — we let yt-dlp do the real validation.
    """
    return bool(re.match(r"^https?://", text)) or bool(
        re.match(r"^[a-zA-Z0-9_-]+\.[a-z]{2,}", text)  # e.g. "youtu.be/xyz"
    )
