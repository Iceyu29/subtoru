"""yt-dlp Python API wrapper for media downloading.

Uses yt-dlp's Python API (not subprocess) to download media from URLs.
When audio_only is False, downloads the best video+audio merge to avoid
"blank video" (audio-only stream selected by mistake).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yt_dlp
from yt_dlp.utils import DownloadError

from subtoru.utils.logging import logger, print_error


class _YDLLogger:
    """Route yt-dlp log messages through Subtoru's logger."""

    def debug(self, msg: str) -> None:
        logger.debug(msg)

    def info(self, msg: str) -> None:
        logger.info(msg)

    def warning(self, msg: str) -> None:
        logger.warning(msg)

    def error(self, msg: str) -> None:
        logger.error(msg)


def download(
    url: str,
    *,
    output_dir: str | Path,
    audio_only: bool = False,
    format_selector: str | None = None,
    cookies_from_browser: str | None = None,
    progress_hook: Any | None = None,
) -> tuple[Path, dict]:
    """Download media using yt-dlp.

    Args:
        url: The URL to download from.
        output_dir: Directory to save the downloaded file.
        audio_only: If True, extract audio to WAV.
        format_selector: yt-dlp format string. If None, auto-selects:
            - ``bestaudio/best`` when audio_only is True
            - ``bestvideo+bestaudio/best`` otherwise (avoids blank-video)
        cookies_from_browser: Browser name for cookie extraction.
        progress_hook: Optional yt-dlp progress hook callback.

    Returns:
        A tuple of (downloaded_file_path, info_dict).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Choose the right default format to avoid the "blank video" problem:
    # bestaudio/best selects audio-only streams, which produces a file with
    # no video track. For video downloads we need bestvideo+bestaudio/best.
    if format_selector is None:
        format_selector = "bestaudio/best" if audio_only else "bestvideo+bestaudio/best"

    ydl_opts: dict[str, Any] = {
        "format": format_selector,
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "logger": _YDLLogger(),
        "merge_output_format": "mp4",  # ensure merged output is mp4
        "noprogress": True,  # suppress yt-dlp's own progress (we use Rich)
    }

    if progress_hook:
        ydl_opts["progress_hooks"] = [progress_hook]

    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)

    if audio_only:
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Extracting info and downloading from {url}...")
            info_dict = ydl.extract_info(url, download=True)
            if not info_dict:
                raise RuntimeError("Failed to extract info from URL")

            # Retrieve the path to the downloaded (and potentially post-processed) file
            requested_downloads = info_dict.get("requested_downloads", [])
            if requested_downloads:
                downloaded_file = Path(requested_downloads[0]["filepath"])
            else:
                # Fallback if requested_downloads is missing
                expected_filename = ydl.prepare_filename(info_dict)
                if audio_only:
                    downloaded_file = Path(expected_filename).with_suffix(".wav")
                else:
                    downloaded_file = Path(expected_filename)

            title = info_dict.get("title", "Unknown Title")
            video_id = info_dict.get("id", "Unknown ID")
            uploader = info_dict.get("uploader", "Unknown Uploader")
            duration = info_dict.get("duration", 0)

            logger.info(
                f"Downloaded: {title} "
                f"(ID: {video_id}, Uploader: {uploader}, Duration: {duration}s)"
            )

            # Sanitize the filename — yt-dlp %(title)s can produce names with
            # reserved chars, control chars, excessive length, etc.
            from subtoru.utils.naming import sanitize_filename

            clean_name = sanitize_filename(downloaded_file.name)
            if clean_name != downloaded_file.name:
                clean_path = downloaded_file.parent / clean_name
                downloaded_file.rename(clean_path)
                downloaded_file = clean_path
                logger.debug(f"Renamed to sanitized filename: {clean_name}")

            return downloaded_file, info_dict

    except DownloadError as e:
        msg = f"Download failed: {e}. " "The media might be private, removed, or region-locked."
        print_error(msg)
        raise RuntimeError(msg) from e
