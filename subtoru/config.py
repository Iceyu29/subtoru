"""Configuration loading: CLI flags > env vars > config file > defaults.

Uses pydantic-settings for typed config with automatic env-var binding and
python-dotenv for .env file support. Supports subtoru.toml and [tool.subtoru]
in pyproject.toml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


def _find_config_file() -> Path | None:
    """Locate the nearest config file, checking subtoru.toml first, then pyproject.toml."""
    cwd = Path.cwd()
    for name in ("subtoru.toml",):
        candidate = cwd / name
        if candidate.is_file():
            return candidate
    # Also check pyproject.toml for a [tool.subtoru] table
    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file():
        return pyproject
    return None


def _load_toml_config(path: Path | None) -> dict[str, Any]:
    """Load config from a TOML file, handling both subtoru.toml and pyproject.toml."""
    if path is None:
        return {}
    try:
        import tomllib
    except ImportError:
        # Python 3.10 compat
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}

    # If it's pyproject.toml, extract [tool.subtoru]
    if path.name == "pyproject.toml":
        return data.get("tool", {}).get("subtoru", {})
    return data


class SubxtConfig(BaseSettings):
    """Unified configuration for all Subtoru operations.

    Precedence: CLI flag (set at call site) > env var > config file > default here.
    Field names map to env vars as SUBTORU_<FIELD> (e.g. SUBTORU_WHISPER_MODEL) except
    where ``env=`` overrides that (GEMINI_API_KEY, HF_TOKEN keep their canonical names).
    """

    # --- Secrets (env-var names match what the upstream tools already expect) ---
    gemini_api_key: str | None = Field(default=None, json_schema_extra={"env": "GEMINI_API_KEY"})
    gemini_api_key2: str | None = Field(default=None, json_schema_extra={"env": "GEMINI_API_KEY2"})
    hf_token: str | None = Field(default=None, json_schema_extra={"env": "HF_TOKEN"})

    # --- Download defaults ---
    download_format: str | None = None  # None = auto-select in downloader
    audio_only: bool = False
    cookies_from_browser: str | None = None

    # --- Transcription defaults ---
    whisper_model: str = "large-v2"
    source_language: str | None = None  # None = auto-detect
    device: str = "auto"  # "auto", "cuda", or "cpu"
    compute_type: str | None = None  # None = auto-select based on device
    whisper_batch_size: int = 16
    no_align: bool = False
    align_model: str | None = None
    task: str = "transcribe"  # "transcribe" or "translate" (X→English)

    # Prompting — improve recognition of domain-specific terms
    initial_prompt: str | None = None  # prompt for first transcription window
    hotwords: str | None = None  # comma-separated rare/technical terms

    # VAD (Voice Activity Detection) tuning
    vad_method: str = "pyannote"  # "pyannote" or "silero"
    vad_onset: float = 0.5  # lower = more sensitive speech detection
    vad_offset: float = 0.363  # lower = more sensitive speech end detection
    chunk_size: int = 5  # seconds; reduce for very long audio

    # Alignment options
    interpolate_method: str = "nearest"  # "nearest", "linear", or "ignore"
    return_char_alignments: bool = False  # character-level alignment in JSON
    suppress_numerals: bool = False  # suppress numbers (wav2vec2 can't align them)

    # Output formatting
    segment_resolution: str = "sentence"  # "sentence" or "chunk"
    max_line_width: int | None = None  # max chars per line before breaking
    max_line_count: int | None = None  # max lines per segment
    highlight_words: bool = False  # underline words as spoken in SRT/VTT

    # Decoding parameters (advanced)
    beam_size: int = 5
    patience: float = 1.0
    temperature: float = 0.0  # 0 = greedy, >0 = sampling
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4

    # Performance
    threads: int = 4  # CPU threads per worker

    # Diarization
    diarize: bool = False
    diarize_model: str = "pyannote/speaker-diarization-3.1"
    num_speakers: int | None = None  # exact speaker count (overrides min/max)
    min_speakers: int | None = None
    max_speakers: int | None = None

    # --- Translation defaults ---
    target_language: str | None = None  # comma-separated for multiple
    gemini_model: str = "gemini-flash-latest"
    translate_batch_size: int | None = None  # None = use package default
    description: str | None = None
    start_line: int | None = None

    # --- Export defaults ---
    export_formats: str = "srt"  # comma-separated
    bilingual: bool = False
    bilingual_format: str = "srt"
    bilingual_order: str = "original-first"  # or "translated-first"
    txt_style: str = "lines"  # "lines" or "paragraph"

    # --- Output ---
    output_dir: str | None = None

    # --- Behaviour ---
    verbose: bool = False
    quiet: bool = False
    log_file: str | None = None

    model_config = {
        "env_prefix": "SUBTORU_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @classmethod
    def load(cls, config_path: str | None = None, **overrides: Any) -> SubxtConfig:
        """Build a config by merging TOML file, env vars, and explicit overrides.

        ``overrides`` come from CLI flags — they have highest precedence per §7.
        """
        # 1. Load TOML defaults
        path = Path(config_path) if config_path else _find_config_file()
        toml_values = _load_toml_config(path)

        # 2. Let pydantic-settings handle env vars on top of toml defaults
        # We pass toml values as kwargs (lower than env, which pydantic-settings reads directly)
        merged = {**toml_values}

        # 3. CLI overrides (highest precedence) — drop None values so they don't mask lower layers
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v

        # Handle canonical env vars that don't follow the SUBXT_ prefix
        for env_name, field_name in [
            ("GEMINI_API_KEY", "gemini_api_key"),
            ("GEMINI_API_KEY2", "gemini_api_key2"),
            ("HF_TOKEN", "hf_token"),
        ]:
            if field_name not in merged and os.environ.get(env_name):
                merged[field_name] = os.environ[env_name]

        return cls(**merged)

    def resolve_device(self) -> str:
        """Return 'cuda' or 'cpu', auto-detecting if needed."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def resolve_compute_type(self) -> str:
        """Return the compute type, defaulting based on device."""
        if self.compute_type:
            return self.compute_type
        return "float16" if self.resolve_device() == "cuda" else "int8"

    def get_target_languages(self) -> list[str]:
        """Parse comma-separated target language string into a list."""
        if not self.target_language:
            return []
        return [lang.strip() for lang in self.target_language.split(",") if lang.strip()]

    def get_export_formats(self) -> list[str]:
        """Parse comma-separated export formats into a list."""
        return [fmt.strip().lower() for fmt in self.export_formats.split(",") if fmt.strip()]
