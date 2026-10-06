"""whisperX Python API wrapper — transcription, alignment, and diarization.

Uses whisperX's Python API (not subprocess) for:
- Speech-to-text transcription with batched inference
- Forced word-level alignment via wav2vec2
- Optional speaker diarization via pyannote

The whisperX import is deferred so that users who only need translation/export
don't require the heavy GPU dependencies.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pysubs2

from subtoru.config import SubxtConfig
from subtoru.models import Cue, WordTiming
from subtoru.utils.logging import logger, print_warning


def transcribe(
    audio_path: str | Path,
    *,
    config: SubxtConfig,
    transcribe_callback: Callable[[float], None] | None = None,
    align_callback: Callable[[float], None] | None = None,
    diarize_callback: Callable[[float], None] | None = None,
    verbose_display: bool = False,
) -> tuple[list[Cue], dict]:
    """Transcribe audio using whisperX.

    Args:
        audio_path: Path to audio/media file.
        config: Subtoru configuration.
        transcribe_callback: Progress callback for transcription (0-100%).
        align_callback: Progress callback for alignment (0-100%).
        diarize_callback: Progress callback for diarization (0-100%).
        verbose_display: If True, display each transcribed segment as it's
            processed (live transcript via Rich console).

    Returns:
        A tuple of (list of Cue objects, raw whisperX result dict).

    Raises:
        ImportError: If whisperX is not installed.
    """
    try:
        import whisperx
    except ImportError as e:
        raise ImportError("whisperX is not installed. Install with: pip install subtoru") from e

    audio_path_str = str(audio_path)

    device = config.resolve_device()
    if device == "cpu":
        print_warning("No CUDA GPU detected — falling back to CPU. " "Transcription will be slow.")

    compute_type = config.resolve_compute_type()

    # Build ASR options dict from config (decoding parameters)
    asr_options: dict = {}
    if config.initial_prompt:
        asr_options["initial_prompt"] = config.initial_prompt
    if config.hotwords:
        asr_options["hotwords"] = config.hotwords
    if config.suppress_numerals:
        asr_options["suppress_numerals"] = True
    if config.beam_size != 5:
        asr_options["beam_size"] = config.beam_size
    if config.patience != 1.0:
        asr_options["patience"] = config.patience
    if config.temperature != 0.0:
        asr_options["temperatures"] = [config.temperature]
    if config.no_speech_threshold != 0.6:
        asr_options["no_speech_threshold"] = config.no_speech_threshold
    if config.compression_ratio_threshold != 2.4:
        asr_options["compression_ratio_threshold"] = config.compression_ratio_threshold

    # VAD options
    vad_options = {
        "vad_onset": config.vad_onset,
        "vad_offset": config.vad_offset,
    }

    logger.info(
        f"Loading whisperX model '{config.whisper_model}' " f"on {device} ({compute_type})..."
    )
    model = whisperx.load_model(
        config.whisper_model,
        device,
        compute_type=compute_type,
        language=config.source_language,
        asr_options=asr_options if asr_options else None,
        vad_method=config.vad_method,
        vad_options=vad_options,
        task=config.task,
        threads=config.threads,
    )

    logger.info(f"Loading audio from {audio_path_str}...")
    audio = whisperx.load_audio(audio_path_str)

    logger.info(
        f"Transcribing audio (task={config.task}, " f"batch_size={config.whisper_batch_size})..."
    )

    # When verbose_display is enabled, intercept whisperX's print() calls
    # to show live transcript text through Rich console
    _original_print = None
    if verbose_display:
        import builtins

        from subtoru.utils.progress import make_verbose_print_interceptor

        _original_print = builtins.print
        builtins.print = make_verbose_print_interceptor()

    try:
        result = model.transcribe(
            audio,
            batch_size=config.whisper_batch_size,
            chunk_size=config.chunk_size,
            progress_callback=transcribe_callback,
            verbose=verbose_display,
        )
    finally:
        if _original_print is not None:
            import builtins

            builtins.print = _original_print

    # Word-level alignment (unless --no-align)
    if not config.no_align and result.get("segments"):
        logger.info("Aligning transcription (word-level timing)...")
        align_kwargs: dict = {
            "language_code": result["language"],
            "device": device,
        }
        if config.align_model:
            align_kwargs["model_name"] = config.align_model
        model_a, metadata = whisperx.load_align_model(**align_kwargs)
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            interpolate_method=config.interpolate_method,
            return_char_alignments=config.return_char_alignments,
            progress_callback=align_callback,
        )

    # Speaker diarization (if --diarize)
    if config.diarize:
        if not config.hf_token:
            print_warning(
                "Diarization requested but HF_TOKEN is not set. "
                "You need a Hugging Face token with "
                "pyannote/speaker-diarization access."
            )

        # Import from whisperx.diarize (not top-level whisperx)
        from whisperx.diarize import DiarizationPipeline

        logger.info(f"Running speaker diarization " f"(model: {config.diarize_model})...")
        diarize_pipeline = DiarizationPipeline(
            model_name=config.diarize_model,
            token=config.hf_token,
            device=device,
        )
        diarize_segments = diarize_pipeline(
            audio,
            num_speakers=config.num_speakers,
            min_speakers=config.min_speakers,
            max_speakers=config.max_speakers,
            progress_callback=diarize_callback,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

    # Convert whisperX segments → our Cue model
    cues = _segments_to_cues(result.get("segments", []))
    logger.info(f"Transcription complete: {len(cues)} cues")
    return cues, result


def _segments_to_cues(segments: list[dict]) -> list[Cue]:
    """Convert whisperX segment dicts to Cue objects.

    Speaker labels go into Cue.speaker (NOT into the text) so that the
    translator never sees them — see §5 gotcha in the spec.
    """
    cues = []
    for i, seg in enumerate(segments):
        words = []
        for w in seg.get("words", []):
            # whisperX sometimes emits words without timing (failed alignment)
            if "start" in w and "end" in w:
                words.append(
                    WordTiming(
                        word=w.get("word", ""),
                        start=w["start"],
                        end=w["end"],
                        score=w.get("score"),
                    )
                )
        cues.append(
            Cue(
                index=i + 1,
                start=seg["start"],
                end=seg["end"],
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker"),  # kept separate from text
                words=words,
            )
        )
    return cues


def cues_to_srt(cues: list[Cue], output_path: Path) -> Path:
    """Write cues to an SRT file using pysubs2.

    Speaker labels are NOT embedded in the subtitle text — the SRT is
    the clean transcript that gets handed to the translator.
    """
    subs = pysubs2.SSAFile()
    for cue in cues:
        event = pysubs2.SSAEvent(
            start=cue.start_ms,
            end=cue.end_ms,
            text=cue.text,
        )
        # Store speaker in the "Name" field (ASS metadata, stripped in SRT)
        if cue.speaker:
            event.name = cue.speaker
        subs.append(event)
    subs.save(str(output_path))
    logger.debug(f"Wrote SRT: {output_path}")
    return output_path


def save_raw_json(result: dict, output_path: Path) -> Path:
    """Save the raw whisperX result dict as JSON for word-level exports."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.debug(f"Wrote raw JSON: {output_path}")
    return output_path


def load_srt_to_cues(srt_path: Path) -> list[Cue]:
    """Load an existing SRT file into a list of Cue objects.

    This is the shared utility for turning any SRT (source or translated)
    into our canonical Cue representation.
    """
    subs = pysubs2.load(str(srt_path))
    cues = []
    for i, event in enumerate(subs):
        cues.append(
            Cue(
                index=i + 1,
                start=event.start / 1000.0,
                end=event.end / 1000.0,
                text=event.text,
                speaker=event.name if event.name else None,
            )
        )
    return cues
