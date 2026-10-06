# Subtoru

**Subtitle generation & translation CLI** — download, transcribe, translate, export.

`subtoru` automates end-to-end subtitle creation: optionally download source media,
transcribe it with [whisperX](https://github.com/m-bain/whisperX), translate the
transcript with [gemini-srt-translator](https://github.com/MaKTaiL/gemini-srt-translator),
and export the result into whichever subtitle/text formats you need.

```
[optional] URL ──yt-dlp──▶ media file
                              │
                    media/audio ──whisperX──▶ source-language subtitles (+ word timings)
                                                      │
                                      gemini-srt-translator ──▶ translated subtitles
                                                      │
                                      [optional] ──▶ export to N formats
                                                 └─▶ bilingual file
```

## Installation

### Install

```bash
pip install subtoru
```

> **Note:** whisperX requires PyTorch with CUDA for GPU acceleration. Follow
> [whisperX's install guide](https://github.com/m-bain/whisperX#setup) if you
> need help setting up CUDA/cuDNN. CPU fallback works but is slow.

### Development

```bash
git clone https://github.com/Iceyu29/subtoru.git
cd subtoru
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quickstart

### Full pipeline — URL to translated subtitles

```bash
# Transcribe + translate a YouTube video to French
subtoru run "https://youtube.com/watch?v=VIDEO_ID" -t French

# Local file, multiple languages, extra formats
subtoru run interview.mp4 -t "French,Spanish,Japanese" -f "srt,vtt,json" --bilingual
```

### Individual stages

```bash
# Just download
subtoru download "https://youtube.com/watch?v=VIDEO_ID" --audio-only

# Just transcribe
subtoru transcribe audio.wav --whisper-model large-v3 --diarize --hf-token $HF_TOKEN

# Just translate
subtoru translate transcript.srt -t French --gemini-api-key $GEMINI_API_KEY

# Just export
subtoru export transcript.srt -f "vtt,json,lrc,csv,txt" --translated-srt transcript.fr.srt -t fr
```

### List available Gemini models

```bash
subtoru list-models
```

## Configuration

### Precedence (highest wins)

1. Explicit CLI flag
2. Environment variable
3. Config file (`subtoru.toml` or `[tool.subtoru]` in `pyproject.toml`)
4. Built-in default

### Environment variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key for translation |
| `GEMINI_API_KEY2` | Secondary key for quota rotation |
| `HF_TOKEN` | Hugging Face token (required for diarization) |

A `.env` file in the working directory is loaded automatically.

### Obtaining API Keys

**Gemini API Key (`GEMINI_API_KEY`)**
Required for translation via gemini-srt-translator.
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API key** and generate a new key.
*Note: The free tier has generous limits, but may get rate-limited on very large files. You can provide a `GEMINI_API_KEY2` to rotate quotas.*

**Hugging Face Token (`HF_TOKEN`)**
Required *only* if you enable speaker diarization (`--diarize`).
1. Go to [Hugging Face Settings > Tokens](https://huggingface.co/settings/tokens).
2. Create a new token with **Read** access.
3. You must also accept the user agreements for these two models:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

### Config file (`subtoru.toml`)

```toml
whisper_model = "large-v3"
export_formats = "srt,vtt,json"
target_language = "French"
bilingual = true
output_dir = "./subtitles"
```

## Command Reference

### `subtoru run` — Full pipeline

| Flag | Purpose | Default |
|---|---|---|
| `INPUT` (positional) | URL or local media file | required |
| `--target-language, -t` | Target language(s), comma-separated | required unless `--no-translate` |
| `--skip-download` | Treat INPUT as local file | auto-detected |
| `--no-translate` | Stop after transcription | off |
| `--export-formats, -f` | Output formats (comma-separated) | `srt` |
| `--bilingual` | Emit combined original+translated file | off |
| `--bilingual-format` | `srt` or `ass` | `srt` |
| `--bilingual-order` | `original-first` or `translated-first` | `original-first` |
| `--output-dir, -o` | Root output directory | `./subtoru-output/<slug>/` |
| `--resume` | Skip already-completed stages | off |
| `--dry-run` | Print resolved plan, don't execute | off |
| `--config` | Path to config file | `./subtoru.toml` |
| `-v / --verbose` | Debug output | off |
| `-q / --quiet` | Warnings only | off |
| `--log-file` | Write log to file | none |

### `subtoru download <URL>`

| Flag | Purpose | Default |
|---|---|---|
| `--audio-only` | Extract audio only | off |
| `--format` | yt-dlp format selector | `bestaudio/best` |
| `--cookies-from-browser` | Browser for cookie extraction | none |
| `--output-dir, -o` | Output directory | working dir |

### `subtoru transcribe <MEDIA>`

| Flag | Purpose | Default |
|---|---|---|
| `--whisper-model` | Model name | `large-v2` |
| `--source-language` | ISO code or omit for auto-detect | auto |
| `--device` | `cuda` or `cpu` | auto-detect |
| `--compute-type` | `float16` / `float32` / `int8` | auto |
| `--whisper-batch-size` | Batched inference size | `16` |
| `--no-align` | Skip word-level alignment | off |
| `--align-model` | Override alignment model | auto |
| `--task` | `transcribe` or `translate` (built-in translation to English) | `transcribe` |
| `--initial-prompt` | Domain vocabulary or context prompt | none |
| `--hotwords` | Rare/technical terms hint | none |
| `--vad-onset` | VAD onset threshold (lower = more sensitive) | `0.5` |
| `--vad-offset` | VAD offset threshold | `0.363` |
| `--chunk-size` | VAD chunk size in seconds | `5` |
| `--suppress-numerals`| Suppress numeric/currency symbols during alignment | off |
| `--threads` | CPU threads per worker | `4` |
| `--diarize` | Enable speaker diarization | off |
| `--hf-token` | Hugging Face token | env `HF_TOKEN` |
| `--diarize-model` | Custom diarization model name | `pyannote...` |
| `--num-speakers` | Exact speaker count | none |
| `--min-speakers` | Diarization hint | none |
| `--max-speakers` | Diarization hint | none |

*Note: Many more advanced whisperX options (e.g. `beam_size`, `temperature`, `patience`, `interpolate_method`, `return_char_alignments`, `highlight_words`, `max_line_width`) can be configured via `subtoru.toml`.*

### `subtoru translate <SRT>`

| Flag | Purpose | Default |
|---|---|---|
| `--target-language, -t` | Target language | required |
| `--gemini-api-key` | Gemini API key | env `GEMINI_API_KEY` |
| `--gemini-model` | Gemini model name | package default |
| `--translate-batch-size` | Cues per batch | package default |
| `--description` | Context hint for translation | none |
| `--start-line` | Resume interrupted translation | none |

### `subtoru export <SRT>`

| Flag | Purpose | Default |
|---|---|---|
| `--formats, -f` | Comma-separated output formats | `srt,vtt` |
| `--translated-srt` | Path to translated SRT | none |
| `--target-language, -t` | Language code for naming | none |
| `--bilingual` | Produce bilingual file | off |
| `--bilingual-format` | `srt` or `ass` | `srt` |
| `--bilingual-order` | Which language on top | `original-first` |
| `--txt-style` | `lines` or `paragraph` | `lines` |

### `subtoru list-models`

Lists available Gemini models via gemini-srt-translator.

## Format Support Matrix

SRT is always produced. Everything else is opt-in via `--export-formats`.

### Core formats

| Format | Extension | How | Notes |
|---|---|---|---|
| SubRip | `.srt` | pysubs2 | Always generated |
| WebVTT | `.vtt` | pysubs2 | Web/HTML5 captions |
| JSON | `.json` | custom | Original + translated + speaker + word timings |
| Plain text | `.txt` | custom | Clean transcript, no timestamps |
| LRC | `.lrc` | custom | Karaoke timing; enhanced word-level when available |
| CSV | `.csv` | custom | Proofreading spreadsheet |
| Bilingual SRT | `.srt` | custom | Two lines per cue |
| Bilingual ASS | `.ass` | custom | Styled per-language subtitles |

### Bonus formats (pysubs2)

| Format | Extension | Notes |
|---|---|---|
| ASS | `.ass` | Advanced SubStation Alpha |
| SSA | `.ssa` | SubStation Alpha (legacy) |
| MicroDVD | `.sub` | Frame-based (needs FPS) |
| MPL2 | `.mpl2` | Simple time-based |
| TMP | `.tmp` | Start-time-only |
| TTML/DFXP | `.ttml` | Broadcast/streaming |
| SAMI | `.smi` | Windows Media legacy |
| SubViewer | `.sbv` | YouTube legacy caption format |

## Architecture

```
subtoru/
├── cli.py                 # Typer app; all subcommands
├── config.py              # Config file + env var loading
├── pipeline.py            # Orchestrates run(); manifest; --resume
├── downloader.py          # yt-dlp Python API wrapper
├── transcriber.py         # whisperX Python API wrapper
├── translator.py          # gemini-srt-translator Python API wrapper
├── models.py              # Cue, WordTiming, RunManifest dataclasses
├── exporters/
│   ├── base.py            # Exporter protocol + format registry
│   ├── pysubs2_formats.py # SRT/VTT/ASS/SSA/MicroDVD/MPL2/TMP/TTML/SAMI
│   ├── lrc.py             # LRC + enhanced word-level LRC
│   ├── json_export.py     # Rich JSON with metadata + words
│   ├── txt.py             # Plain transcript
│   ├── csv_export.py      # Proofreading spreadsheet
│   ├── sbv.py             # YouTube SubViewer
│   └── bilingual.py       # Merged source + translated
└── utils/
    ├── logging.py         # Rich logging, progress bars, exit codes
    └── naming.py          # Slugify, path derivation, URL detection
```

### Key design decisions

- **Python APIs, not subprocess** — whisperX, gemini-srt-translator, and yt-dlp are all
  called via their Python interfaces for structured data access.
- **Speaker labels kept out of translation** — `Cue.speaker` is tracked separately and
  never baked into the text sent to gemini-srt-translator, which would mangle labels.
- **Word-level data preserved** — The raw whisperX JSON is saved alongside the SRT so
  exporters that need `Cue.words` (enhanced LRC, JSON) can access it.
- **Lazy imports for heavy deps** — whisperX and gemini-srt-translator are imported only
  when needed, so users who only export/convert don't require GPU libraries.

## Resumability

Every `run` writes a `manifest.json` to the output directory after each stage.
With `--resume`:

- Re-downloads only if the media file is missing
- Re-transcribes only if the SRT/JSON is missing
- Re-translates only for languages not already done
- Re-exports only formats not already produced

## Credits

Subtoru stands on the shoulders of giants. It orchestrates several incredible open-source projects to build its pipeline:

- [**whisperX**](https://github.com/m-bain/whisperX) by Max Bain — Core engine for fast speech recognition, word-level alignment, and diarization.
  - Powered under the hood by [**faster-whisper**](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN) and [**pyannote.audio**](https://github.com/pyannote/pyannote-audio) (Hervé Bredin).
- [**gemini-srt-translator**](https://github.com/MaKTaiL/gemini-srt-translator) by MaKTaiL — Robust LLM-based subtitle translation using Google's Gemini models.
- [**yt-dlp**](https://github.com/yt-dlp/yt-dlp) — The definitive command-line media downloader for fetching audio/video.
- [**pysubs2**](https://github.com/tkarabela/pysubs2) by Tomas Karabela — Excellent Python library for parsing, generating, and converting subtitle formats (SRT, VTT, ASS, etc.).
- [**FFmpeg**](https://ffmpeg.org/) — The universal multimedia framework powering the underlying audio extraction and processing.
- [**Typer**](https://github.com/tiangolo/typer) & [**Rich**](https://github.com/Textualize/rich) — For providing the elegant CLI interface and beautiful terminal progress bars.

## License

Subtoru is open-source software licensed under the [MIT License](LICENSE).
