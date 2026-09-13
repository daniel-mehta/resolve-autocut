# Resolve AutoCut

**AI-assisted video cleanup for DaVinci Resolve**

Local media inference, optimized for Apple Silicon.

## Overview

Resolve AutoCut is a macOS application that analyzes video/audio files to detect and remove filler words ("uh", "um", "hmm") automatically. It generates:

- An **FCPXML timeline** intended for import into DaVinci Resolve
- **SRT subtitles** with timestamps adjusted to match the edited timeline
- **JSON analysis** for audit and reproducibility

Interview media, extracted audio, transcripts, and analysis remain local. No
interview content is uploaded to a cloud inference service. Internet access may
be required to install dependencies and download model files.

## Current MVP Status

This is the first Minimum Viable Product (MVP) version. It implements:

- Local file selection via GUI or command-line
- Media inspection with FFprobe
- Audio extraction for analysis
- UHM (Uh-and-Um Detector) filler detection
- MLX Whisper transcription with word-level timestamps
- Proposed filler review with enable/disable controls
- Cut interval generation with configurable padding
- Conservative cut padding (default 50ms)
- Frame-aligned, ripple-delete FCPXML 1.9 timeline export (manual Resolve import validation pending)
- Editable SRT caption export
- Caption timestamp remapping after cuts
- JSON analysis/export
- Basic usable GUI (tkinter)
- Automated invariant and regression tests
- Clear documentation

## Requirements

### Hardware

- **macOS with Apple Silicon (M1 or newer)**
- **32 GB RAM** recommended for longer videos
- **10 GB free disk space** for model caching

### Software

- **Python 3.12 or 3.13**
- **FFmpeg / FFprobe** (required)
- **uv** (recommended for dependency management)

### Python Dependencies

See `pyproject.toml` for the full list. Key dependencies:

- `numpy` - Scientific computing
- `onnxruntime` - For running UHM ONNX model
- `huggingface-hub` - For downloading UHM model
- `mlx` + `mlx-whisper` - For Apple Silicon-optimized transcription
- `scipy` + `soundfile` - Audio processing
- `tkinter` - GUI (included with Python)

## Installation

### Quick Start with uv

```bash
git clone https://github.com/daniel-mehta/resolve-autocut.git
cd resolve-autocut
uv venv --python 3.12
source .venv/bin/activate
uv sync --all-extras --locked
brew install ffmpeg
```

### Without uv (using pip)

```bash
git clone https://github.com/daniel-mehta/resolve-autocut.git
cd resolve-autocut
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
brew install ffmpeg
```

## How Models Are Downloaded

The first time you run analysis, the application will automatically download:

1. **UHM model** (`uhm-web-fp16.onnx`) from Hugging Face Hub (~51 MB)
2. **MLX Whisper model** (default: `base`) from Hugging Face (~140 MB)

Models are cached in `~/.cache/huggingface/hub/`. Review the UHM source-available
license before use; commercial licensing may be required at scale.

The built-in Whisper choices use public MLX-converted repositories and do not
require an `HF_TOKEN`:

| Choice | Repository |
| --- | --- |
| `tiny` | `mlx-community/whisper-tiny-mlx` |
| `base` | `mlx-community/whisper-base-mlx` |
| `small` | `mlx-community/whisper-small-mlx` |
| `medium` | `mlx-community/whisper-medium-mlx` |

Programmatic callers may instead provide a full Hugging Face repository ID for
an MLX-compatible model. Private or gated repositories retain Hugging Face's
normal authentication requirements and errors.

## How to Launch the Program

### GUI Mode

```bash
uv run python -m resolve_autocut.gui
```

### Command-Line Mode

```python
from resolve_autocut import ResolveAutoCut, run_analysis_pipeline

result = run_analysis_pipeline(
    media_path="/path/to/your/video.mp4",
    output_dir="/path/to/output",
    cut_padding=0.05,
    whisper_model="base"
)
```

## How to Use (GUI)

1. **Select Media File**: Click "Browse" to select your video or audio file
2. **Configure Options**: Adjust cut padding, Whisper model, and confidence threshold (default 0.75)
3. **Click "Analyze"**: Wait for processing to complete
4. **Review Detections**: Enable/disable individual cuts in the Review tab
5. **Export Results**: Export FCPXML, SRT, and JSON files

## How to Import into DaVinci Resolve

1. Open DaVinci Resolve
2. Go to **File → Import → Timeline**
3. Select the `.fcpxml` file
4. The timeline will appear with your media and cuts applied

## Privacy

- Interview media, extracted audio, transcripts, and analysis stay on your Mac
- No interview content is sent to a cloud inference API
- Dependency and model downloads contact package registries and Hugging Face
- Generated JSON includes the local source-media path; exports are gitignored under `outputs/`

## UHM Attribution

Filler detection powered by **UHM by Desert Ant Labs**.

- Repository: https://huggingface.co/desert-ant-labs/uhm
- Homepage: https://desertant.com/
- License: Desert Ant Labs Source-Available License 1.0
- License URL: https://license.desertant.com/1.0

See `THIRD_PARTY_NOTICES.md` for complete licensing information.

## MIT License

The Resolve AutoCut source code is MIT licensed. Third-party models and libraries retain their original licenses.

## Verification status

The UHM ONNX integration has been exercised against local interview audio. Its
actual schema is a 16 kHz mono `float32 [1, 480000]` input and a
`float32 [1, 1499, 6]` softmax output. The pipeline advances each 30-second
input by its 29.98-second output coverage (20 ms of shared input context), with
a padded tail; it does not invoke the model once per output frame.

MLX Whisper 0.4.3 has also been exercised locally using its supported
`transcribe(..., path_or_hf_repo=..., word_timestamps=True)` API. Resolve
AutoCut requires timestamped word results and does not invent them from
segment timings.

The complete pipeline was run on both a non-reencoded 60-second excerpt and the
full 1,913.024-second gitignored interview sample with the real UHM ONNX model
and local MLX Whisper `tiny`. The full validation produced 6 conservative
0.75-threshold detections, 5,362 timestamped words, 702 captions, and 6 cuts.
The padded proposals span 4.98 seconds; conservative alignment removes 73
complete 16 fps frames (4.5625 seconds). Generated XML, SRT, and JSON passed
programmatic invariant checks against the same snapped cuts. Manual import into
the target DaVinci Resolve version remains required; FCPXML generation is
verified, Resolve acceptance is not.

## Known Limitations

- No B-roll generation
- No speaker diarization
- No multicamera editing
- DaVinci Resolve GUI import has not yet been manually verified
- GUI cancellation is cooperative between model steps; an in-progress Whisper
  call cannot be interrupted safely

## Development

```bash
# Run tests
uv run python -m pytest

# Run specific test
uv run python -m pytest tests/test_intervals.py -v
```

## Project Structure

```
resolve-autocut/
├── LICENSE
├── README.md
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── src/resolve_autocut/
│   ├── __init__.py
│   ├── app.py
│   ├── models.py
│   ├── media.py
│   ├── uhm.py
│   ├── transcription.py
│   ├── intervals.py
│   ├── captions.py
│   ├── timeline.py
│   └── gui.py
└── tests/
```

---

**Resolve AutoCut** - Version 0.1.0 | MIT License
