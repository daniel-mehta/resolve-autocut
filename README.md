# Resolve AutoCut

**AI-assisted video cleanup for DaVinci Resolve**

Local, open-source, Apple Silicon optimized.

## Overview

Resolve AutoCut is a macOS application that analyzes video/audio files to detect and remove filler words ("uh", "um", "hmm") automatically. It generates:

- An **FCPXML timeline** that can be imported directly into DaVinci Resolve
- **SRT subtitles** with timestamps adjusted to match the edited timeline
- **JSON analysis** for audit and reproducibility

The application runs entirely locally on your Mac - no cloud APIs, no watermarks, no paid services.

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
- Resolve-compatible FCPXML timeline export
- Editable SRT caption export
- Caption timestamp remapping after cuts
- JSON analysis/export
- Basic usable GUI (tkinter)
- Automated tests (109 passing)
- Clear documentation

## Requirements

### Hardware

- **macOS** (tested on macOS with Apple Silicon)
- **Apple Silicon M1/M2/M3** or Intel Mac (Apple Silicon recommended for MLX)
- **32 GB RAM** recommended for longer videos
- **10 GB free disk space** for model caching

### Software

- **Python 3.12** (required)
- **FFmpeg / FFprobe** (required)
- **uv** (recommended for dependency management)

### Python Dependencies

See `pyproject.toml` for the full list. Key dependencies:

- `numpy` - Scientific computing
- `onnxruntime` - For running UHM ONNX model
- `huggingface-hub` - For downloading UHM model
- `mlx` + `mlx-whisper` - For Apple Silicon-optimized transcription
- `scipy` + `soundfile` - Audio processing
- `pydub` - Optional audio manipulation
- `tkinter` - GUI (included with Python)

## Installation

### Quick Start with uv

```bash
git clone https://github.com/your-org/resolve-autocut.git
cd resolve-autocut
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
brew install ffmpeg
```

### Without uv (using pip)

```bash
git clone https://github.com/your-org/resolve-autocut.git
cd resolve-autocut
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
brew install ffmpeg
```

## How Models Are Downloaded

The first time you run analysis, the application will automatically download:

1. **UHM model** (`uhm-web-fp16.onnx`) from Hugging Face Hub (~80 MB)
2. **MLX Whisper model** (default: `base`) from Hugging Face (~140 MB)

Models are cached in `~/.cache/huggingface/hub/`. You may need to accept licenses on first use.

## How to Launch the Program

### GUI Mode

```bash
python -m resolve_autocut.gui
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
2. **Configure Options**: Adjust cut padding, whisper model, confidence threshold
3. **Click "Analyze"**: Wait for processing to complete
4. **Review Detections**: Enable/disable individual cuts in the Review tab
5. **Export Results**: Export FCPXML, SRT, and JSON files

## How to Import into DaVinci Resolve

1. Open DaVinci Resolve
2. Go to **File → Import → Timeline**
3. Select the `.fcpxml` file
4. The timeline will appear with your media and cuts applied

## Privacy

- All processing happens on your Mac
- No data leaves your machine
- No cloud APIs
- No watermarks

## UHM Attribution

Filler detection powered by **UHM by Desert Ant Labs**.

- Repository: https://huggingface.co/desert-ant-labs/uhm
- Homepage: https://desertant.com/
- License: Desert Ant Labs Source-Available License 1.0
- License URL: https://license.desertant.com/1.0

See `THIRD_PARTY_NOTICES.md` for complete licensing information.

## MIT License

The Resolve AutoCut source code is MIT licensed. Third-party models and libraries retain their original licenses.

## Known Limitations

- No B-roll generation
- No speaker diarization
- No multicamera editing
- FCPXML compatibility tested with basic imports
- Only `base` Whisper model tested extensively

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_intervals.py -v
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
