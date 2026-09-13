# Resolve AutoCut

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![macOS](https://img.shields.io/badge/platform-macOS-000000?logo=apple)
![Python 3.12--3.13](https://img.shields.io/badge/python-3.12--3.13-3776AB?logo=python&logoColor=white)

Resolve AutoCut is a local AI-assisted cleanup tool for DaVinci Resolve. It detects selected filler words, creates reviewable proposed cuts, and exports a ripple-edited FCPXML timeline plus editable SRT captions.

It is for editors who want a faster first pass over interview-style footage without sending that footage to a cloud inference service. Resolve AutoCut processes media locally on Apple Silicon and does not re-encode the original media. Resolve AutoCut itself has no subscription and adds no watermark. It proposes edits for you to review; it does not blindly destructively edit your source recording.

> This is an early v0.1.0 release. The core pipeline has been validated on real interview footage, including a roughly 32-minute source, and by automated regression tests. Additional media formats, DaVinci Resolve configurations, and edge cases may still expose bugs.

## What it does — and does not do

- Detects `uh`, `um`, and `hmm` with local UHM inference.
- Produces a reviewable list of detections. You can enable or disable each proposed cut before export.
- Transcribes locally with MLX Whisper and exports captions retimed to the edited timeline.
- Exports FCPXML for a ripple-edited Resolve timeline, SRT captions, and JSON analysis.
- Does **not** create B-roll, identify speakers, edit multicamera footage, or promise to find every filler correctly.

Removing spoken audio can create an unnatural cut. The default confidence threshold is deliberately conservative; review every proposed edit before using it in a final project.

## Requirements

- macOS on Apple Silicon (M1 or newer)
- Python 3.12 or 3.13
- FFmpeg and FFprobe
- [`uv`](https://docs.astral.sh/uv/) recommended for installation and running

32 GB of RAM is recommended for longer media. This release does not claim support for Intel Macs, Windows, or Linux.

## Install and first launch

```bash
git clone https://github.com/daniel-mehta/resolve-autocut.git
cd resolve-autocut
brew install ffmpeg
uv sync --all-extras --locked
uv run python -m resolve_autocut.gui
```

If you do not use Homebrew, install FFmpeg/FFprobe through another trusted macOS distribution and make both commands available on your `PATH`.

For a pip-based setup, create a Python 3.12 or 3.13 virtual environment, install the project with `pip install -e .`, install FFmpeg/FFprobe separately, then run `python -m resolve_autocut.gui`.

## Models and downloads

On first analysis, Resolve AutoCut downloads the UHM detector and the selected MLX Whisper model from Hugging Face, then uses its local cache on later runs. The built-in model choices are public repositories and do not require a Hugging Face token:

| Choice | Repository |
| --- | --- |
| `tiny` | `mlx-community/whisper-tiny-mlx` |
| `base` | `mlx-community/whisper-base-mlx` |
| `small` | `mlx-community/whisper-small-mlx` |
| `medium` | `mlx-community/whisper-medium-mlx` |

Advanced Python callers may provide another MLX-compatible Hugging Face repository ID. Private or gated repositories may require normal Hugging Face authentication.

## Use the app

1. Launch the app and select an interview video or audio file.
2. Choose a Whisper model, optional cut padding, and a confidence threshold. The default is `0.75` to favor more conservative proposals.
3. Click **Analyze**. The app extracts audio locally, detects fillers, and creates word-timestamped captions.
4. In the **Review** tab, inspect every proposed filler and enable or disable the corresponding cut. Adjust padding if needed and re-check the selection.
5. Export FCPXML, SRT, JSON, or all three. A typical set of names is `interview-autocut.fcpxml`, `interview-autocut.srt`, and `interview-autocut.json`.

Cancellation is cooperative: the app can stop between model operations, but it may not stop immediately while an active Whisper call is running.

## Import into DaVinci Resolve

1. In DaVinci Resolve, choose **File → Import → Timeline**.
2. Select `interview-autocut.fcpxml` and complete Resolve's media-relink prompt if it appears.
3. Review the imported timeline for relinking, picture/audio sync, cut placement, and duration before continuing your edit.
4. Import `interview-autocut.srt` through Resolve's subtitle import workflow, then review and edit captions as needed.

FCPXML export uses frame-aligned ripple timing, and caption timestamps are remapped to the same edited timeline. Resolve AutoCut has not been manually validated against every Resolve version or configuration, so an import review is always part of the workflow.

## Privacy

Interview media, extracted audio, transcripts, and analysis remain local. Resolve AutoCut does not upload interview content to a cloud inference service. Internet access is required for initial dependency and model downloads unless they are already cached.

The exported JSON may contain the local source-media path and transcript-derived analysis. Treat exports and logs as project material, and share them only when appropriate.

## Verification and limitations

The automated suite covers interval calculation, frame-aligned ripple timing, caption remapping, SRT generation, and model/repository contracts. The local UHM and MLX Whisper pipeline has also completed real-media runs, including a roughly 32-minute interview source and a 60-second run with the default `base` Whisper model.

Known limitations:

- This is an early v0.1.0 release; media-format and Resolve compatibility are not guaranteed.
- Filler detection focuses on `uh`, `um`, and `hmm`; it can miss fillers or produce false positives.
- Review proposed cuts before using an exported timeline.
- No B-roll generation, speaker diarization, or multicamera editing.
- Cancellation may wait for an active Whisper call to finish.

## License and third-party notices

Resolve AutoCut-authored code is licensed under the [MIT License](LICENSE). Filler detection is powered by UHM by Desert Ant Labs, which is **not** MIT licensed and is not described here as open source; it is offered under the Desert Ant Labs Source-Available License 1.0. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and third-party terms.

## Problems or feedback

GitHub Issues are the preferred support channel. If you encounter a bug, unexpected cut, import problem, or unsupported media file, please [open an issue](https://github.com/daniel-mehta/resolve-autocut/issues/new) with enough information to reproduce it.

Please do not upload private recordings or sensitive media to a public issue. Logs, error messages, Resolve version, macOS version, media properties, and a minimal reproducible example are usually sufficient.

You can also reach me on [LinkedIn](https://www.linkedin.com/in/dan-mehta/).

## Contributing

Contributions and well-scoped bug reports are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.
